import asyncio
import logging
import re
import uuid

from typing import List, Literal, Optional, Union

from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableSequence

from langfuse.langchain import CallbackHandler

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agent.utils.parsers.output_parse import StructureJsonOutputParser
from vector_db import ElasticVectorDB, QdrantVectorDB
from .prompt_template import (
    FindRelevantChunkTemplate,
    GetNewChunkTitleSummaryTemplate,
    UpdateChunkTitleSummaryTemplate,
)
from .proposition_indexing import PropositionIndexingPipeline
from .state import ChunkingState
from ..base_chunker import BaseChunker
from ..contextual_chunking import ContextualChunker

logger = logging.getLogger(__name__)


class AgenticChunker(BaseChunker):
    """
    An asynchronous intelligent document chunking system that uses LLMs to
    create semantically coherent chunks

    This chunker implements an agentic approach to document segmentation by:
    1. Extracting propositions (atomic facts) from documents
    2. Dynamically organizing propositions into semantically related chunks
    3. Generating and updating chunk titles and summaries as content evolves
    4. Optionally applying contextual enhancement to improve chunk quality

    The system maintains chunk coherence by using LLMs to determine proposition
    relevance and automatically merge related content while keeping distinct
    topics separate

    Attributes
    ----------
    config: dict
        Configuration dictionary passed to LangChain runnable invocations.
    contextual_chunker: ContextualChunker
        Component responsible for processing and applying document-level
        contextual enrichment to the generated chunks.
    chunks: dict[str, dict]
        Internal chunk registry with propositions, titles, summaries, and
        metadata
    chunking_workflow: CompiledStateGraph
        LangGraph workflow orchestrating the chunking pipeline
    enable_contextual_chunking: bool
        Flag for applying contextual enhancement
    id_truncate_limit: int
        UUID truncation length for chunk identifiers (default: 5)
    lc_client: Union[ChatDeepSeek, ChatOpenAI, ChatOllama]
        LangChain language model client for generating chunk metadata and
        making semantic decisions during the chunking process
    proposition_indexing_pipeline: PropositionIndexingPipeline
        Pipeline for extracting propositions from documents
    storage_context: StorageContext
        Storage context for persistence management
    vector_store: VectorStore
        Initialized vector store for chunk storage
    _cached_outlines: str
        Cached string representation of chunk outlines for performance
        optimization
    _chunk_lock: asyncio.Lock
        Lock to ensure thread-safe chunk operations in concurrent contexts
    _find_relevant_chunk_chain: RunnableSequence
        Pre-compiled LLM chain for finding semantically relevant chunks for
        propositions
    _get_new_chunk_title_summary_chain: RunnableSequence
        Pre-compiled LLM chain for generating titles and summaries for new
        chunks
    _semaphore: asyncio.Semaphore
        Semaphore to limit the number of concurrent LLM API calls, preventing
        rate limit errors and managing resource utilization
    _update_chunk_title_summary_chain: RunnableSequence
        Pre-compiled LLM chain for updating existing chunk titles and summaries
    """

    __slots__ = (
        "chunks",
        "chunking_workflow",
        "contextual_chunker",
        "config",
        "enable_contextual_chunking",
        "id_truncate_limit",
        "lc_client",
        "lf_handler",
        "max_concurrent_llm_calls",
        "proposition_indexing_pipeline",
        "storage_context",
        "vector_store",
        "_cached_outlines",
        "_chunk_lock",
        "_find_relevant_chunk_chain",
        "_get_new_chunk_title_summary_chain",
        "_semaphore",
        "_update_chunk_title_summary_chain",
    )

    def __init__(
        self,
        lc_client: Union[ChatDeepSeek, ChatOpenAI, ChatOllama],
        lf_handler: Optional[CallbackHandler],
        vector_db: Union[ElasticVectorDB, QdrantVectorDB],
        collection_name: str,
        enable_contextual_chunking: Optional[bool] = True,
        max_concurrent_llm_calls: Optional[int] = 5,
    ):
        """
        Initialize the agentic chunker with LLM client and vector storage

        Parameters
        ----------
        lc_client: Union[ChatDeepSeek, ChatOpenAI, ChatOllama]
            LangChain LLM client for semantic decisions and metadata generation
        lf_handler: Optional[CallbackHandler]
            Langfuse callback handler for LLM tracing. None disables tracing
        vector_db: Union[ElasticVectorDB, QdrantVectorDB]
            Vector database instance for storing chunk embeddings.
        collection_name: str
            Base name for the collection. "_agentic" suffix is auto-appended
        enable_contextual_chunking: Optional[bool], default=True
            Whether to apply contextual enhancement to chunks
        max_concurrent_llm_calls: Optional[int], default=5
            Maximum number of concurrent LLM calls to prevent rate limiting
        """
        super().__init__(vector_db, collection_name, "_agentic")

        self.lc_client = lc_client
        self.config = {"callbacks": [lf_handler]} if lf_handler else {}
        self.max_concurrent_llm_calls = max_concurrent_llm_calls

        # Vector database
        self.vector_store, self.storage_context = vector_db.init_ingestion(
            f"{collection_name}_agentic"
        )

        # Proposition Indexing
        self.proposition_indexing_pipeline = PropositionIndexingPipeline(
            self.lc_client, lf_handler=lf_handler
        )

        # Agentic chunking objects
        self.chunks = {}
        self.id_truncate_limit = 5
        self._cached_outlines = ""

        self.enable_contextual_chunking = enable_contextual_chunking

        # Concurrency control
        self._chunk_lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(max_concurrent_llm_calls)

        self._init_template_chain()
        self.chunking_workflow = self._build_chunking_graph()

        # Contextual Chunker
        self.contextual_chunker = ContextualChunker(
            self.lc_client, lf_handler=lf_handler
        )

    def _init_template_chain(self) -> None:
        """
        Initialize and cache LLM chains for chunk management operations

        Creates three pre-configured chains used throughout the chunking process:
        - Finding relevant chunks for new propositions
        - Generating titles and summaries for new chunks
        - Updating titles and summaries of existing chunks
        """
        self._find_relevant_chunk_chain = (
            FindRelevantChunkTemplate().formulate() | self.lc_client
        ).with_config({"run_name": "AgenticChunking_FindRelevantChunk"})

        self._get_new_chunk_title_summary_chain = (
            GetNewChunkTitleSummaryTemplate().formulate()
            | self.lc_client
            | StructureJsonOutputParser()
        ).with_config({"run_name": "AgenticChunking_GetNewChunkTitleSummary"})

        self._update_chunk_title_summary_chain = (
            UpdateChunkTitleSummaryTemplate().formulate()
            | self.lc_client
            | StructureJsonOutputParser()
        ).with_config({"run_name": "AgenticChunking_UpdateChunkTitleSummary"})

        return None

    async def _add_propositions(self, propositions: List[str]) -> None:
        """
        Process and add multiple propositions to appropriate chunks concurrently

        This async implementation processes propositions sequentially to
        maintain semantic coherence, as each proposition's placement may depend
        on chunks created by previous propositions

        Parameters
        ----------
        propositions: List[str]
            Atomic propositions extracted from the document via the proposition
            indexing pipeline
        """
        for proposition in propositions:
            try:
                await self._add_proposition(proposition)
            except Exception as e:
                logger.error(
                    f"Failed to add proposition '{proposition[:50]}...': {str(e)}",
                    exc_info=True,
                )
                raise

        return None

    async def _add_proposition(self, proposition: str) -> None:
        """
        Add a single proposition to the most semantically relevant chunk
        asynchronously

        The method follows this logic:
        1. If no chunks exist, create the first chunk
        2. Otherwise, search for the most relevant existing chunk
        3. If a relevant chunk is found, add the proposition and update metadata
        4. If no relevant chunk exists, create a new chunk

        Parameters
        ----------
        proposition: str
            A single proposition to be added to the appropriate chunk
        """
        logger.info(f"\nAdding: '{proposition}'")

        async with self._chunk_lock:
            has_chunks = bool(self.chunks)

        if not has_chunks:
            logger.info("No chunks, creating a new one")
            await self._create_new_chunk(proposition)
            return None

        # find relevant chunks
        try:
            chunk_id = await self._find_relevant_chunk(proposition)
        except Exception as e:
            logger.error(f"Errors when finding relevant chunks: {e}")

        # If chunk is find then add the proposition to it
        if self._is_hash_prefix(chunk_id):
            logger.info(f"Adding to chunk: {self.chunks[chunk_id]['title']}")
            await self._add_proposition_to_chunk(chunk_id, proposition)

        # Create new chunks
        else:
            logger.info("No matching chunk found, creating new")
            await self._create_new_chunk(proposition)

        return None

    async def _add_proposition_to_chunk(self, chunk_id: str, proposition: str) -> None:
        """
        Adds a proposition to an existing chunk and updates its summary and title
        asynchronously

        Parameters
        ----------
        chunk_id: str
            Truncated UUID identifying the target chunk
        proposition: str
            Proposition to append to the chunk's proposition list
        """
        async with self._chunk_lock:
            chunk = self.chunks[chunk_id]
            chunk["propositions"].append(proposition)
            chunk_copy = chunk.copy()

        # Update chunk summary and title
        try:
            summary, title = await self._update_chunk_title_summary(chunk_copy)
        except Exception as e:
            logger.error(f"Error when updating chunk's title and its summary: {e}")

        async with self._chunk_lock:
            self.chunks[chunk_id]["summary"] = summary
            self.chunks[chunk_id]["title"] = title
            self._cached_outlines = ""

        return None

    def _build_chunking_graph(self) -> CompiledStateGraph:
        """Construct the LangGraph workflow for the agentic chunking pipeline"""
        workflow = StateGraph(ChunkingState)

        # Add nodes
        workflow.add_node("extract_propositions", self._extract_propositions_node)
        workflow.add_node("process_propositions", self._process_propositions_node)
        workflow.add_node("generate_chunks", self._generate_chunks_node)
        workflow.add_node("contextual_chunking", self._contextual_chunking_node)
        workflow.add_node("prepare_metadata", self._prepare_metadata_node)
        workflow.add_node("ingest_chunks", self._ingest_chunks_node)

        # Define edges
        workflow.set_entry_point("extract_propositions")
        workflow.add_edge("extract_propositions", "process_propositions")
        workflow.add_edge("process_propositions", "generate_chunks")

        # Conditional edge for contextual chunking
        workflow.add_conditional_edges(
            "generate_chunks",
            self._should_apply_contextual_chunking,
            {"contextual": "contextual_chunking", "skip": "prepare_metadata"},
        )

        workflow.add_edge("contextual_chunking", "prepare_metadata")
        workflow.add_edge("prepare_metadata", "ingest_chunks")
        workflow.add_edge("ingest_chunks", END)

        return workflow.compile()

    async def _create_new_chunk(self, proposition: str) -> None:
        """
        Initialize a new chunk from a proposition asynchronously

        Creates a chunk with a unique ID, generates a descriptive title and
        summary using an LLM, and adds it to the internal chunk registry.
        The cached outline is cleared to reflect the new chunk structure

        Parameters
        ----------
        proposition: str
            The proposition for which a new chunk will be created
        """
        # Create new chunk id
        chunk_id = str(uuid.uuid4())[: self.id_truncate_limit]

        # Create chunk summary & title
        try:
            summary, title = await self._get_new_chunk_title_summary(proposition)
        except Exception as e:
            logger.error(f"Error when getting the chunk's title and its summary: {e}")

        async with self._chunk_lock:
            self.chunks[chunk_id] = {
                "chunk_id": chunk_id,
                "propositions": [proposition],
                "title": title,
                "summary": summary,
                "chunk_index": len(self.chunks),
            }
            self._cached_outlines = ""

        logger.info(f"Created new chunk ({chunk_id}): {title}")

    async def _find_relevant_chunk(self, proposition: str) -> str:
        """
        Identify the most semantically relevant chunk for a proposition
        asynchronously

        Parameters
        ----------
        proposition: str
            The proposition for which a relevant chunk is to be found

        Returns
        -------
        str
            The ID of the relevant chunk, or "No chunks" if none is found
        """
        # Get current chunk outline
        outline = self._get_chunk_outline()

        # find relevant chunk using LLM
        return await self._ainvoke_llm_chain(
            self._find_relevant_chunk_chain,
            {"proposition": proposition, "chunk_outline": outline},
        )

    def _get_chunks(self) -> List[str]:
        """
        Returns the chunks in the list of string format

        Returns
        -------
        List[str]
            The retrieved chunks
        """
        return [" ".join(chunk["propositions"]) for chunk in self.chunks.values()]

    def _get_chunk_outline(self) -> str:
        """
        Gets a string that represents the current chunks, including their IDs,
        titles, and summaries

        Returns
        -------
        str
            A string outlining the current chunks, including their IDs, titles,
            and summaries
        """
        if self._cached_outlines:
            return self._cached_outlines

        outline_parts = [
            f"Chunk ID: {c['chunk_id']}\nChunk Name: {c['title']}\n"
            f"Chunk Summary: {c['summary']}\n"
            for c in self.chunks.values()
        ]

        self._cached_outlines = "\n".join(outline_parts)
        return self._cached_outlines

    def _get_chunk_metadata(self) -> List[str]:
        """
        Get chunk titles and summaries for contextual processing.

        Returns
        -------
        List[str]
            Formatted chunk metadata (title and summary pairs)
        """
        return [
            f"Title: {c['title']}\n\nSummary: {c['summary']}"
            for c in self.chunks.values()
        ]

    async def _get_new_chunk_title_summary(self, proposition: str) -> tuple:
        """
        Generates a new chunk title and summary for the given proposition using
        an LLM asynchronously

        Parameters
        ----------
        proposition: str
            The proposition for which a new chunk summary is generated

        Returns
        -------
        tuple
            The generated (summary, title) for the new chunk
        """
        # Get chunk summary from LLM response
        response = await self._ainvoke_llm_chain(
            self._get_new_chunk_title_summary_chain, {"proposition": proposition}
        )

        return response["summary"], response["title"]

    def _is_hash_prefix(self, text: str) -> bool:
        """
        Check if the given text is the first 5 characters of a valid hash ID. A
        valid hash ID typically contains only hexadecimal characters (0-9, a-f)

        Parameters
        ----------
        text : str
            The text to check

        Returns
        -------
        bool
            True if the text is valid, False otherwise
        """
        # Regex to match exactly 5 hexadecimal characters
        pattern = r"^[a-f0-9]{5}$"
        return bool(re.match(pattern, text.lower()))

    async def _ainvoke_llm_chain(self, chain: RunnableSequence, payload: dict) -> str:
        """
        Invoke an LLM chain asynchronously with standardized error handling,
        formatting, and rate limiting.

        Parameters
        ----------
        chain: RunnableSequence
            Pre-configured LangChain runnable sequence
        payload: Dict
            Input data for the LLM

        Returns
        -------
        str
            LLM response content
        """
        async with self._semaphore:
            try:
                response = await chain.ainvoke(payload, config=self.config)
                return response.content
            except Exception as e:
                logger.error(
                    f"LLM chain invocation failed: {str(e)}",
                    exc_info=True,
                    extra={"payload": payload},
                )
                raise e

    async def _update_chunk_title_summary(self, chunk: dict) -> tuple:
        """
        Updates the title and summary of the specified chunk using an LLM
        asynchronously

        Parameters
        ----------
        chunk: dict
            A dictionary containing the propositions, summary, and other metadata
            for the chunk

        Returns
        -------
        tuple
            The updated (summary, title) generated by the LLM
        """
        # Get new chunk summary from LLM response
        propositions_text = "\n".join(chunk["propositions"])

        try:
            response = await self._ainvoke_llm_chain(
                self._update_chunk_title_summary_chain,
                {
                    "current_title": chunk["title"],
                    "current_summary": chunk["summary"],
                    "proposition": propositions_text,
                },
            )
        except Exception as e:
            logger.error(
                f"Error in executing LLM chain for updating chunk's title and summary: {e}"
            )

        logger.info(
            f"Updated chunk ({chunk['chunk_id']}) title: "
            f"{chunk['title']} -> {response['title']}"
        )

        return response.get("summary", ""), response.get("title", "")

    async def _contextual_chunking_node(self, state: ChunkingState) -> ChunkingState:
        """
        LangGraph node: Apply contextual enrichment to chunks asynchronously

        Enhances chunks with additional document-level context using the
        ContextualChunker. If contextual chunking fails, falls back to
        using un-enriched chunks without raising an error

        Parameters
        ----------
        state: ChunkingState
            Current workflow state containing document, chunks, and metadata

        Returns
        -------
        ChunkingState
            Updated state with contextually enriched chunks in 'final_chunks',
            or original chunks if enrichment fails
        """
        try:
            logger.info("Applying contextual chunking")

            enhanced_chunks = await self.contextual_chunker.achunk(
                document=state["document"],
                chunks=state["chunks"],
                chunks_summary=state["chunk_metadata"],
            )

            return {**state, "final_chunks": enhanced_chunks, "error": None}

        except Exception as e:
            logger.error(f"Error in contextual chunking: {str(e)}", exc_info=True)
            return {**state, "final_chunks": state["chunks"], "error": None}

    async def _extract_propositions_node(self, state: ChunkingState) -> ChunkingState:
        """
        LangGraph node: Extract atomic propositions from the document
        asynchronously

        Invokes the proposition indexing pipeline to decompose the document
        into discrete, self-contained propositions suitable for semantic
        chunking

        Parameters
        ----------
        state: ChunkingState
            Current workflow state containing the source document

        Returns
        -------
        ChunkingState
            Updated state with extracted propositions, or error information if
            extraction fails
        """
        try:
            logger.info("Start proposition indexing")

            # extract proposition
            propositions = await self.proposition_indexing_pipeline.aindex(
                state["document"]
            )
            return {"propositions": propositions, "error": None}

        except Exception as e:
            logger.error(f"Error extracting propositions: {str(e)}", exc_info=True)
            return {**state, "error": f"Proposition extraction failed: {str(e)}"}

    async def _generate_chunks_node(self, state: ChunkingState) -> ChunkingState:
        """
        LangGraph node: Consolidate propositions into final chunk strings

        Retrieves all chunks from the internal registry and formats them
        along with their metadata for downstream processing

        Parameters
        ----------
        state: ChunkingState
            Current workflow state containing processed propositions

        Returns
        -------
        ChunkingState
            Updated state with consolidated chunks and metadata, or error
            information if generation fails
        """
        try:
            logger.info("Generating chunks from propositions")
            chunks = self._get_chunks()
            chunk_metadata = self._get_chunk_metadata()

            return {
                **state,
                "chunks": chunks,
                "chunk_metadata": chunk_metadata,
                "error": None,
            }
        except Exception as e:
            logger.error(f"Error generating chunks: {str(e)}", exc_info=True)
            return {**state, "error": f"Chunk generation failed: {str(e)}"}

    async def _ingest_chunks_node(self, state: ChunkingState) -> ChunkingState:
        """
        LangGraph node: Store processed chunks in the vector database
        asynchronously

        Ingests final chunks with their metadata into the configured vector
        database, making them available for semantic search and retrieval

        Parameters
        ----------
        state: ChunkingState
            Current workflow state containing final chunks and metadata

        Returns
        -------
        ChunkingState
            Updated state with error information if ingestion fails, otherwise
            unchanged
        """
        try:
            logger.info("Ingesting chunks into vector database")

            self.process_and_ingest(
                chunks=state["final_chunks"],
                base_metadata=state["base_metadata"],
            )

            logger.info("Agentic chunking completed successfully")
            return {**state, "error": None}
        except Exception as e:
            logger.error(f"Error ingesting chunks: {str(e)}", exc_info=True)
            return {**state, "error": f"Chunk ingestion failed: {str(e)}"}

    async def _process_propositions_node(self, state: ChunkingState) -> ChunkingState:
        """
        LangGraph node: Route propositions to appropriate chunks asynchronously

        Processes each proposition by either adding it to an existing
        semantically relevant chunk or creating a new chunk. This is where
        the core agentic decision-making occurs

        Parameters
        ----------
        state: ChunkingState
            Current workflow state containing extracted propositions

        Returns
        -------
        ChunkingState
            Updated state after proposition processing, or error information
            if processing fails
        """
        try:
            logger.info(f"Processing {len(state['propositions'])} propositions")
            await self._add_propositions(state["propositions"])
            return {**state, "error": None}

        except Exception as e:
            logger.error(f"Error processing propositions: {str(e)}", exc_info=True)
            return {**state, "error": f"Proposition processing failed: {str(e)}"}

    async def _prepare_metadata_node(self, state: ChunkingState) -> ChunkingState:
        """
        LangGraph node: Format chunk metadata for database ingestion

        Prepares metadata dictionary containing chunk titles and other
        relevant information. Selects either contextually enriched or
        standard chunks depending on what's available in the state

        Parameters
        ----------
        state: ChunkingState
            Current workflow state with generated or enriched chunks

        Returns
        -------
        ChunkingState
            Updated state with formatted metadata ready for ingestion, or error
            information if preparation fails
        """
        try:
            logger.info("Preparing chunk metadata")

            # Use final_chunks if available, otherwise use chunks
            final_chunks = state.get("final_chunks", state["chunks"])

            base_metadata = {"chunk_titles": [c["title"] for c in self.chunks.values()]}

            return {
                **state,
                "final_chunks": final_chunks,
                "base_metadata": base_metadata,
                "error": None,
            }
        except Exception as e:
            logger.error(f"Error preparing metadata: {str(e)}", exc_info=True)
            return {**state, "error": f"Metadata preparation failed: {str(e)}"}

    def _should_apply_contextual_chunking(
        self, state: ChunkingState
    ) -> Literal["contextual", "skip"]:
        """
        LangGraph conditional: Determine whether to apply contextual enrichment

        Routing function that directs the workflow to either the contextual
        chunking node or directly to metadata preparation based on configuration
        and error state

        Parameters
        ----------
        state: ChunkingState
            Current workflow state

        Returns
        -------
        Literal["contextual", "skip"]
            "contextual" if contextual chunking should be applied and no errors
            exist, "skip" to bypass contextual enrichment
        """
        if state.get("error"):
            return "skip"

        if self.enable_contextual_chunking:
            return "contextual"

        return "skip"

    async def achunk(self, document: str) -> None:
        """
        Execute the LangGraph workflow to chunk documents using agentic approach
        asynchronously

        Parameters
        ----------
        document: str
            Documents from which propositions will be extracted
        """
        logger.info("Starting async agentic chunking with LangGraph workflow")

        # Initialize state
        initial_state: ChunkingState = {
            "document": document,
            "propositions": [],
            "chunks": [],
            "chunk_metadata": [],
            "final_chunks": [],
            "base_metadata": {},
            "error": None,
        }

        try:
            # Execute the graph asynchronously
            final_state = await self.chunking_workflow.ainvoke(initial_state)

            # Check for errors in final state
            if final_state.get("error"):
                raise RuntimeError(f"Chunking workflow error: {final_state['error']}")

        except Exception as e:
            logger.error(f"Agentic chunking workflow failed: {str(e)}", exc_info=True)
            raise
