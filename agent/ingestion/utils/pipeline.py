import asyncio
import logging
import uuid

from pathlib import Path
from typing import Callable, List, Optional

from agent.base import RAGAgentBase
from agent.utils.parsers.document import DocumentParser
from .agentic_chunking import AgenticChunker
from .semantic_chunking import SemanticChunker
from .recursive_chunking import RecursiveChunker


logger = logging.getLogger(__name__)


class IngestionPipeline:
    """
    A pipeline for processing and chunking documents for Retrieval-Augmented
    Generation (RAG).

    This pipeline provides multiple chunking strategies (agentic, recursive,
    and semantic) that can be executed concurrently to process documents before
    storing them in a vector database. It handles document parsing, validation,
    and manages the complete ingestion workflow

    Attributes
    ----------
    semantic_chunker: SemanticChunker
        Chunker that uses semantic similarity to determine chunk boundaries
    recursive_chunker: RecursiveChunker
        Chunker that uses recursive text splitting based on separators
    agentic_chunker: AgenticChunker
        Chunker that uses LLM-based contextual understanding for intelligent chunking
    document_parser: DocumentParser
        Parser responsible for extracting text from the provided documents
    VALID_MODES: frozenset
        Set of valid chunking mode names: {'agentic', 'recursive', 'semantic'}
    """

    __slots__ = (
        "semantic_chunker",
        "recursive_chunker",
        "agentic_chunker",
        "document_parser",
        "VALID_MODES",
    )

    def __init__(self, agent_base: RAGAgentBase, collection_name: str):
        """
        Initializes the RAG ingestion pipeline with required models and database
        configuration.

        Parameters
        ----------
        agent_base: RAGAgentBase
            The base agent providing the language model client, embedding model,
            vector database, and Langfuse handler
        collection_name: str
            The name of the collection/index in the vector database where
            ingested document chunks will be stored.
        """
        # Init llm client, embedding model
        lc_client = agent_base.get_lc_client()
        li_embed_model = agent_base.get_li_embed_model()

        # Init langfuse handler, vector db
        lf_handler = agent_base.init_langfuse_tracing(session_id=str(uuid.uuid4()))
        vector_db = agent_base.get_vector_db()

        # Init document parser
        self.document_parser = DocumentParser(agent_base)

        # Init semantic chunker
        self.semantic_chunker = SemanticChunker(
            lc_client=lc_client,
            embed_model=li_embed_model,
            lf_handler=lf_handler,
            vector_db=vector_db,
            collection_name=collection_name,
            breakpoint_percentile_threshold=95,
            buffer_size=1,
        )

        # Init recursive chunker
        self.recursive_chunker = RecursiveChunker(
            vector_db=vector_db,
            collection_name=collection_name,
        )

        # Init agentic chunker
        self.agentic_chunker = AgenticChunker(
            lc_client=lc_client,
            lf_handler=lf_handler,
            vector_db=vector_db,
            collection_name=collection_name,
            enable_contextual_chunking=True,
        )

        self.VALID_MODES = frozenset({"agentic", "recursive", "semantic"})

    def _validate_and_parse_document(self, folder_path: str, modes: List[str]) -> str:
        """
        Validates chunking modes and parses the document from the folder path

        Parameters
        ----------
        folder_path: str
            The path to the folder containing documents to be processed
        modes: List[str]
            List of chunking modes to validate

        Returns
        -------
        str
            A str containing the parsed_text

        Raises
        ------
        ValueError
            If invalid modes are specified or no content is found in the folder
        """
        # Validate modes
        invalid_modes = set(modes) - self.VALID_MODES
        if invalid_modes:
            raise ValueError(
                f"Invalid chunking modes: {invalid_modes}. Valid modes are: {self.VALID_MODES}"
            )

        # Parse document
        parsed_text = self.document_parser.parse(folder_path)

        # Validate parsed content
        if not parsed_text:
            raise ValueError("No content found in the specified folder")

        return parsed_text

    async def _achunk(self, document: str, modes: List[str]) -> None:
        """
        Performs asynchronous chunking of a document using the specified
        chunking strategies

        Parameters
        ----------
        document: str
            The text content of the document to be chunked.
        modes: List[str]
            List of chunking modes to execute ('agentic', 'recursive', 'semantic')
        """
        mode_map = {
            "agentic": lambda: self.agentic_chunker.achunk(document=document),
            "recursive": lambda: asyncio.to_thread(
                self.recursive_chunker.chunk, document=document
            ),
            "semantic": lambda: self.semantic_chunker.achunk(document=document),
        }

        # Filter out invalid modes
        valid_modes = [mode for mode in modes if mode in mode_map]

        if not valid_modes:
            logger.warning("No valid chunking modes specified")
            return None

        logger.info(f"Starting chunking tasks: {valid_modes}")

        # Async wrapper to handle try/except for each individual task
        async def _safe_chunk(mode: str):
            try:
                await mode_map[mode]()
                logger.info(f"Chunking completed successfully for mode '{mode}'")
                return mode, True
            except Exception as e:
                logger.error(f"Chunking failed for mode '{mode}': {e}", exc_info=True)
                return mode, False

        try:
            # Create tasks using the safe wrapper
            tasks = [
                asyncio.create_task(_safe_chunk(mode), name=f"{mode}_chunking")
                for mode in valid_modes
            ]

            # Execute all tasks concurrently
            results = await asyncio.gather(*tasks)

            # Check status
            completed_modes = [mode for mode, success in results if success]
            failed_modes = [mode for mode, success in results if not success]

            if completed_modes:
                logger.info(
                    f"Successfully completed chunking with modes: {completed_modes}"
                )
            if failed_modes:
                logger.warning(f"Failed chunking modes: {failed_modes}")

        except Exception as e:
            logger.error(
                f"Unexpected error orchestrating chunking tasks: {e}", exc_info=True
            )

        return None

    def ingest(self, folder_path: str, mode: Optional[List[str]] = None) -> Callable:
        """
        Synchronously processes documents from a folder and chunks them

        Parameters
        ----------
        folder_path: str
            The path to the folder containing documents to be processed
        mode: Optional[List[str]], default=None
            The mode(s) of chunking to be used. Options are 'agentic',
            'recursive', and/or 'semantic'. Defaults to ['recursive']
        """
        if mode is None:
            mode = ["recursive"]

        # Validate and parse document
        parsed_text = self._validate_and_parse_document(folder_path, mode)

        # Run async chunking in sync context
        try:
            loop = asyncio.get_running_loop()
            loop.run_until_complete(self._achunk(document=parsed_text, modes=mode))
        except RuntimeError:
            # No running loop, create new one
            asyncio.run(self._achunk(document=parsed_text, modes=mode))

    async def aingest(
        self, folder_path: str, mode: Optional[List[str]] = None
    ) -> Callable:
        """
        Asynchronously processes documents from a folder and chunks them

        Parameters
        ----------
        folder_path: str
            The path to the folder containing documents to be processed
        mode: Optional[List[str]], default=None
            The mode(s) of chunking to be used. Options are 'agentic',
            'recursive', and/or 'semantic'. Defaults to ['recursive']
        """
        if mode is None:
            mode = ["recursive"]

        # Validate and parse document
        parsed_text = self._validate_and_parse_document(folder_path, mode)

        return await self._achunk(document=parsed_text, modes=mode)
