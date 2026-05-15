import asyncio
import logging

from typing import List, Optional, TypedDict

from llama_index.core.schema import NodeWithScore

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agent.base import RAGAgentBase
from agent.retrieve import RetrieveAgent
from agent.search.utils.relevancy import calculate_relevancy

logger = logging.getLogger(__name__)


class SearchState(TypedDict):
    collections: Optional[List[str]]
    document_relevancy: dict
    metadata: dict
    query: str
    relevant_nodes: List[NodeWithScore]


class SearchRelevancyAgent:
    """
    An agent that manages the retrieval of documents and evaluates their
    relevancy to a user's query

    This agent executes a workflow to first retrieve nodes from a vector
    database and then compute node-level relevancy scores, providing a
    quantifiable measure of document utility

    Attributes
    ----------
    agent_base: RAGAgentBase
        The base RAG agent instance containing shared components and
        configuration
    search_workflow: CompiledStateGraph
        The compiled workflow representing the retrieval and relevancy
        computation steps
    """

    __slots__ = ("agent_base", "lf_handler", "search_workflow")

    def __init__(self, agent_base: RAGAgentBase):
        """
        Initialize the SearchRelevancyAgent with a base agent configuration

        Parameters
        ----------
        agent_base: RAGAgentBase
            The base RAG agent instance containing the vector database, language
            model client, and retrieval configuration parameters
        """
        # Init agent base
        self.agent_base = agent_base
        self.search_workflow = self.build_workflow()

    def _document_search(self, state: SearchState) -> SearchState:
        """
        Evaluate the relevancy of retrieved documents based on their node-level
        similarity scores

        Parameters
        ----------
        state: SearchState
            The current state of the search workflow, including retrieved nodes

        Returns
        -------
        SearchState
            A dictionary containing the `document_relevancy` field, which maps
            document names to their relevancy scores as percentages
        """
        nodes = state.get("relevant_nodes", [])
        logger.info(f"Calculating relevancy for {len(nodes)} retrieved nodes.")

        doc_relevancy = calculate_relevancy(nodes)
        return {"document_relevancy": doc_relevancy}

    async def _aretrieve(self, state: SearchState) -> SearchState:
        """
        Retrieve relevant nodes from specified collections based on the user's
        query

        Parameters
        ----------
        state: SearchState
            The current state of the search workflow, including the query,
            collections, and metadata

        Returns
        -------
        SearchState
            A dictionary containing the `relevant_nodes` field, which holds
            the list of retrieved nodes along with their similarity scores
            from the knowledge base
        """
        # Retrieve relevant documents
        rag_retriever = RetrieveAgent(self.agent_base)
        retrieved_node, _ = await rag_retriever.aretrieve_info(
            state["query"],
            collections_prefix=state.get("collections", []),
            metadata=state.get("metadata", {}),
            recursive_only=True,
        )

        return {"relevant_nodes": retrieved_node}

    def build_workflow(self) -> CompiledStateGraph:
        """
        Build and compile the document search workflow as a state graph

        The workflow has two main steps:
            1. 'retrieve': retrieve relevant nodes
            2. 'document_search': calculate document relevancy

        Returns
        -------
        CompiledStateGraph
            The compiled state graph representing the document search workflow
        """
        builder = StateGraph(SearchState)
        builder.add_node("document_search", self._document_search)
        builder.add_node("retrieve", self._aretrieve)

        builder.add_edge(START, "retrieve")
        builder.add_edge("retrieve", "document_search")
        builder.add_edge("document_search", END)

        workflow = builder.compile()

        return workflow

    async def asearch_relevancy(
        self,
        query: str,
        collections: Optional[List[str]] = None,
        metadata: Optional[dict] = None,
    ):
        """
        Asynchronously execute the RAG search workflow for a given query and
        return document relevancy scores

        Parameters
        ----------
        query: str
            The user query for which document relevancy needs to be evaluated
        collections: Optional[List[str]]
            List of collection names to search within; if None, searches all
            collections. Default is None
        metadata: Optional[dict]
            Additional metadata to include to assist in collection selection.
            Default is None

        Returns
        -------
        dict
            A dictionary mapping document names to their relevancy scores as
            percentages. Returns empty dict if workflow execution fails
        """
        # Initialize workflow for document search
        try:
            response = await self.search_workflow.ainvoke(
                {
                    "query": query,
                    "collections": collections,
                    "metadata": metadata,
                }
            )
            relevancy_scores = response.get("document_relevancy", {})

            logger.info(
                f"Workflow finished. Returned relevancy for {len(relevancy_scores)} documents."
            )
            return relevancy_scores

        except Exception as e:
            logger.error(f"Workflow execution failed: {str(e)}", exc_info=True)
            return {}

    def search_relevancy(
        self,
        query: str,
        collections: Optional[List[str]] = None,
        metadata: Optional[dict] = None,
    ):
        """
        Execute the RAG search workflow for a given query and return document
        relevancy scores

        This is a synchronous wrapper around asearch_relevancy(). For better
        performance in async contexts, use asearch_relevancy() directly

        Parameters
        ----------
        query: str
            The user query for which document relevancy needs to be evaluated
        collections: Optional[List[str]]
            List of collection names to search within; if None, searches all
            collections. Default is None
        metadata: Optional[dict]
            Additional metadata to include to assist in collection selection.
            Default is None

        Returns
        -------
        dict
            A dictionary mapping document names to their relevancy scores as
            percentages. Returns empty dict if workflow execution fails
        """
        try:
            loop = asyncio.get_running_loop()
            return asyncio.create_task(
                self.asearch_relevancy(query, collections, metadata)
            )

        except RuntimeError:
            return asyncio.run(self.asearch_relevancy(query, collections, metadata))
