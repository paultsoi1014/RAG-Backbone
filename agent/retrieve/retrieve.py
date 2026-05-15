import logging
import uuid

from typing import List, Optional, Tuple, Union

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore

from langfuse.langchain import CallbackHandler

from agent.base import RAGAgentBase
from .rewrite import QueryRewriter
from .retriever import FusionRetriever, SimpleRetriever
from .retriever.utils.router import CollectionsRouter

from vector_db import QdrantVectorDB, ElasticVectorDB

logger = logging.getLogger(__name__)


class RetrieveAgent:
    """
    Retrieves relevant documents from a vector database for a RAG agent

    This class orchestrates document searches across single or multiple
    collections, supporting both simple dense retrieval and fusion-based
    retrieval

    Attributes
    ----------
    collection_router: CollectionsRouter
        Intelligently selects relevant document collections based on the query
    dense_top_k: int
        The number of top-k relevant documents to retrieve via dense retrieval
    enable_query_preprocess: bool
        Flag indicating whether query rewriting and expansion are enabled
    fuse_top_k: int
        Maximum number of documents to return when using fusion retrieval
    num_queries: int
        The number of expanded queries to generate when preprocessing is enabled
    query_rewriter: QueryRewriter
        Rewrites and expands queries to improve retrieval accuracy
    repacking_strategy: Optional[str]
        The strategy used for packing or ordering retrieved document chunks
        within the context window
        - 'forward': Original order
        - 'reverse': Reversed order
        - 'sides': Alternating from edges inward
    sparse_top_k: int
        The number of top-k relevant documents to retrieve via sparse retrieval
    vector_db: Union[QdrantVectorDB, ElasticVectorDB]
        Vector database instance for retrieving document collections
    """

    __slots__ = (
        "collection_router",
        "dense_top_k",
        "enable_query_preprocess",
        "fuse_top_k",
        "num_queries",
        "query_rewriter",
        "repacking_strategy",
        "sparse_top_k",
        "vector_db",
    )

    def __init__(self, agent_base: RAGAgentBase):
        """
        Initialize the RAGAgentRetriever with configuration from a base agent

        Parameters
        ----------
        agent_base: RAGAgentBase
            Base RAG agent containing DB, LLM clients, and config parameters
        """
        # Init vector db and its corresponding parameters
        self.vector_db: Union[QdrantVectorDB, ElasticVectorDB] = (
            agent_base.get_vector_db()
        )

        # Init langfuse handler
        lf_handler: Optional[CallbackHandler] = agent_base.init_langfuse_tracing(
            session_id=str(uuid.uuid4())
        )

        # Set retrieval settings
        retrieve_settings = agent_base.get_retrieval_settings()
        self.enable_query_preprocess = retrieve_settings["enable_query_preprocess"]
        self.num_queries = retrieve_settings["num_queries"]
        self.dense_top_k = retrieve_settings["dense_top_k"]
        self.sparse_top_k = retrieve_settings["sparse_top_k"]
        self.fuse_top_k = retrieve_settings["fuse_top_k"]
        self.repacking_strategy = retrieve_settings["repacking_strategy"]

        # Init query rewriter for query preprocessing
        self.query_rewriter = QueryRewriter(
            lc_client=agent_base.get_lc_client(), lf_handler=lf_handler
        )

        # Init collection router for retrieving relevant collections
        self.collection_router = CollectionsRouter(
            lc_client=agent_base.get_lc_client(), lf_handler=lf_handler
        )

    def _filter_existing_collections(self, collections: List[str]) -> List[str]:
        """
        Filters out requested collections that do not exist in the vector DB

        Parameters
        ----------
        collections: List[str]
            A list of collection names to check.

        Returns
        -------
        List[str]
            A filtered list containing only existing collection names. Raises
            ValueError if none exist
        """
        # Retrieve all existing collections from vector DB
        existing_collections, _ = self.vector_db.get_collections()

        # Filter collections that exist
        filtered_collections = [
            col for col in collections if col in existing_collections
        ]

        if filtered_collections and len(filtered_collections) < len(collections):
            missing = set(collections) - set(filtered_collections)
            logger.warning(
                f"The following requested collections were not found in the DB: {missing}"
            )

        if not filtered_collections:
            logger.error(
                f"None of the requested collections exist. Requested: {collections}"
            )
            raise ValueError(
                "None of the specified collections exist in the vector database."
            )

        logger.info(
            f"Number of collections to be retrieved: {len(filtered_collections)}"
        )

        return filtered_collections

    def _get_retriever(
        self, collections: List[str]
    ) -> Union[VectorStoreIndex, List[VectorStoreIndex]]:
        """
        Initializes the appropriate retriever(s) based on the collections

        Parameters
        ----------
        collections: List[str]
            The document collections to retrieve information from

        Returns
        -------
        Union[VectorStoreIndex, List[VectorStoreIndex]]
            A single retriever or a list of retrievers
        """
        return self.vector_db.init_retriever(
            collections,
            dense_top_k=self.dense_top_k,
            sparse_top_k=self.sparse_top_k,
        )

    async def _get_relevant_collections_prefix(
        self, queries: Union[str, List[str]], metadata: Optional[dict] = None
    ) -> List[str]:
        """
        Determines the most relevant document collections prefix based on the
        queries

        Parameters
        ----------
        queries: Union[str, List[str]]
            The user queries to identify relevant document collections
        metadata: Optional[dict], default=None
            Additional metadata to assist in collection selection

        Returns
        -------
        List[str]
            A list of relevant document collection prefixes
        """
        # Retrieve all available collection from vector db
        _, collection_prefixes = self.vector_db.get_collections()
        logger.info(f"Available collections: {collection_prefixes}")

        # Retrieve all related collection for retrieval
        try:
            relevant_prefixes = await self.collection_router.aget_relevant_collections(
                collection_prefixes, queries=queries, metadata=metadata
            )
            logger.info(f"Router identified relevant prefixes: {relevant_prefixes}")
        except Exception as e:
            logger.error(f"Error when executing collections routing: {e}")

        return relevant_prefixes

    async def aretrieve_info(
        self,
        queries: Union[str, List[str]],
        collections_prefix: Optional[List[str]] = None,
        metadata: Optional[dict] = None,
        recursive_only: Optional[bool] = False,
    ) -> Tuple[List[NodeWithScore], Union[str, List[str]]]:
        """
        Asynchronously retrieves relevant text from the vector database

        Parameters
        ----------
        queries: Union[str, List[str]]
            Query(s) for retrieving relevant documents
        collections_prefix: Optional[List[str]], default=None
            The document collections to search within. If None, relevant
            collections will be routed automatically
        metadata: Optional[dict], default=None
            Additional metadata to assist in collection selection
        recursive_only: Optional[bool], default=False
            If True, only considers collections with the `_recursive` suffix

        Returns
        -------
        Tuple[List[NodeWithScore], Union[str, List[str]]]
            A tuple containing the list of retrieved nodes and the relevant text
        """
        # Query Preprocessing
        if self.enable_query_preprocess:
            try:
                queries = await self.query_rewriter.arewrite(
                    queries, num_queries=self.num_queries
                )
            except Exception as e:
                logger.error(f"Error when conducting query preprocessing: {e}")

        # Collections
        if not collections_prefix:
            collections_prefix = await self._get_relevant_collections_prefix(
                queries, metadata=metadata
            )

        if recursive_only:
            # Only keep recursive suffix
            collections = [f"{prefix}_recursive" for prefix in collections_prefix]
        else:
            collections = [
                f"{prefix}_{suffix}"
                for prefix in collections_prefix
                for suffix in ("agentic", "semantic", "recursive")
            ]

        # Filter non-existing collections
        relevant_collections = self._filter_existing_collections(collections)

        if not relevant_collections:
            logger.warning("No relevant collections found for retrieval")
            return [], "No Collection found"

        # Get retriever(s) for the selected collections
        retrievers = self._get_retriever(collections=relevant_collections)

        # Execute retrieval based on number of retrievers
        if isinstance(retrievers, list):
            logger.info("Multiple retrievers detected; using FusionRetriever")

            # Use fusion retriever for multiple collections
            fusion_retriever = FusionRetriever(
                retrievers=retrievers,
                top_n=self.fuse_top_k,
                repacking_strategy=self.repacking_strategy,
            )
            retrieved_node, retrieved_text = await fusion_retriever.aretrieve(
                queries=queries
            )
        else:
            logger.info("Single retriever detected; using SimpleRetriever")

            simple_retriever = SimpleRetriever(
                retrievers, repacking_strategy=self.repacking_strategy
            )
            retrieved_node, retrieved_text = await simple_retriever.aretrieve(
                queries=queries
            )

        return retrieved_node, retrieved_text
