from typing import List, Union, Tuple

from llama_index.core import VectorStoreIndex
from langfuse.callback import CallbackHandler

from agent.base import RAGAgentBase

from utils.retriever.fusion_retrieve import FusionRetriever
from utils.retriever.simple_retrieve import SimpleRetriever


class RAGAgentRetriever(RAGAgentBase):
    def __init__(self, lf_handler: CallbackHandler):
        """
        A class that extends `RAGAgentBase` to handle the retrieval of
        relevant documents from a vector database based on a user's query and
        subject area. This class supports both simple and fusion-based document
        retrieval methods

        Attributes
        ----------
        vector_db: Union[QdrantVectorDB]
            An instance of the vector database for storing and retrieving
            document collections
        langchain_client: Union[FastEmbedEmbedding, HuggingFaceEmbedding,
                                OllamaEmbedding, OpenAIEmbedding]
            A client instance for interacting with the language model, inherited
            from `RAGAgentBase`
        lf_handler : langfuse.callback.CallbackHandler
            Handler for logging and tracing LLM interactions via Langfuse
        similarity_top_k: int
            The number of top-k relevant documents to retrieve from the database
            through dense retrieval
        sparse_top_k: int
            The number of top-k relevant documents to retrieve from the database
            through sparse retrieval
        """
        RAGAgentBase.__init__(self)

        # Init langfuse handler for llm tracing & management
        self.lf_handler = lf_handler

    def _check_collection_exist(self, collections: List[str]) -> bool:
        """
        Checks whether the specified document collections exist in the vector
        database.

        Parameters
        ----------
        collections : List[str]
            A list of document collections to check.

        Returns
        -------
        bool
            True if all collections exist, otherwise False.
        """
        # Retrieve all available collection from vector db
        existing_collections = self.vector_db.get_collections()

        return all(item in existing_collections for item in collections)

    def _get_retriever(
        self, collections: List[str], query_rewrite: bool
    ) -> Union[VectorStoreIndex, List[VectorStoreIndex]]:
        """
        Initializes the appropriate retriever(s) based on the provided document
        collections and query rewrite option

        Parameters
        ----------
        collections : List[str]
            The document collections to retrieve information from
        query_rewrite : bool
            Whether to rewrite the query before performing the retrieval

        Returns
        -------
        Union[VectorStoreIndex, List[VectorStoreIndex]]
            A retriever or a list of retrievers initialized with the relevant
            collections that will be used to retrieve top-k relevant documents
        """
        return self.vector_db.init_retriever(
            collections,
            similarity_top_k=self.similarity_top_k,
            sparse_top_k=self.sparse_top_k,
            query_rewrite=query_rewrite,
        )

    def retrieve_info(
        self, query: str, collections: str, query_rewrite: bool
    ) -> Tuple[str, List[str]]:
        """
        Retrieves relevant text from the vector database using the provided
        retrievers

        Parameters
        ----------
        query : str
            The user query for retrieving relevant documents
        collections : List[str]
            The document collections to search within
        query_rewrite : bool
            Whether to rewrite the query before retrieval

        Returns
        -------
        Tuple[str, List[str]]
            A string containing the relevant text retrieved from the database
            and A list of relevant document collections used for the retrieval
            process
        """
        if not self._check_collection_exist(collections=collections):
            raise ValueError(f"Collection(s) {collections} do not exist")

        # Get query engine to answer question
        retrievers = self._get_retriever(
            collections=collections, query_rewrite=query_rewrite
        )

        if isinstance(retrievers, list):
            # Init fusion retriever to fuse retrieved information from more than 1
            fusion_retriever = FusionRetriever(
                lc_client=self.langchain_client,
                lf_handler=self.lf_handler,
                retrievers=retrievers,
                top_n=2,
            )

            # Retrieve relevant information in the vector database
            _, retrieved_text = fusion_retriever.retrieve(
                query=query, query_rewrite=query_rewrite
            )

        else:
            simple_retriever = SimpleRetriever(retrievers)
            _, retrieved_text = simple_retriever.retrieve(query=query)

        return retrieved_text
