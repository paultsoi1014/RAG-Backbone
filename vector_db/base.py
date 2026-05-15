from abc import ABC, abstractmethod
from typing import List, Tuple, Union
from llama_index.core import StorageContext, VectorStoreIndex


class BaseVectorDB(ABC):
    """
    Abstract base class for a vector database interface. Concrete implementations
    should provide methods for initializing ingestion, setting up retrievers,
    and managing collections.
    """

    @abstractmethod
    def init_ingestion(
        self, collection_name: str
    ) -> Tuple[VectorStoreIndex, StorageContext]:
        """
        Initializes the ingestion pipeline for a given collection in the vector
        database. This method prepares the collection for storing vectorized
        data and sets up the corresponding storage context

        Parameters
        ----------
        collection_name: str
            The name of the collection to initialize for ingestion

        Returns
        -------
        Tuple[VectorStoreIndex, StorageContext]
            Initialized vector store and its corresponding utility container
            for storing nodes, indices, and vectors
        """
        pass

    @abstractmethod
    def init_retriever(
        self,
        collections: List[str],
        similarity_top_k: int,
        sparse_top_k: int,
        query_rewrite: bool,
    ) -> Union[VectorStoreIndex, List[VectorStoreIndex]]:
        """
        Initializes one or more retrievers for querying data from the specified
        collections. This method supports both single-collection and
        multi-collection retrieval, including hybrid dense and sparse search
        modes and optional query rewriting

        Parameters
        ----------
        collections: List[str]
            List of collection names to initialize for retrieval
        similarity_top_k: int
            The number of top results to retrieve through dense retrieval
        sparse_top_k: int
            The number of top results to retrieve through sparse retrieval
        query_rewrite: bool
            Whether or not to enable query rewriting during retrieval

        Returns
        -------
        Optional[Union[VectorStoreIndex, List[VectorStoreIndex]]]
            A single retriever if one collection is specified, or a list of
            retrievers if multiple collections or query rewrite is enabled
        """
        pass

    @abstractmethod
    def get_collections(self) -> Tuple[List[str], List[str]]:
        """
        Retrieves a list of all collection names stored in the vector database,
        along with a list of unique collection name prefixes with specific
        suffixes removed

        Returns
        -------
        Tuple[List[str], List[str]]
            - List of all full collection names.
            - List of unique collection prefixes with known suffixes removed
        """
        pass
