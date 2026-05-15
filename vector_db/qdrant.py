from typing import List, Tuple, Union
from qdrant_client import QdrantClient, AsyncQdrantClient

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.vector_stores.types import BasePydanticVectorStore

from vector_db.base import BaseVectorDB


class QdrantVectorDB(BaseVectorDB):
    __slots__ = ("url", "client", "aclient")

    def __init__(self, url: str):
        """
        A class to interact with a Qdrant vector database, providing methods for
        initializing ingestion, retrieving relevant collections, and querying
        the database

        Attributes
        ----------
        url: str
            The URL of the Qdrant database server
        client: QdrantClient
            A synchronous client instance used to interact with the Qdrant
            database
        aclient: AsyncQdrantClient
            An asynchronous client instance for performing non-blocking
            operations with Qdrant
        """
        self.url = url
        self.client = QdrantClient(url=url)
        self.aclient = AsyncQdrantClient(url=url)

        self._test_connection()

    def _test_connection(self):
        """
        Test the connection to Qdrant by attempting to get collections

        Raises
        ------
        Exception
            If the connection test fails
        """
        try:
            # Attempt a simple operation to verify connection
            self.client.get_collections()

        except Exception as e:
            raise Exception(
                f"Connection test failed for Qdrant server at {self.url}. "
                f"Error: {str(e)}"
            ) from e

    def init_ingestion(
        self, collection_name: str
    ) -> Tuple[BasePydanticVectorStore, StorageContext]:
        """
        Initializes the ingestion pipeline for a specific collection in the
        Qdrant vector database

        Parameters
        ----------
        collection_name: str
            The name of the collection in Qdrant to initialize for ingestion

        Returns
        -------
        Tuple[BasePydanticVectorStore, StorageContext]
            Initialized vector store and its corresponding utility container
            for storing nodes, indices, and vectors
        """
        # Qdrant vector db init
        vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=collection_name,
            fastembed_sparse_model="Qdrant/bm42-all-minilm-l6-v2-attentions",
            enable_hybrid=True,
        )

        # container for storing nodes, indices, and vectors
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        return vector_store, storage_context

    def init_retriever(
        self,
        collections: List[str],
        dense_top_k: int,
        sparse_top_k: int,
    ) -> Union[VectorStoreIndex, List[VectorStoreIndex]]:
        """
        Initializes a retriever for querying one or more collections from the
        Qdrant vector database. This method handles both single collection and
        multi-collection retrieval. If multiple collections are provided or query
        rewriting is enabled, multiple retrievers are initialized

        Parameters
        ----------
        collections: List[str]
            List of collection names to initialize for retrieval
        dense_top_k: int
            The number of top results to retrieve through dense retrieval
        sparse_top_k: int
            The number of top results to retrieve through sparse retrieval

        Returns
        -------
        Optional[Union[VectorStoreIndex, List[VectorStoreIndex]]]
            A single retriever if one collection is specified, or a list of
            retrievers if multiple collections or query rewrite is enabled
        """
        # Validate collections input
        if not collections:
            raise Exception("No collections can be referred to")

        # Single collection without query rewrite
        if len(collections) == 1:
            # Qdrant vector db init
            vector_store = QdrantVectorStore(
                client=self.client,
                aclient=self.aclient,
                collection_name=collections[0],
                enable_hybrid=True,
                fastembed_sparse_model="Qdrant/bm42-all-minilm-l6-v2-attentions",
            )

            return VectorStoreIndex.from_vector_store(
                vector_store=vector_store
            ).as_retriever(
                similarity_top_k=dense_top_k,
                sparse_top_k=sparse_top_k,
                vector_store_query_mode="hybrid",
            )

        # Multiple collections or query rewrite enabled
        vector_retrievers = []
        for collection in collections:
            async_vector_store = QdrantVectorStore(
                client=self.client,
                aclient=self.aclient,
                collection_name=collection,
                enable_hybrid=True,
                fastembed_sparse_model="Qdrant/bm42-all-minilm-l6-v2-attentions",
            )
            vector_retrievers.append(
                VectorStoreIndex.from_vector_store(
                    vector_store=async_vector_store,
                ).as_retriever(
                    similarity_top_k=dense_top_k,
                    sparse_top_k=sparse_top_k,
                    vector_store_query_mode="hybrid",
                )
            )

        return vector_retrievers

    def get_collections(self) -> Tuple[list, list]:
        """
        Retrieves all collection names from the Qdrant vector database and
        extracts unique collection prefixes by removing specified suffixes

        Returns
        -------
        Tuple[List[str], List[str]]
            - The full list of collection names as stored in the Qdrant database
            - A list of unique collection prefixes with specified suffixes
              removed
        """
        suffixes_to_remove = {"recursive", "agentic", "semantic"}

        # Get all collections from Qdrant
        collections = [
            collection["name"]
            for collection in self.client.get_collections().model_dump()["collections"]
        ]

        # Filter out unique collections
        unique_collection_prefix = set()
        for name in collections:
            parts = name.split("_")
            if parts[-1] in suffixes_to_remove:
                unique_collection_prefix.add("_".join(parts[:-1]))
            else:
                unique_collection_prefix.add(name)

        return collections, list(unique_collection_prefix)

    def delete_collection(self, collection: str) -> bool:
        """Deletes a specified collection from the Qdrant vector database"""
        return self.client.delete_collection(collection_name=collection)
