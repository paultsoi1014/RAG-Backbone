from typing import List, Tuple, Union
from qdrant_client import QdrantClient, AsyncQdrantClient

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore


class QdrantVectorDB:
    __slots__ = ["client", "aclient"]

    def __init__(self, url: str):
        """
        A class to interact with a Qdrant vector database, providing methods for
        initializing ingestion, retrieving relevant collections, and querying
        the database

        Parameters
        ----------
        url: str
            The URL of the Qdrant database server

        Attributes
        ----------
        client: QdrantClient
            A synchronous client instance used to interact with the Qdrant
            database
        aclient: AsyncQdrantClient
            An asynchronous client instance for performing non-blocking
            operations with Qdrant
        """
        self.client = QdrantClient(url=url)
        self.aclient = AsyncQdrantClient(url=url)

    def init_ingestion(
        self, collection_name: str
    ) -> Tuple[VectorStoreIndex, StorageContext]:
        """
        Initializes the ingestion pipeline for a specific collection in the
        Qdrant vector database

        Parameters
        ----------
        collection_name: str
            The name of the collection in Qdrant to initialize for ingestion

        Returns
        -------
        Tuple[VectorStoreIndex, StorageContext]
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
        similarity_top_k: int,
        sparse_top_k: int,
        query_rewrite: bool,
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
        # Validate collections input
        if not collections:
            raise Exception("No collections can be referred to")

        # Single collection without query rewrite
        if len(collections) == 1 and not query_rewrite:
            # Qdrant vector db init
            vector_store = QdrantVectorStore(
                client=self.client,
                collection_name=collections[0],
                enable_hybrid=True,
                fastembed_sparse_model="Qdrant/bm42-all-minilm-l6-v2-attentions",
            )

            return VectorStoreIndex.from_vector_store(
                vector_store=vector_store
            ).as_retriever(
                similarity_top_k=similarity_top_k,
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
                    similarity_top_k=similarity_top_k,
                    sparse_top_k=sparse_top_k,
                    vector_store_query_mode="hybrid",
                )
            )

        return vector_retrievers

    def get_collections(self) -> list:
        """
        Retrieves the list of all collection names in the Qdrant vector database

        Returns
        -------
        list
            A list of collection names currently stored in the Qdrant database
        """
        collections = [
            collection["name"]
            for collection in self.client.get_collections().model_dump()["collections"]
        ]

        return collections

    def delete_collection(self, collection: str) -> bool:
        """Deletes a specified collection from the Qdrant vector database"""
        return self.client.delete_collection(collection_name=collection)
