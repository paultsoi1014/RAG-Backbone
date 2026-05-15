from typing import List, Tuple, Union
from elasticsearch import Elasticsearch

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llama_index.vector_stores.elasticsearch import (
    ElasticsearchStore,
    AsyncDenseVectorStrategy,
)

from vector_db.base import BaseVectorDB


class ElasticVectorDB(BaseVectorDB):
    __slots__ = ["api_key", "client", "url"]

    def __init__(self, url: str, api_key: str):
        """
        A class to interact with an Elasticsearch vector database, providing
        methods for initializing ingestion, retrieving relevant collections, and
        querying the database

        Parameters
        ----------
        url: str
            The URL of the Elasticsearch server
        api_key: str
            API key required for providers that need authentication

        Attributes
        ----------
        client: Elasticsearch
            A synchronous client instance used to interact with Elasticsearch
        url: str
            The URL of the Elasticsearch server
        """
        super().__init__()

        self.api_key = api_key
        self.url = url
        self.client = Elasticsearch(url, api_key=api_key)

    def init_ingestion(
        self, collection_name: str
    ) -> Tuple[BasePydanticVectorStore, StorageContext]:
        """
        Initializes the ingestion pipeline for a specific collection in the
        Elasticsearch vector database.

        Parameters
        ----------
        collection_name: str
            The name of the collection (index) in Elasticsearch to initialize
            for ingestion.

        Returns
        -------
        Tuple[BasePydanticVectorStore, StorageContext]
            Initialized vector store and its corresponding utility container
            for storing nodes, indices, and vectors.
        """
        vector_store = ElasticsearchStore(
            es_url=self.url,
            es_api_key=self.api_key,
            index_name=collection_name,
            retrieval_strategy=AsyncDenseVectorStrategy(hybrid=True),
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        return vector_store, storage_context

    def init_retriever(
        self,
        collections: List[str],
        similarity_top_k: int,
        sparse_top_k: int,
    ) -> Union[VectorStoreIndex, List[VectorStoreIndex]]:
        """
        Initializes a retriever for querying one or more collections from the
        Elasticsearch vector database. Handles both single and multi-collection
        retrieval

        Parameters
        ----------
        collections: List[str]
            List of collection names to initialize for retrieval
        similarity_top_k: int
            The number of top results to retrieve through dense retrieval
        sparse_top_k: int
            The number of top results to retrieve through sparse retrieval
        num_queries: int
            The number of queries to process, which determines whether to use a
            single retriever or multiple retrievers

        Returns
        -------
        Union[VectorStoreIndex, List[VectorStoreIndex]]
            A single retriever if one collection is specified and query rewrite
            is disabled, or a list of retrievers otherwise
        """
        if not collections:
            raise Exception("No collections can be referred to")

        if len(collections) == 1:
            vector_store = ElasticsearchStore(
                es_api_key=self.api_key,
                es_url=self.url,
                index_name=collections[0],
                retrieval_strategy=AsyncDenseVectorStrategy(hybrid=True),
            )

            return VectorStoreIndex.from_vector_store(
                vector_store=vector_store
            ).as_retriever(
                similarity_top_k=similarity_top_k,
                sparse_top_k=sparse_top_k,
            )

        # Multiple collections or query rewrite enabled
        vector_retrievers = []
        for collection in collections:
            async_vector_store = ElasticsearchStore(
                es_api_key=self.api_key,
                es_url=self.url,
                index_name=collection,
                retrieval_strategy=AsyncDenseVectorStrategy(hybrid=True),
            )

            vector_retrievers.append(
                VectorStoreIndex.from_vector_store(
                    vector_store=async_vector_store,
                ).as_retriever(
                    similarity_top_k=similarity_top_k,
                    sparse_top_k=sparse_top_k,
                )
            )

        return vector_retrievers

    def get_collections(self) -> Tuple[list, list]:
        """
        Retrieves all collection (index) names from the Elasticsearch database
        and extracts unique collection prefixes by removing specified suffixes

        Returns
        -------
        Tuple[List[str], List[str]]
            - The full list of collection (index) names as stored in Elasticsearch
            - A list of unique collection prefixes with specified suffixes removed
        """
        suffixes_to_remove = {"recursive", "agentic", "semantic"}

        indices = self.client.cat.indices(format="json")
        collections = [item["index"] for item in indices]

        unique_collection_prefix = set()
        for name in collections:
            parts = name.split("_")
            if parts[-1] in suffixes_to_remove:
                unique_collection_prefix.add("_".join(parts[:-1]))
            else:
                unique_collection_prefix.add(name)

        return collections, list(unique_collection_prefix)

    def delete_collection(self, collection: str) -> bool:
        """
        Deletes a specified collection (index) from the Elasticsearch database.

        Parameters
        ----------
        collection: str
            The name of the collection (index) to delete.

        Returns
        -------
        bool
            True if deletion was successful, False otherwise.
        """
        try:
            self.client.indices.delete(index=collection)
            return True
        except Exception:
            return False
