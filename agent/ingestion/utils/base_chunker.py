import logging

from abc import ABC
from typing import Any, Dict, List, Optional, Union

from llama_index.core.ingestion import IngestionPipeline, DocstoreStrategy
from llama_index.core.schema import Document
from llama_index.core.storage.docstore import SimpleDocumentStore

from vector_db import QdrantVectorDB, ElasticVectorDB

logger = logging.getLogger(__name__)


class BaseChunker(ABC):
    """
    Abstract base class for all chunking strategies. Provides common
    functionality for creating LlamaIndex Documents with metadata and
    running ingestion pipelines.

    Attributes
    ----------
    collection: str
        Base name of the collection where documents will be stored
    vector_store: Union[QdrantVectorDB, ElasticVectorDB]
        The vector store used for storing chunked documents
    storage_context: StorageContext
        Context for storage configuration and management
    """

    __slots__ = ("collection", "vector_store", "storage_context")

    def __init__(
        self,
        vector_db: Union[QdrantVectorDB, ElasticVectorDB],
        collection_name: str,
        chunking_suffix: str,
    ):
        """
        Initialize the BaseChunker

        Parameters
        ----------
        vector_db: Union[QdrantVectorDB, ElasticVectorDB]
            Vector database client for document indexing and retrieval
        collection_name: str
            Base name of the collection where documents will be stored
        chunking_suffix: str
            Suffix to append to collection name (e.g., "_semantic", "_recursive")
        """
        self.collection = collection_name

        revised_collection_name = f"{self.collection}{chunking_suffix}"
        self.vector_store, self.storage_context = vector_db.init_ingestion(
            revised_collection_name
        )

    def _create_ingestion_pipeline(self) -> IngestionPipeline:
        """
        Create a standardized ingestion pipeline for vector store insertion

        Returns
        -------
        IngestionPipeline
            An ingestion pipeline configured for the vector store
        """
        return IngestionPipeline(
            vector_store=self.vector_store,
            docstore_strategy=DocstoreStrategy.UPSERTS,
            docstore=SimpleDocumentStore(),
        )

    def create_documents_with_metadata(
        self,
        chunks: List[str],
        base_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        Create LlamaIndex Document objects from text chunks with standardized
        metadata

        Parameters
        ----------
        chunks: List[str]
            List of text chunks to convert to Document objects
        base_metadata: Optional[Dict[str, Any]]
            Base metadata to apply to all chunks. If None, creates empty dict

        Returns
        -------
        List[Document]
            List of LlamaIndex Document objects with metadata
        """
        if base_metadata is None:
            base_metadata = {}

        chunk_documents = []
        for i, chunk_text in enumerate(chunks):
            # Create metadata for this specific chunk
            chunk_metadata = base_metadata.copy()
            chunk_metadata["chunk_index"] = i
            chunk_metadata["total_chunks"] = len(chunks)
            chunk_metadata["chunking_strategy"] = self.__class__.__name__.replace(
                "Chunker", ""
            ).lower()
            chunk_metadata["collection"] = self.collection

            # Create LlamaIndex Document with metadata
            doc = Document(text=chunk_text, metadata=chunk_metadata)
            chunk_documents.append(doc)

        return chunk_documents

    def ingest_documents(
        self, documents: List[Document], show_progress: Optional[bool] = True
    ) -> list:
        """
        Ingest documents into the vector store using the configured pipeline

        Parameters
        ----------
        documents: List[Document]
            List of LlamaIndex Document objects to ingest
        show_progress: Optional[bool]
            Whether to show progress during ingestion. Default is True

        Returns
        -------
        list
            List of ingested nodes
        """
        pipeline = self._create_ingestion_pipeline()
        nodes = pipeline.run(documents=documents, show_progress=show_progress)

        logger.info(f"Ingested {len(nodes)} nodes using {self.__class__.__name__}")

        return nodes

    def process_and_ingest(
        self,
        chunks: List[str],
        base_metadata: Optional[Dict[str, Any]] = None,
        show_progress: Optional[bool] = True,
    ) -> list:
        """
        Convenience method to create documents and ingest them in one step

        Parameters
        ----------
        chunks: List[str]
            List of text chunks to process and ingest
        base_metadata: Optional[Dict[str, Any]]
            Base metadata to apply to all chunks. Defaults to None
        show_progress: Optional[bool]
            Whether to show progress during ingestion. Defaults to True

        Returns
        -------
        list
            List of ingested nodes
        """
        documents = self.create_documents_with_metadata(
            chunks=chunks, base_metadata=base_metadata
        )

        return self.ingest_documents(documents, show_progress=show_progress)
