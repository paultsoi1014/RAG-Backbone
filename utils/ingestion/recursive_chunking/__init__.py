import logging
from typing import List, Optional, Union

from langchain_text_splitters import RecursiveCharacterTextSplitter

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.ingestion import IngestionPipeline, DocstoreStrategy
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.readers.string_iterable import StringIterableReader
from llama_index.vector_stores.qdrant import QdrantVectorStore


class RecursiveChunker:
    __slots__ = [
        "vector_store",
        "storage_context",
        "text_splitter",
    ]

    def __init__(
        self,
        vector_store: Union[QdrantVectorStore],
        storage_context: StorageContext,
        chunk_size: Optional[int] = 100,
        chunk_overlap: Optional[int] = 20,
        separators: Optional[List[str]] = ["\n", "\n\n"],
    ):
        """
        A class that handles chunking of documents by splitting documents into
        smaller chunks based on separator and other configurable parameters

        Attributes
        ----------
        vector_store : Union[QdrantVectorStore]
            The vector store used for storing chunked documents
        storage_context : StorageContext
            Context for storage configuration and management
        text_splitter : RecursiveCharacterTextSplitter
            The text splitter used for chunking the document
        enable_contextual_chunking : bool
            A flag to enable contextual chunking (default is False)

        Parameters
        ----------
        chunk_size : Optional[int]
            The maximum size of each chunk (default is 100)
        chunk_overlap : Optional[int]
            The overlap between each chunk (default is 20)
        separators : Optional[List[str]]
            A list of separators used for splitting the document
        """
        self.vector_store = vector_store
        self.storage_context = storage_context

        # Define the text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            separators=separators,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

    def _pipeline(self) -> IngestionPipeline:
        """
        Defines the ingestion pipeline configuration for vector store insertion

        Returns
        -------
        IngestionPipeline
            An ingestion pipeline configured for the specified vector store
        """
        return IngestionPipeline(
            vector_store=self.vector_store,
            docstore_strategy=DocstoreStrategy.UPSERTS,
            docstore=SimpleDocumentStore(),
        )

    def __call__(self, document: str) -> VectorStoreIndex:
        """
        Chunk the document and ingest the chunks into the vector store

        Parameters
        ----------
        document: str
            The document to be chunked and ingested into the vector store

        Returns
        -------
        VectorStoreIndex
            The vector store index built from the ingested document chunks
        """
        # Split the document into chunks
        chunks = self.text_splitter.split_text(document)

        # Load data from an iterable of strings to Document type
        chunk_documents = StringIterableReader().load_data(chunks)

        # Define ingestion pipeline and run
        nodes = self._pipeline().run(documents=chunk_documents, show_progress=True)
        logging.info(f"Ingested {len(nodes)} Nodes")

        # build llama index
        return VectorStoreIndex.from_documents(
            chunk_documents, storage_context=self.storage_context
        )
