import logging
from typing import List, Optional, Union

from langchain_text_splitters import RecursiveCharacterTextSplitter

from agent.ingestion.utils.base_chunker import BaseChunker
from vector_db import QdrantVectorDB, ElasticVectorDB

logger = logging.getLogger(__name__)


class RecursiveChunker(BaseChunker):
    """
    Document chunker that recursively splits text using configurable separators.
    It splits documents based on a hierarchy of separators while maintaining
    specified chunk sizes and overlaps

    Attributes
    ----------
    text_splitter: RecursiveCharacterTextSplitter
        The text splitter used for chunking the document
    """

    __slots__ = "text_splitter"

    def __init__(
        self,
        vector_db: Union[QdrantVectorDB, ElasticVectorDB],
        collection_name: str,
        chunk_size: Optional[int] = 100,
        chunk_overlap: Optional[int] = 20,
        separators: Optional[List[str]] = ["\n", "\n\n"],
    ):
        """
        Initialize the RecursiveChunker with vector database and chunking
        parameters

        Sets up the recursive text splitter with the specified configuration and
        initializes the parent BaseChunker with the vector database connection

        Parameters
        ----------
        vector_db: Union[QdrantVectorDB, ElasticVectorDB]
            Vector database client for document indexing and retrieval
        collection_name: str
            Name of the collection where documents will be stored
        chunk_size: Optional[int]
            The maximum size of each chunk (default is 100)
        chunk_overlap: Optional[int]
            The overlap between each chunk (default is 20)
        separators: Optional[List[str]]
            A list of separators used for splitting the document
        """
        super().__init__(vector_db, collection_name, "_recursive")

        # Define the text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            separators=separators,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

    def chunk(self, document: str) -> None:
        """
        Chunk the document using LangChain and ingest the chunks into the vector
        store with metadata

        Parameters
        ----------
        document: str
            The document to be chunked and ingested into the vector store
        """
        # Split the document into chunks
        chunks = self.text_splitter.split_text(document)

        # Prepare metadata
        nodes = self.process_and_ingest(chunks=chunks)
        logger.info(f"Ingested {len(nodes)} Nodes")
