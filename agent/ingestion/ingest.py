import logging
import os
import tempfile
import uuid

from typing import List, Optional

from agent.base import RAGAgentBase
from .utils.pipeline import IngestionPipeline

logger = logging.getLogger(__name__)


class IngestionAgent:
    """
    Handles the ingestion of documents and raw text into a vector database

    This class serves as an interface for the ingestion pipeline, providing
    synchronous and asynchronous methods to process files or direct text input
    using various chunking strategies

    Attributes
    ----------
    agent_base: RAGAgentBase
        The base agent containing the vector database connection, LLM clients,
        and embedding models required for the ingestion process.
    """

    __slots__ = "agent_base"

    def __init__(self, agent_base: RAGAgentBase):
        """
        Initializes the IngestionAgent with a configured base agent

        Parameters
        ----------
        agent_base: RAGAgentBase
            The base RAG agent instance
        """
        self.agent_base = agent_base

    def folder_ingestion(
        self,
        folder_path: str,
        collection_name: str,
        mode: Optional[List[str]] = ["recursive"],
    ):
        """
        Ingests files from a specified folder into the vector database.

        This method processes documents from the specified folder, chunks them
        according to the requested modes, and stores the resulting embeddings
        in a vector database for later retrieval.

        Parameters
        ----------
        folder_path: str
            The directory containing the files to be ingested
        collection_name: str
            The name of the destination collection in the vector database.
        mode: Optional[List[str]], default=["recursive"]
            The chunking strategies to apply. Options include 'agentic',
            'recursive', and/or 'semantic'
        """
        # Init RAG ingestion pipeline
        rag_ingestion = IngestionPipeline(
            agent_base=self.agent_base, collection_name=collection_name
        )
        rag_ingestion.ingest(folder_path, mode=mode)

    async def afolder_ingestion(
        self,
        folder_path: str,
        collection_name: str,
        mode: Optional[List[str]] = ["recursive"],
    ):
        """
        Asynchronously ingests files from a specified folder into the vector
        database

        This method processes documents from the specified folder, chunks them
        according to the requested modes, and stores the resulting embeddings
        in a vector database for later retrieval

        Parameters
        ----------
        folder_path: str
            The directory containing the files to be ingested
        collection_name: str
            The name of the destination collection in the vector database
        mode: Optional[List[str]], default=["recursive"]
            The chunking strategies to apply. Options include 'agentic',
            'recursive', and/or 'semantic'
        """
        # Init RAG ingestion pipeline
        rag_ingestion = IngestionPipeline(
            agent_base=self.agent_base, collection_name=collection_name
        )
        await rag_ingestion.aingest(folder_path, mode=mode)

    def text_ingestion(
        self, text: str, collection_name: str, mode: Optional[List[str]] = ["recursive"]
    ):
        """
        Ingests raw text into the vector database by temporarily saving it to a
        file

        Parameters
        ----------
        text: str
            The raw text content to be ingested
        collection_name: str
            The name of the destination collection in the vector database
        mode: Optional[List[str]], default=["recursive"]
            The chunking strategies to apply. Options include 'agentic',
            'recursive', and/or 'semantic'
        """
        # create temporarily directory
        temp_dir = tempfile.mkdtemp(prefix="rag_text_ingest_")

        # Init RAG ingestion pipeline
        rag_ingestion = IngestionPipeline(
            agent_base=self.agent_base, collection_name=collection_name
        )

        try:
            # Create filename and filepath
            filename = f"text_input_{uuid.uuid4().hex[:8]}.txt"
            filepath = os.path.join(temp_dir, filename)

            # Save text to temporary file
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(text)

            # file ingestion
            rag_ingestion.ingest(folder_path=temp_dir, mode=mode)

        finally:
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
                if os.path.exists(temp_dir):
                    os.rmdir(temp_dir)

            except OSError as e:
                logger.warning(f"Warning: Could not clean up temporary files: {e}")

    async def atext_ingestion(
        self, text: str, collection_name: str, mode: Optional[List[str]] = ["recursive"]
    ):
        """
        Asynchronously ingests raw text into the vector database by temporarily
        saving it to a file

        Parameters
        ----------
        text: str
            The raw text content to be ingested
        collection_name: str
            The name of the destination collection in the vector database
        mode: Optional[List[str]], default=["recursive"]
            The chunking strategies to apply. Options include 'agentic',
            'recursive', and/or 'semantic'
        """
        # create temporarily directory
        temp_dir = tempfile.mkdtemp(prefix="rag_text_ingest_")

        # Init RAG ingestion pipeline
        rag_ingestion = IngestionPipeline(
            agent_base=self.agent_base, collection_name=collection_name
        )

        try:
            # Create filename and filepath
            filename = f"text_input_{uuid.uuid4().hex[:8]}.txt"
            filepath = os.path.join(temp_dir, filename)

            # Save text to temporary file
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(text)

            # file ingestion
            await rag_ingestion.aingest(folder_path=temp_dir, mode=mode)

        finally:
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
                if os.path.exists(temp_dir):
                    os.rmdir(temp_dir)

            except OSError as e:
                logger.warning(f"Warning: Could not clean up temporary files: {e}")
