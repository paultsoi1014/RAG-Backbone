import asyncio
from typing import List, Optional, Union

from agent.base import RAGAgentBase
from agent.retrieve import RetrieveAgent
from agent.ingestion import IngestionAgent
from agent.search import SearchRelevancyAgent


class RAGAgent:
    __slots__ = ("agent_base", "ingestion_agent", "retrieve_agent", "search_agent")

    def __init__(self, agent_base: Optional[RAGAgentBase] = None):
        """
        A comprehensive agent class that integrates capabilities for both
        document ingestion, document search, and response generation within a
        Retrieval-Augmented Generation (RAG) framework

        Attributes
        ----------
        agent_base: Optional[RAGAgentBase]
            The base RAG agent instance containing shared components and
            configuration. If `None`, a new instance is created
        ingestion_agent: IngestionAgent
            The agent responsible for document ingestion tasks
        retrieve_agent: RetrieverAgent
            The agent responsible for document retrieval tasks
        search_agent: SearchRelevancyAgent
            The agent responsible for providing document relevancy
        """
        self.agent_base = agent_base if agent_base else RAGAgentBase()

        # Initialize sub-agents for ingestion, retrieval and search
        self.ingestion_agent = IngestionAgent(self.agent_base)
        self.retrieve_agent = RetrieveAgent(self.agent_base)
        self.search_agent = SearchRelevancyAgent(self.agent_base)

    def folder_ingestion(
        self,
        folder_path: str,
        collection_name: str,
        mode: Optional[List[str]] = ["recursive"],
    ) -> None:
        """
        Ingests files from a specified folder into the vector database

        This method processes documents from the specified folder, chunking them
        into recursively or semantically, and stores them in a vector database
        for later retrieval

        Parameters
        ----------
        folder_dir : str
            The directory containing the files to be ingested
        collection_name: str
            The name of the collection for the ingested documents
        mode: Optional[List[str]]
            The mode of chunking to be used. Options are 'agentic', 'recursive'
            and/or 'semantic'. Default is using recursive chunking method
        """
        return self.ingestion_agent.folder_ingestion(folder_path, collection_name, mode)

    async def afolder_ingestion(
        self,
        folder_path: str,
        collection_name: str,
        mode: Optional[List[str]] = ["recursive"],
    ) -> None:
        """
        Asychronously Ingests files from a specified folder into the vector
        database

        This method processes documents from the specified folder, chunking them
        into recursively or semantically, and stores them in a vector database
        for later retrieval

        Parameters
        ----------
        folder_dir: str
            The directory containing the files to be ingested
        collection_name: str
            The name of the collection for the ingested documents
        mode: Optional[List[str]]
            The mode of chunking to be used. Options are 'agentic', 'recursive'
            and/or 'semantic'. Default is using recursive chunking method
        """
        return await self.ingestion_agent.afolder_ingestion(
            folder_path, collection_name, mode
        )

    def text_ingestion(
        self, text: str, collection_name: str, mode: Optional[List[str]] = ["recursive"]
    ) -> None:
        """
        Ingests raw text into the vector database by first saving it to a
        temporary file

        Parameters
        ----------
        text: str
            The raw text content to be ingested
        collection_name: str
            The name of the collection for the ingested documents
        mode: Optional[List[str]]
            The mode of chunking to be used. Options are 'agentic', 'recursive'
            and/or 'semantic'. Default is using recursive chunking method
        """
        return self.ingestion_agent.text_ingestion(text, collection_name, mode)

    async def atext_ingestion(
        self, text: str, collection_name: str, mode: Optional[List[str]] = ["recursive"]
    ) -> None:
        """
        Ingests raw text into the vector database by first saving it to a
        temporary file

        Parameters
        ----------
        text: str
            The raw text content to be ingested
        collection_name: str
            The name of the collection for the ingested documents
        mode: Optional[List[str]]
            The mode of chunking to be used. Options are 'agentic', 'recursive'
            and/or 'semantic'. Default is using recursive chunking method
        """
        return await self.ingestion_agent.atext_ingestion(text, collection_name, mode)

    def retrieve_information(
        self,
        query: Union[str, List[str]],
        collections: Optional[List[str]] = None,
        metadata: Optional[dict] = {},
        recursive_only: Optional[bool] = False,
    ) -> dict:
        """
        Retrieve user query's relevant information from vector database

        Parameters
        ----------
        query: Union[str, List[str]]
            The user query for which document relevancy needs to be evaluated
        collections: Optional[List[str]]
            List of collection names to search within; if None, searches all
            collections. Default is `None`
        metadata: Optional[dict]
            Additional metadata to include to assist in collection selection.
            Default is empty dictionary
        recursive_only: Optional[bool], default=False
            If True, only considers collections with the `_recursive` suffix

        Returns
        -------
        dict
            A dictionary containing the relevant text retrieved from the
            database and A list of relevant document collections
        """
        try:
            loop = asyncio.get_running_loop()
            retrieved_node, retrieved_documents = asyncio.create_task(
                self.retrieve_agent.aretrieve_info(
                    query,
                    collections_prefix=collections,
                    metadata=metadata,
                    recursive_only=recursive_only,
                )
            )
        except RuntimeError:
            retrieved_node, retrieved_documents = asyncio.run(
                self.retrieve_agent.aretrieve_info(
                    query,
                    collections_prefix=collections,
                    metadata=metadata,
                    recursive_only=recursive_only,
                )
            )

        return {"nodes": retrieved_node, "documents": retrieved_documents}

    async def aretrieve_information(
        self,
        query: Union[str, List[str]],
        collections: Optional[List[str]] = None,
        metadata: Optional[dict] = {},
        recursive_only: Optional[bool] = False,
    ):
        """
        Asychronously Retrieve user query's relevant information from vector
        database

        Parameters
        ----------
        query: Union[str, List[str]]
            The user query for which document relevancy needs to be evaluated
        collections: Optional[List[str]]
            List of collection names to search within; if None, searches all
            collections. Default is `None`
        metadata: Optional[dict]
            Additional metadata to include to assist in collection selection.
            Default is empty dictionary
        recursive_only: Optional[bool], default=False
            If True, only considers collections with the `_recursive` suffix

        Returns
        -------
        dict
            A dictionary containing the relevant text retrieved from the
            database and A list of relevant document collections
        """
        retrieved_node, retrieved_documents = await self.retrieve_agent.aretrieve_info(
            query,
            collections_prefix=collections,
            metadata=metadata,
            recursive_only=recursive_only,
        )

        return {"nodes": retrieved_node, "documents": retrieved_documents}

    def search_document_relevancy(
        self,
        query: str,
        collections: Optional[List[str]] = None,
        metadata: Optional[dict] = {},
    ) -> dict:
        """
        Execute the RAG search workflow for a given query and return document
        relevancy scores

        Parameters
        ----------
        query: str
            The user query for which document relevancy needs to be evaluated
        collections: Optional[List[str]]
            List of collection names to search within; if None, searches all
            collections. Default is `None`
        metadata: Optional[dict]
            Additional metadata to include to assist in collection selection.
            Default is empty dictionary

        Returns
        -------
        dict
            A dictionary mapping document names to their relevancy scores as
            percentages
        """
        try:
            loop = asyncio.get_running_loop()
            return asyncio.create_task(
                self.search_agent.asearch_relevancy(
                    query, collections=collections, metadata=metadata
                )
            )
        except RuntimeError:
            return asyncio.run(
                self.search_agent.asearch_relevancy(
                    query,
                    collections=collections,
                    metadata=metadata,
                )
            )

    async def asearch_document_relevancy(
        self,
        query: str,
        collections: Optional[List[str]] = None,
        metadata: Optional[dict] = {},
    ) -> dict:
        """
        Asynchronously Execute the RAG search workflow for a given query and
        return document relevancy scores

        Parameters
        ----------
        query: str
            The user query for which document relevancy needs to be evaluated
        collections: Optional[List[str]]
            List of collection names to search within; if None, searches all
            collections. Default is `None`
        metadata: Optional[dict]
            Additional metadata to include to assist in collection selection.
            Default is empty dictionary

        Returns
        -------
        dict
            A dictionary mapping document names to their relevancy scores as
            percentages
        """
        relevancy = await self.search_agent.asearch_relevancy(
            query, collections=collections, metadata=metadata
        )

        return relevancy
