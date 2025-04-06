from typing import Union, Optional

from langchain_anthropic import ChatAnthropic
from langchain_deepseek import ChatDeepSeek
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from llama_index.core import StorageContext
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore

from langfuse.callback import CallbackHandler

from utils.ingestion.semantic_chunking import SemanticChunker
from utils.ingestion.recursive_chunking import RecursiveChunker
from utils.parsers.document_parse import DocumentParser


class RAGIngestionPipeline:
    __slots__ = [
        "lc_client",
        "li_embed_model",
        "lf_handler",
        "vector_store",
        "storage_context",
    ]

    def __init__(
        self,
        lc_client: Union[ChatAnthropic, ChatDeepSeek, ChatGroq, ChatOpenAI],
        li_embed_model: Union[
            FastEmbedEmbedding, HuggingFaceEmbedding, OllamaEmbedding, OpenAIEmbedding
        ],
        lf_handler: CallbackHandler,
        vector_store: Union[QdrantVectorStore],
        storage_context: StorageContext,
    ):
        """
        A pipeline for processing and chunking documents for Retrieval-Augmented
        Generation (RAG)

        Attributes
        ----------
        lc_client: Union[ChatAnthropic, ChatDeepSeek, ChatGroq, ChatOpenAI]
            The LangChain language model client
        li_embed_model: Union[FastEmbedEmbedding, HuggingFaceEmbedding,
                        OllamaEmbedding, OpenAIEmbedding]
            The embedding model used for generating embeddings before storing
            into vector database
        lf_handler: langfuse.callback.CallbackHandler
            Handler for logging and tracing LLM interactions via Langfuse
        vector_store: Union[QdrantVectorStore]
            The vector store used for storing document embeddings
        storage_context: StorageContext
            The context for storing data during processing

        Methods
        -------
        __call__(folder_path: str, mode: str) -> Union[RecursiveChunker, SemanticChunker]:
            Processes documents from the specified folder and chunks them based on
            the given mode
        """
        # init langchain llm client
        self.lc_client = lc_client

        # init llamaindex embedding model
        self.li_embed_model = li_embed_model

        # Init langfuse handler
        self.lf_handler = lf_handler

        # init vector db store and storage context
        self.vector_store = vector_store
        self.storage_context = storage_context

    def __call__(
        self, folder_path: str, mode: Optional[str] = "recursive"
    ) -> Union[RecursiveChunker, SemanticChunker]:
        """
        Processes documents from the specified folder and chunks them based on
        the mode selected

        Parameters
        ----------
        folder_path: str
            The path to the folder containing documents to be processed
        mode: Optional[str]
            The mode of chunking to be used. Options are 'recursive' or
            'semantic'. Default is using recursive chunking method

        Returns
        -------
        Union[RecursiveChunker, SemanticChunker]
            An instance of the appropriate chunker based on the specified mode
        """
        # Document parsing
        document_parser = DocumentParser(self.lc_client, lf_handler=self.lf_handler)
        parsed_text = document_parser.parse(folder_path)

        # Check if it is empty parsed text
        if not parsed_text:
            raise ValueError("No content found in the specified folder")

        if mode == "semantic":
            semantic_chunker = SemanticChunker(
                embed_model=self.li_embed_model,
                vector_store=self.vector_store,
                storage_context=self.storage_context,
                breakpoint_percentile_threshold=95,
                buffer_size=1,
            )
            return semantic_chunker(document=parsed_text)

        elif mode == "recursive":
            recursive_chunker = RecursiveChunker(
                vector_store=self.vector_store,
                storage_context=self.storage_context,
            )
            return recursive_chunker(document=parsed_text)

        else:
            raise ValueError(f"Invalid chunking strategy selected: {mode}")
