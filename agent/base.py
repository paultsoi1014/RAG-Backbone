import logging
import uuid

from typing import Optional, Union

from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from langfuse.langchain import CallbackHandler

from config import Config
from vector_db import QdrantVectorDB, ElasticVectorDB

logging.basicConfig(level=logging.INFO)


class RAGAgentBase:
    __slots__ = ("config", "_lc_client", "_li_embed_model", "_vector_db")

    """
    Base class for a Retrieval-Augmented Generation (RAG) agent, managing
    LLM clients, embedding models, and vector database connections.

    This class serves as the foundational component for RAG operations,
    centralizing configuration management and the initialization of
    ingestion, retrieval and document search component

    Attributes
    ----------
    config: Config
        The configuration object loaded from environment variables
    _lc_client: Union[ChatDeepseek, ChatOpenAI]
        An instance of the langchain LLM client for document ingestions 
    _li_embed_model: Union[FastEmbedEmbedding, HuggingFaceEmbedding,
                    OllamaEmbedding, OpenAIEmbedding]
        The initialized embedding model instance used for document indexing and 
        query embedding
    _vector_db: Union[QdrantVectorDB, ElasticDB]
        The vector database used for storing and retrieving vectorized documents
    """

    def __init__(self):
        """
        Initializes the RAG agent by loading configuration and setting up
        core components

        This method performs the following steps:
        1. Loads the application configuration from the environment
        2. Initializes the primary LLM client (langchain).
        3. Initializes the embedding model and registers it with global Settings
        4. Connects to the configured vector database (Qdrant or Elastic)
        """
        # Load configuration
        self.config = Config.from_env()

        # Init llm langchain client
        self._lc_client = self._init_llm_client(
            provider=self.config.llm.provider,
            model=self.config.llm.model,
            api_key=self.config.llm.api_key,
            api_base=self.config.llm.api_base,
            temperature=self.config.llm.temperature,
        )

        # Init embedding model
        self._li_embed_model = self._init_embedding_model(
            provider=self.config.embed_model.provider,
            model_name=self.config.embed_model.model,
            model_base=self.config.embed_model.base_url,
            api_key=self.config.embed_model.api_key,
            embedding_dimension=self.config.embed_model.dimension,
        )
        Settings.embed_model = self._li_embed_model

        # Init vector database
        self._vector_db = self._init_vector_db(
            provider=self.config.vector_db.provider,
            api_key=self.config.vector_db.api_key,
            db_url=self.config.vector_db.url,
        )

    def _init_embedding_model(
        self,
        provider: str,
        model_name: str,
        model_base: str,
        api_key: Optional[str] = None,
        embedding_dimension: Optional[int] = None,
    ) -> Union[
        OllamaEmbedding, OpenAIEmbedding, FastEmbedEmbedding, HuggingFaceEmbedding
    ]:
        """
        Initializes and returns the appropriate embedding model based on the
        provider

        Parameters
        ----------
        provider: str
            The name of the embedding provider ("ollama", "openai", "fastembed",
            and "huggingface")
        model_name: str
            The name of the model to be used for embeddings
        model_base: str
            The base URL of the model's API (used for the "custom" provider)
        api_key: Optional[str]
            The API key for accessing the embedding service (required for
            "openai" provider)
        embedding_dimension: Optional[int]
            The dimension of embedding, default to None

        Returns
        -------
        Union[OllamaEmbedding, OpenAIEmbedding, FastEmbedEmbedding, HuggingFaceEmbedding]
            The initialized embedding model corresponding to the configured
            provider
        """
        if provider == "ollama":
            return OllamaEmbedding(model_name=model_name, base_url=model_base)

        elif provider == "openai":
            return OpenAIEmbedding(
                model=model_name, api_key=api_key, dimensions=embedding_dimension
            )

        elif provider == "fastembed":
            return FastEmbedEmbedding(model_name=model_name)

        elif provider == "huggingface":
            return HuggingFaceEmbedding(model_name=model_name)

        raise ValueError(f"Unsupported embedding model provider: {provider}")

    def _init_llm_client(
        self,
        provider: str,
        model: str,
        api_key: str,
        api_base: Optional[str] = None,
        temperature: Optional[float] = 0.0,
    ) -> Union[ChatDeepSeek, ChatOpenAI, ChatOllama]:
        """
        Initializes and returns the appropriate LLM client based on the provider

        Parameters
        ----------
        provider: str
            The name of the LLM provider ("custom", "deepseek", or "openai")
        model: str
            The name of the model to be used for generating responses
        api_key: str
            The API key for accessing the LLM service
        api_base: Optional[str]
            The base URL of the API (used for "custom" provider)
        temperature: Optional[float]
            The temperature setting for response variability (default is 0.0)

        Returns
        -------
        Union[ChatDeepSeek, ChatOpenAI, ChatOllama]
            The initialized LLM client corresponding to the configured provider
        """
        clients = {
            "custom": ChatOpenAI(
                openai_api_base=api_base,
                openai_api_key=api_key,
                model=model,
                temperature=temperature,
            ),
            "deepseek": ChatDeepSeek(
                model=model,
                api_key=api_key,
                temperature=temperature,
            ),
            "openai": ChatOpenAI(
                openai_api_key=api_key,
                model=model,
                temperature=temperature,
            ),
            "ollama": ChatOllama(
                model=model,
                base_url=api_base,
                temperature=temperature,
            ),
        }

        return clients.get(provider)

    def _init_vector_db(
        self, provider: str, db_url: str, api_key: Optional[str] = None
    ) -> Union[QdrantVectorDB, ElasticVectorDB]:
        """
        Initializes and returns a vector database instance based on the
        specified provider

        Parameters
        ----------
        provider: str
            The name of the vector database provider (e.g., "qdrant", "elastic")
        db_url: str
            The URL endpoint of the vector database
        api_key: Optional[str]
            API key required for providers that need authentication. Defaults to
            None

        Returns
        -------
        Union[QdrantVectorDB, ElasticVectorDB]
            An instance of the vector database
        """
        if provider.lower() == "qdrant":
            return QdrantVectorDB(url=db_url)
        elif provider.lower() == "elastic":
            return ElasticVectorDB(url=db_url, api_key=api_key)

        raise NotImplementedError(f"Provider '{provider}' is not supported.")

    def init_langfuse_tracing(
        self,
        public_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        host: Optional[str] = None,
        session_id: Optional[str] = str(uuid.uuid4()),
    ) -> CallbackHandler:
        """
        Initializes the Langfuse tracing callback handler for observability and
        performance monitoring

        Parameters
        ----------
        public_key: Optional[str]
            The public API key for authenticating with the Langfuse service.
            Defaults to the value of the LANGFUSE_PUBLIC_KEY constant
        secret_key: Optional[str]
            The secret API key for authenticating with the Langfuse service.
            Defaults to the value of the LANGFUSE_SECRET_KEY constant
        host: Optional[str]
            The base URL of the Langfuse API. Defaults to the value of the
            LANGFUSE_HOST constant
        session_id: Optional[str]
            A unique identifier for the tracing session. Defaults to a newly
            generated UUID

        Returns
        -------
        Optional[CallbackHandler]
            An initialized Langfuse `CallbackHandler` if tracing is enabled in
            the configuration (`self.config.langfuse.enable`); otherwise, None
        """
        if self.config.langfuse.enable:
            # Define values for public key, secret key and host
            public_key = (
                self.config.langfuse.public_key if public_key is None else public_key
            )
            secret_key = (
                self.config.langfuse.secret_key if secret_key is None else secret_key
            )
            host = self.config.langfuse.host if host is None else host

            return CallbackHandler(
                public_key=public_key,
                secret_key=secret_key,
                host=host,
                session_id=session_id,
            )
        else:
            return None

    def get_lc_client(self) -> Union[ChatDeepSeek, ChatOpenAI]:
        """
        Retrieves the initialized langchain client

        Returns
        -------
        Union[ChatDeepSeek, ChatOpenAI, ChatOllama]
            The active langchain client instance configured for the agent
        """
        return self._lc_client

    def get_li_embed_model(
        self,
    ) -> Union[
        OllamaEmbedding, OpenAIEmbedding, FastEmbedEmbedding, HuggingFaceEmbedding
    ]:
        """
        Retrieves the initialized LlamaIndex embedding model

        Returns
        -------
        Union[
        OllamaEmbedding, OpenAIEmbedding, FastEmbedEmbedding, HuggingFaceEmbedding
        ]
            The active embedding model instance used for document indexing and
            query embedding
        """
        return self._li_embed_model

    def get_vector_db(self) -> Union[QdrantVectorDB, ElasticVectorDB]:
        """
        Retrieves the initialized vector database instance

        Returns
        -------
        Union[QdrantVectorDB, ElasticVectorDB]
            The active vector database client instance
        """
        return self._vector_db

    def get_parser_settings(self) -> dict:
        """
        Retrieves the current configuration settings for document parsing

        Returns
        -------
        dict
            A dictionary containing parser parameters with the following keys:
            - "type" (str): The type or name of the document parser
              being used
            - "url" (str): The endpoint or URL associated with the parser
        """

        return {"type": self.config.parser.type, "url": self.config.parser.url}

    def get_retrieval_settings(self) -> dict:
        """
        Retrieves the current configuration settings for document retrieval

        Returns
        -------
        dict
            A dictionary containing retrieval parameters with the following keys:
            - "fuse_top_k" (int): The number of top documents to return after
              fusion
            - "dense_top_k" (int): The limit for dense vector (semantic)
              retrieval
            - "sparse_top_k" (int): The limit for sparse keyword (lexical)
              retrieval
            - "enable_query_preprocess" (bool): Flag indicating whether query
               preprocessing or expansion is enabled
            - "num_queries" (int): The number of queries to generate if query
              expansion is used
            - "repacking_strategy" (str): The strategy used for packing or
              ordering retrieved document chunks within the context window
        """
        return {
            "fuse_top_k": self.config.retrieval.fuse_top_k,
            "dense_top_k": self.config.retrieval.dense_top_k,
            "sparse_top_k": self.config.retrieval.sparse_top_k,
            "enable_query_preprocess": self.config.retrieval.enable_query_preprocess,
            "num_queries": self.config.retrieval.num_queries,
            "repacking_strategy": self.config.retrieval.repacking_strategy,
        }
