from config import (
    VECTOR_DB_PROVIDER,
    VECTOR_DB_URL,
    EMBED_MODEL_NAME,
    EMBED_MODEL_BASE,
    EMBED_MODEL_PROVIDER,
    EMBED_DIMENSION,
    LLM_API_KEY,
    LLM_API_BASE,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_PROVIDER,
    SIMILARITY_TOP_K,
    SPARSE_TOP_K,
)
from typing import Optional, Union

from langchain_anthropic import ChatAnthropic
from langchain_deepseek import ChatDeepSeek
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from utils.vector_db.qdrant import QdrantVectorDB


class RAGAgentBase:
    __slots__ = [
        "langchain_client",
        "llamaindex_embed_model",
        "vector_db",
        "similarity_top_k",
        "sparse_top_k",
    ]

    def __init__(self):
        """
        Initializes an instance of the RAGAgentBase class by setting up
        the required components for language model (LLM) interaction, embedding
        model, and vector database retrieval

        Attributes
        ----------
        langchain_client: Union[ChatAnthropic, ChatDeepSeek, ChatGroq, ChatOpenAI]
            An instance of the LangChain client for generating responses
        llamaindex_embed_model: Union[FastEmbedEmbedding, HuggingFaceEmbedding,
                                 OllamaEmbedding, OpenAIEmbedding]
            An embedding model for indexing documents
        vector_db: Union[QdrantVectorDB]
            The vector database used for storing and retrieving vectorized
            documents
        similarity_top_k: int
            The number of top relevant documents to retrieve through dense
            retrieval
        sparse_top_k: int
            The number of top relevant documents to retrieve through sparse
            retrieval
        """
        # Init llm langchain client
        self.langchain_client = self._initialize_llm_client(
            provider=LLM_PROVIDER,
            model=LLM_MODEL,
            api_key=LLM_API_KEY,
            api_base=LLM_API_BASE,
            temperature=LLM_TEMPERATURE,
        )

        # Init embedding model
        self.llamaindex_embed_model = self._initialize_embedding_model(
            provider=EMBED_MODEL_PROVIDER,
            model_name=EMBED_MODEL_NAME,
            model_base=EMBED_MODEL_BASE,
            api_key=LLM_API_KEY,
        )
        Settings.embed_model = self.llamaindex_embed_model

        # Init vector database
        self.vector_db = self._initialize_vector_db(
            provider=VECTOR_DB_PROVIDER, db_url=VECTOR_DB_URL
        )

        # Init variable for retrieval
        self.similarity_top_k = SIMILARITY_TOP_K
        self.sparse_top_k = SPARSE_TOP_K

    def _initialize_embedding_model(
        self,
        provider: str,
        model_name: str,
        model_base: str,
        api_key: Optional[str] = None,
    ) -> Union[OllamaEmbedding, OpenAIEmbedding]:
        """
        Initializes and returns the appropriate embedding model based on the
        provider

        Parameters
        ----------
        provider: str
            The name of the embedding provider ("custom" or "openai")
        model_name : str
            The name of the model to be used for embeddings
        model_base : str
            The base URL of the model's API (used for the "custom" provider)
        api_key : Optional[str]
            The API key for accessing the embedding service (required for
            "openai" provider)

        Returns
        -------
        Union[OllamaEmbedding, OpenAIEmbedding]
            The initialized embedding model corresponding to the configured
            provider
        """
        if provider == "ollama":
            return OllamaEmbedding(model_name=model_name, base_url=model_base)

        elif provider == "openai":
            return OpenAIEmbedding(
                model=model_name, api_key=api_key, dimensions=EMBED_DIMENSION
            )

        elif provider == "fastembed":
            return FastEmbedEmbedding(model_name=model_name)

        elif provider == "huggingface":
            return HuggingFaceEmbedding(model_name=model_name)

        raise ValueError(f"Unsupported embedding model provider: {provider}")

    def _initialize_llm_client(
        self,
        provider: str,
        model: str,
        api_key: str,
        api_base: Optional[str] = None,
        temperature: Optional[float] = 0,
    ) -> Union[ChatAnthropic, ChatDeepSeek, ChatGroq, ChatOpenAI]:
        """
        Initializes and returns the appropriate LLM client based on the provider

        Parameters
        ----------
        provider : str
            The name of the LLM provider ("anthropic", "custom", or "openai")
        model : str
            The name of the model to be used for generating responses
        api_key : str
            The API key for accessing the LLM service
        api_base : Optional[str]
            The base URL of the API (used for "custom" provider)
        temperature : Optional[float]
            The temperature setting for response variability (default is 0)

        Returns
        -------
        Union[ChatAnthropic, ChatDeepSeek, ChatGroq, ChatOpenAI]
            The initialized LLM client corresponding to the configured provider
        """
        clients = {
            "anthropic": ChatAnthropic(
                model_name=model,
                api_key=api_key,
                temperature=temperature,
            ),
            "custom": ChatOpenAI(
                openai_api_base=api_base,
                openai_api_key=api_key,
                model=model,
                temperature=temperature,
            ),
            "deepseek": ChatDeepSeek(
                api_key=api_key,
                model=model,
                temperature=temperature,
            ),
            "groq": ChatGroq(
                groq_api_key=api_key,
                model_name=model,
                temperature=temperature,
            ),
            "openai": ChatOpenAI(
                openai_api_key=api_key,
                model=model,
                temperature=temperature,
            ),
        }

        return clients.get(provider)

    def _initialize_vector_db(
        self, provider: str, db_url: str
    ) -> Union[QdrantVectorDB]:
        """
        Initializes and returns a vector database instance based on the
        specified provider

        Parameters
        ----------
        provider: str
            The name of the vector database provider (e.g., "qdrant")

        Returns
        -------
        Union[QdrantVectorDB]
            An instance of the vector database
        """
        if provider.lower() == "qdrant":
            return QdrantVectorDB(url=db_url)

        raise NotImplementedError(f"Provider '{provider}' is not supported.")
