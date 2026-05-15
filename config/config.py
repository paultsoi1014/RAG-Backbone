import os

from dataclasses import dataclass
from dotenv import load_dotenv
from typing import Optional

if os.getenv("ENVIRONMENT", "development") == "development":
    load_dotenv(dotenv_path=".env", override=True)


@dataclass
class AppConfig:
    """Application configuration settings"""

    api_port: int
    mcp_port: int
    chat_mode: bool = False

    @classmethod
    def from_env(cls) -> "AppConfig":

        api_port = int(os.getenv("API_PORT", 8008))
        chat_mode = os.getenv("CHAT_MODE", "false").lower() == "true"
        mcp_port = int(os.getenv("MCP_PORT", 8009))

        return cls(api_port=api_port, chat_mode=chat_mode, mcp_port=mcp_port)


@dataclass
class EmbedModelConfig:
    """Embedding model configuration settings"""

    api_key: str
    base_url: str
    dimension: int
    model: str
    provider: str

    @classmethod
    def from_env(cls) -> "EmbedModelConfig":
        return cls(
            api_key=os.getenv("EMBED_MODEL_API_KEY", "EMPTY"),
            base_url=os.getenv("EMBED_MODEL_BASE", "http://192.168.3.236:11434"),
            dimension=int(os.getenv("EMBED_DIMENSION", "1024")),
            model=os.getenv("EMBED_MODEL_NAME", "bge-m3"),
            provider=os.getenv("EMBED_MODEL_PROVIDER", "ollama"),
        )


@dataclass
class LangfuseConfig:
    """Langfuse tracing configuration settings"""

    enable: bool
    host: str
    public_key: Optional[str] = None
    secret_key: Optional[str] = None

    @classmethod
    def from_env(cls) -> "LangfuseConfig":
        public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "None")
        secret_key = os.getenv("LANGFUSE_SECRET_KEY", "None")

        return cls(
            enable=os.getenv("LANGFUSE_ENABLE", "false").lower() == "true",
            host=os.getenv("LANGFUSE_HOST", "http://192.168.100.3:3000"),
            public_key=None if public_key == "None" else public_key,
            secret_key=None if secret_key == "None" else secret_key,
        )


@dataclass
class LLMConfig:
    """LLM configuration settings"""

    api_base: str
    api_key: str
    model: str
    provider: str
    temperature: float

    @classmethod
    def from_env(cls) -> "LLMConfig":
        return cls(
            api_key=os.getenv("LLM_API_KEY", "EMPTY"),
            api_base=os.getenv("LLM_API_BASE", "http://192.168.3.236:11434/v1"),
            model=os.getenv("LLM_MODEL", "qwen2.5:32b"),
            provider=os.getenv("LLM_PROVIDER", "custom"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0")),
        )


@dataclass
class ParserConfig:
    """Parser configuration settings"""

    type: str
    url: str

    @classmethod
    def from_env(cls):
        type = os.getenv("PARSER_TYPE", "marker")
        url = os.getenv("PARSER_URL", "http://192.168.3.236:8001")

        return cls(type=type, url=url)


@dataclass
class RetrievalConfig:
    """Retrieval configuration settings"""

    enable_query_preprocess: bool
    num_queries: int
    fuse_top_k: int
    dense_top_k: int
    sparse_top_k: int
    repacking_strategy: str

    @classmethod
    def from_env(cls):
        enable_query_preprocess = (
            os.getenv("ENABLE_QUERY_PREPROCESS", "false").lower() == "true"
        )

        return cls(
            enable_query_preprocess=enable_query_preprocess,
            num_queries=int(os.getenv("NUM_QUERIES", "3")),
            fuse_top_k=int(os.getenv("FUSE_TOP_K", "5")),
            dense_top_k=int(os.getenv("DENSE_TOP_K", "5")),
            sparse_top_k=int(os.getenv("SPARSE_TOP_K", "5")),
            repacking_strategy=os.getenv("REPACK_STRATEGY", "forward"),
        )


@dataclass
class VectorDBConfig:
    """Vector database configuration settings"""

    provider: str
    url: str
    api_key: Optional[str] = None

    @classmethod
    def from_env(cls) -> "VectorDBConfig":
        return cls(
            api_key=os.getenv("VECTOR_DB_API_KEY"),
            provider=os.getenv("VECTOR_DB_PROVIDER", "qdrant"),
            url=os.getenv("VECTOR_DB_URL", "http://localhost:6333/"),
        )


@dataclass
class Config:
    """Main configuration container"""

    app: AppConfig
    embed_model: EmbedModelConfig
    langfuse: LangfuseConfig
    llm: LLMConfig
    parser: ParserConfig
    retrieval: RetrievalConfig
    vector_db: VectorDBConfig

    @classmethod
    def from_env(cls) -> "Config":
        """Create configuration from environment variables"""
        return cls(
            app=AppConfig.from_env(),
            embed_model=EmbedModelConfig.from_env(),
            langfuse=LangfuseConfig.from_env(),
            llm=LLMConfig.from_env(),
            parser=ParserConfig.from_env(),
            retrieval=RetrievalConfig.from_env(),
            vector_db=VectorDBConfig.from_env(),
        )
