import os
import logging

from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logging.info(f"ENVIRONMENT: {os.getenv('ENVIRONMENT', 'development')}")

# Load environment variables from .env file
if os.getenv("ENVIRONMENT", "development") != "development":
    load_dotenv(override=True)

# Qdrant Vector DB Configuration
VECTOR_DB_PROVIDER = os.getenv("VECTOR_DB_PROVIDER", "qdrant")
VECTOR_DB_URL = os.getenv("VECTOR_DB_URL", "http://localhost:6333/")

# Embedding Model Configuration
EMBED_MODEL_PROVIDER = os.getenv("EMBED_MODEL_PROVIDER", "ollama")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "bge-m3")
EMBED_MODEL_BASE = os.getenv("EMBED_MODEL_BASE", "http://localhost:11434")
EMBED_DIMENSION = int(os.getenv("EMBED_DIMENSION", "1024"))

# LLM Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "custom")
LLM_API_KEY = os.getenv("LLM_API_KEY", "EMPTY")
LLM_API_BASE = os.getenv("LLM_API_BASE", "http://localhost:11434/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:32b")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))

# LLM Trace
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "None")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "None")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "http://localhost:3000")

# Retrieval Configuration
SIMILARITY_TOP_K = int(os.getenv("SIMILARITY_TOP_K", "5"))
SPARSE_TOP_K = int(os.getenv("SPARSE_TOP_K", "5"))

# Parser Configuration
PARSER_URL = os.getenv("PARSER_URL", "http://localhost:8008")
