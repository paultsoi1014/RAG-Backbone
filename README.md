# Retrieved-Augmented Generation (RAG) Backbone

A highly scalable and modular Retrieval-Augmented Generation (RAG) backbone utilizing Large Language Models (LLMs) and vector databases for various downstream applications, including document question-answering, document search, and relevancy scoring.

## 🧠 RAG Algorithm Architecture

The underlying RAG pipeline consists of three main stages: Ingestion, Retrieval (including Reranking & Repacking), and Downstream Applications.

```mermaid
graph TD
    %% 1. LEFT SIDE: Ingestion Flow
    subgraph Ingestion [1. Ingestion Pipeline]
        Docs[Documents] --> Parse[Parser]
        
        %% Ordered to place Agentic closest to the middle for cleaner routing
        Parse --> C_Rec[Recursive Chunking]
        Parse --> C_Sem[Semantic Chunking]
        Parse --> C_Agt[Agentic Chunking]
        
        C_Rec --> Embed[Embedding]
        C_Sem --> Embed
        C_Agt --> Embed
    end

    %% 3. RIGHT SIDE: Retrieval Flow
    subgraph Retrieval [3. Retrieval Pipeline]
        Query[User Query] --> QEmbed[Query Preprocessing & Embedding]
        
        %% Ordered to place Hybrid closest to the middle for cleaner routing to VDB
        QEmbed --> S_Hybrid[Hybrid Search]
        QEmbed --> S_Sparse[Sparse Search]
        QEmbed --> S_Dense[Dense Search]
        
        Rerank[Reranking] --> Repack[Repacking]
        Repack --> Context[Processed Context]
    end

    %% 2. STORAGE
    subgraph Storage [2. Vector Database]
        VDB[(Vector DB)]
    end

    %% 4. DOWNSTREAM
    subgraph Application [4. Downstream Applications]
        RetrieveOut[Document Retrieval]
        Relevancy[Search Document Relevancy]
        
        RetrieveOut --> Nodes[Retrieved Nodes]
        Relevancy --> Score[Relevancy Score]
    end

    %% --- Cross-Subgraph Links ---
    
    %% Connect Ingestion to Vector DB (1 rank difference)
    Embed -->|Ingest| VDB
    
    %% Connect Retrieval queries to Vector DB (2 rank difference)
    %% Using ---> adds an extra invisible level, perfectly aligning the top pipelines!
    S_Hybrid --->|Query| VDB
    S_Sparse --->|Query| VDB
    S_Dense --->|Query| VDB
    
    %% Connect Vector DB results back to the second half of Retrieval
    VDB -->|Top-K| Rerank
    
    %% Connect Retrieval context to Downstream Applications
    %% FIXED: Changed '-->' to '--->'. This adds vertical padding so the layout 
    %% engine doesn't have to push the Application box rightward to route the dotted line.
    Context ---> RetrieveOut
    Context ---> Relevancy
    
    %% Direct relation for scoring
    Query -.->|Input| Relevancy
```

## ⚙️ System Architecture

The overall system architecture demonstrates how external clients and AI agents interact with the RAG Backbone via APIs and the Model Context Protocol (MCP).

```mermaid
graph TD
    %% External Entities
    Client([Client Application])
    AIAgent([AI Agent / LLM])

    %% Interfaces
    subgraph Interfaces [Interfaces]
        FastAPI[FastAPI Server<br>api_server.py]
        FastMCP[FastMCP Server<br>mcp_server.py]
    end

    Client -->|HTTP REST| FastAPI
    AIAgent -->|MCP Protocol| FastMCP

    %% Controllers
    subgraph Controllers [Controllers]
        IngestEndpoint[POST /ingest]
        RetrieveEndpoint[POST /retrieve]
        SearchEndpoint[POST /search_relevancy]
    end

    FastAPI --> IngestEndpoint
    FastAPI --> RetrieveEndpoint
    FastAPI --> SearchEndpoint
    FastMCP -->|Tool: retrieve| RetrieveEndpoint

    %% Core Agent
    subgraph Agent [RAG Agent Core]
        Ingestion[Ingestion Engine<br>Recursive / Semantic / Agentic]
        Retrieval[Retrieval Engine<br>Dense / Sparse / Hybrid]
        Relevancy[Relevancy Engine<br>Scoring]
    end

    IngestEndpoint --> Ingestion
    RetrieveEndpoint --> Retrieval
    SearchEndpoint --> Relevancy

    %% Vector DB
    subgraph DB [Vector Database]
        Qdrant[(Qdrant)]
        Elastic[(Elasticsearch)]
    end

    Ingestion -->|Index Embeddings| DB
    Retrieval <-->|Query & Fetch Nodes| DB
    Relevancy <-->|Fetch Context| DB
```
## 🏗 Architecture & Features

The RAG system is built with a modular architecture and supports both standard REST APIs and Model Context Protocol (MCP).

### 🧩 Core Components
- **Agent (`agent/`)**: Contains the core orchestration logic for the RAG workflows.
  - **Ingestion**: Parses documents (like PDFs and TXT files) and chunks them. Supports multiple advanced chunking strategies: `recursive`, `semantic`, and `agentic`.
  - **Retrieval**: Fetches relevant information nodes and documents leveraging dense, sparse, or hybrid retrieval methods. Supports multi-collection querying and metadata filtering.
  - **Search Relevancy**: Evaluates and scores document relevancy based on the user's query.
- **Controllers (`controller/`)**: Exposes the system capabilities via RESTful FastAPI endpoints (`/ingest`, `/retrieve`, `/search_relevancy`).
- **Vector Database (`vector_db/`)**: Abstracted vector storage layer currently supporting **Qdrant** and **Elasticsearch**.
- **Interfaces**:
  - **FastAPI Server** (`api_server.py`): Standard REST API for broad application integration.
  - **FastMCP Server** (`mcp_server.py`): Model Context Protocol server, enabling native integration with AI agents like Claude Desktop.

## 🚀 Installation

### 1. Clone the repository
```bash
git clone https://git.capax-tech.com/paul_tsoi/rag_backbone.git
cd rag_backbone
```

### 2. Set up a virtual environment
Using `venv` (Python standard library):
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```
Using `conda` (Recommended Python 3.11):
```bash
conda create --name rag_env python=3.11
conda activate rag_env
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configuration
Copy the environment variables example and configure it according to your LLM and Vector DB setup:
```bash
cp .env.example .env
```

## 🐳 Quick Start (Docker)

The easiest way to get started with the Qdrant vector store and the RAG servers is via Docker Compose:

```bash
docker compose up -d --build
```
This command spins up:
- **Qdrant Vector Database** on port `6333`
- **FastAPI REST Server** on port `8008`
- **FastMCP Server** on port `8009`

## 💻 Usage

### 1. Document Ingestion (Python SDK)
You can directly use the RAGAgent within your Python codebase:

```python
from agent.core import RAGAgent

if __name__ == "__main__":
    rag_agent = RAGAgent()
    rag_agent.folder_ingestion(
        folder_path="data/superlinear", # Path to your data directory
        collection_name="super_linear", # Name of the target collection
        mode=["agentic", "semantic", "recursive"], # Chunking modes
    )
```

### 2. Document Ingestion (REST API)
Using the FastAPI server, you can ingest files via HTTP POST requests:

```python
import requests
import json

BASE_URL = "http://127.0.0.1:8008"
api_url = f"{BASE_URL}/ingest"

file_paths = ["file_1.txt", "file_2.pdf"]
files = [("files", open(path, "rb")) for path in file_paths]

metadata = {
    "collection_name": "super_linear_2",
    "mode": ["recursive", "semantic", "agentic"],
}

response = requests.post(
    api_url, 
    files=files, 
    data={"metadata": json.dumps(metadata)}
)
print("Response:", response.json())
```

### 3. Document Retrieval & Search (REST API)
The FastAPI server exposes retrieval endpoints for querying your knowledge base:
- `POST /retrieve`: Retrieves context nodes and documents based on the query, collections, and metadata.
- `POST /search`: Calculates and returns document relevancy scores based on the query.

### 4. Agentic Integration (MCP)
The RAG backbone includes an MCP server (`mcp_server.py`) which exposes a `retrieve` tool. You can integrate this with MCP-compatible clients (like Claude Desktop) to allow your AI assistants to dynamically query your indexed documents.
