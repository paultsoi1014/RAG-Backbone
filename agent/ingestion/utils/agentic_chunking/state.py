from typing import List, TypedDict


class ChunkingState(TypedDict):
    """State schema for the agentic chunking workflow"""

    document: str
    propositions: List[str]
    chunks: List[str]
    chunk_metadata: List[str]
    final_chunks: List[str]
    base_metadata: dict
    error: str | None
