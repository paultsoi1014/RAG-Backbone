import logging

from fastmcp import FastMCP
from typing import Any, Dict, List, Optional, Union

from agent.core import RAGAgent
from config import Config

logger = logging.getLogger(__name__)

# Initialize FastMCP server
cfg = Config.from_env()
mcp = FastMCP(name="RAG Retrieval Server")

# Initialize your RAG agent
rag_agent = RAGAgent()


@mcp.tool(
    description="Retrieve relevant information nodes and documents based on a search query. Supports collection and metadata filtering. Use when addition information is required",
    tags={"retrieve", "rag"},
)
async def retrieve(
    query: Union[str, List[str]],
    collections: Optional[List[str]] = None,
    recursive_only: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
) -> dict:
    """
    Retrieve relevant information nodes and documents based on a search query

    Parameters
    ----------
    query: Union[str, List[str]]
        The search query string
    collections: Optional[List[str]]
        Optional list of collection names to retrieve from. Default is None
    recursive_only: bool
        Flag to restrict the search to recursive retrieval only. Default is
        False
    metadata: Optional[Dict[str, Any]]
        Optional metadata filters to apply to the retrieval. Default is None
    """
    try:
        # Construct arguments dynamically just like in your API route
        search_kwargs = {"query": query}

        if collections:
            search_kwargs["collections"] = collections

        if recursive_only:
            search_kwargs["recursive_only"] = recursive_only

        if metadata:
            search_kwargs["metadata"] = metadata

        # Execute the retrieval
        response = await rag_agent.aretrieve_information(**search_kwargs)

        return {
            "information": response.get("documents", ""),
            "actions_follows": "Answer the user's question based on the information. Do not call the tools subsequently",
        }

    except Exception as e:
        logger.error(f"Retrieval error: {str(e)}")
        raise RuntimeError(f"Retrieval failed: {str(e)}")


if __name__ == "__main__":
    mcp.run(
        transport="http", host="0.0.0.0", port=cfg.app.mcp_port, stateless_http=True
    )
