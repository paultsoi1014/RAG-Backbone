import logging
from fastapi import APIRouter, HTTPException

from agent.core import RAGAgent
from schema.retrieve import RetrieveRequest

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/retrieve", tags=["retrieve"])

rag_agent = RAGAgent()


@router.post("")
async def retrieve(request: RetrieveRequest):
    """
    Retrieve relevant information nodes and documents based on query.

    Parameters
    ----------
    request: RetrieveRequest
        The retrieval request containing the following attributes:
        - query (str): The search query string
        - collections (list, optional): List of collection names to retrieve from
        - recursive_only (bool, optional): Flag to restrict the search to
          recursive retrieval only
        - metadata (dict, optional): Metadata filters to apply to the retrieval

    Returns
    -------
    dict
        A dictionary containing the retrieval response:
        {
            "nodes": [list of relevant information nodes/chunks],
            "documents": [list of complete documents]
        }

    Raises
    ------
    HTTPException
        Status 500 if any error occurs during the retrieval process.
    """
    try:
        search_kwargs = {"query": request.query}

        if request.collections:
            search_kwargs["collections"] = request.collections

        if request.recursive_only:
            search_kwargs["recursive_only"] = request.recursive_only

        if request.metadata:
            search_kwargs["metadata"] = request.metadata

        retrieve_response = await rag_agent.aretrieve_information(**search_kwargs)

        return {
            "nodes": retrieve_response["nodes"],
            "documents": retrieve_response["documents"],
        }

    except Exception as e:
        logger.error(
            f"Retrieval error: {str(e)}"
        )  # Added optional logging here for better debugging
        raise HTTPException(status_code=500, detail=str(e))
