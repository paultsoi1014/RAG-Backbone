import logging
from fastapi import APIRouter, HTTPException

from agent.core import RAGAgent
from schema.search_relevancy import SearchRelevancyRequest

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/search_relevancy", tags=["search_relevancy"])

rag_agent = RAGAgent()


@router.post("")
async def search(request: SearchRelevancyRequest):
    """
    Search for relevant documents based on query and optional filters

    Parameters
    ----------
    request: SearchRelevancyRequest
        The search request containing:
        - query: The search query string
        - collections: Optional list of collection names to search within
        - metadata: Optional metadata filters to apply to the search

    Returns
    -------
    dict
        A dictionary containing the search response with relevant documents:
        {
            "response": [list of relevant documents with scores]
        }

    Raises
    ------
    HTTPException
        Status 500 if any error occurs during the search process
    """
    try:
        search_kwargs = {"query": request.query}

        if request.collections:
            search_kwargs["collections"] = request.collections

        if request.metadata:
            search_kwargs["metadata"] = request.metadata

        search_response = await rag_agent.asearch_document_relevancy(**search_kwargs)

        return {"response": search_response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
