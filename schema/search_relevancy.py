from pydantic import BaseModel


class SearchRelevancyRequest(BaseModel):
    query: str
    collections: list[str] = []
    metadata: dict = {}
