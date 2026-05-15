from pydantic import BaseModel


class RetrieveRequest(BaseModel):
    query: str
    collections: list[str] = []
    recursive_only: bool = False
    metadata: dict = {}
