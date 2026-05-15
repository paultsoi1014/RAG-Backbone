from pydantic import BaseModel
from typing import List


class IngestionRequest(BaseModel):
    collection_name: str
    mode: List[str] = ["recursive"]
