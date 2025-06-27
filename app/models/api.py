from pydantic import BaseModel
from typing import List, Optional, Dict
from app.models.embeddings import Document


class SearchRequest(BaseModel):
    query: str
    limit: Optional[int] = 5
    filters: Optional[Dict[str, str]] = None


class SearchResponse(BaseModel):
    results: List[Document]


class LLMRequest(BaseModel):
    query: str
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_output_tokens: Optional[int] = None
    limit: Optional[int] = 5
    filters: Optional[Dict[str, str]] = None


class LLMResponse(BaseModel):
    answer: str
    source_documents: List[Document]
