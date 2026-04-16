from typing import List, Optional
from pydantic import BaseModel


class AskRequest(BaseModel):
    question: str
    country: str
    language: str


class Citation(BaseModel):
    content_id: str
    type: str
    excerpt: str
    match_score: float


class TraceInfo(BaseModel):
    retrieval_count: int
    latency_ms: int
    model: str


class AskResponse(BaseModel):
    answer: str
    language_used: str
    citations: List[Citation]
    trace: TraceInfo
