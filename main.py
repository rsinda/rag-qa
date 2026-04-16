import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from app.models import AskRequest, AskResponse, Citation, TraceInfo
from app.vector_store import ContentVectorStore
from app.agent import run_agent

logging.basicConfig(level=logging.INFO)
_store: ContentVectorStore | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _store
    _store = ContentVectorStore()
    yield


app = FastAPI(
    title="Multi-Country Content Q&A — Self-RAG", version="2.0.0", lifespan=lifespan
)


@app.get("/health")
def health():
    return {"status": "ok", "collection_count": _store.count() if _store else None}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    if _store is None:
        raise HTTPException(status_code=503, detail="Vector store not initialised")
    state = run_agent(req.question, req.country.upper(), req.language, _store)
    return AskResponse(
        answer=state["answer"],
        language_used=state["language_used"],
        citations=[Citation(**c) for c in state["citations"]],
        trace=TraceInfo(
            retrieval_count=state["retrieval_count"],
            graded_relevant=len(state["relevant_hits"]),
            fallback_used=state["fallback_used"],
            latency_ms=state["latency_ms"],
            model=state["model"],
        ),
    )
