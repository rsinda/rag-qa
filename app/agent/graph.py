"""LangGraph agent."""

from __future__ import annotations
import json, logging, time
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI


from app.config import (
    LLM_MODEL_NAME,
    LLM_API_KEY,
    RETRIEVAL_TOP_K_PRIMARY,
    RETRIEVAL_TOP_K_FALLBACK,
)
from app.vector_store import ContentVectorStore
from app.agent.prompt import SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    question: str
    country: str
    language: str
    hits: List[Dict[str, Any]]
    answer: str
    citations: List[Dict[str, Any]]
    language_used: str
    retrieval_count: int
    latency_ms: int
    model: str
    error: Optional[str]
    _start_ms: float


def _make_llm():
    return ChatGoogleGenerativeAI(
        model=LLM_MODEL_NAME,
        max_retries=2,
        google_api_key=LLM_API_KEY,
    )


def retrieve_node(state: AgentState, store: ContentVectorStore) -> AgentState:
    hits = store.query(
        query_text=state["question"],
        country=state["country"],
        language=state["language"],
        top_k=RETRIEVAL_TOP_K_PRIMARY,
    )
    return {**state, "hits": hits, "retrieval_count": len(hits)}


def check_node(state: AgentState) -> AgentState:
    if not state["hits"]:
        return {
            **state,
            "error": (
                f"No content found for country={state['country']}, "
                f"language={state['language']}."
            ),
        }
    return {**state, "error": None}


def synthesize_node(state: AgentState, llm) -> AgentState:
    if state.get("error"):
        return {
            **state,
            "answer": state["error"],
            "citations": [],
            "language_used": state["language"],
            "model": LLM_MODEL_NAME,
        }

    context = "\n\n".join(
        f"[{h['content_id']}] ({h['type']})\n{h['body']}" for h in state["hits"]
    )
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Question: {state['question']}\n\nContent:\n{context}"),
    ]
    try:
        raw = llm.invoke(messages).content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1].lstrip("json").strip()
        parsed = json.loads(raw)
        answer = parsed.get("answer", "")
        used_ids = set(parsed.get("used_content_ids", []))
    except Exception as exc:
        logger.warning("LLM synthesis failed: %s", exc)
        answer, used_ids = "Could not generate an answer at this time.", set()

    citations = [
        {
            "content_id": h["content_id"],
            "type": h["type"],
            "excerpt": h["body"][:300],
            "match_score": h["match_score"],
        }
        for h in state["hits"]
        if h["content_id"] in used_ids
    ]

    if not citations and state["hits"]:
        logger.info("fallback: citing top-2 since LLM returned no IDs")
        citations = [
            {
                "content_id": h["content_id"],
                "type": h["type"],
                "excerpt": h["body"][:300],
                "match_score": h["match_score"],
            }
            for h in state["hits"][:2]
        ]
    return {
        **state,
        "answer": answer,
        "citations": citations,
        "language_used": state["language"],
        "model": LLM_MODEL_NAME,
    }


def format_node(state: AgentState) -> AgentState:
    elapsed = int((time.time() * 1000) - state["_start_ms"])
    return {**state, "latency_ms": elapsed}


# Graph builder
_graph_cache: Dict[str, Any] = {}


def build_graph(store: ContentVectorStore):
    llm = _make_llm()
    g = StateGraph(AgentState)
    g.add_node("retrieve", lambda s: retrieve_node(s, store))
    g.add_node("check", check_node)
    g.add_node("synthesize", lambda s: synthesize_node(s, llm))
    g.add_node("format", format_node)
    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "check")
    g.add_edge("check", "synthesize")
    g.add_edge("synthesize", "format")
    g.add_edge("format", END)

    # Save graph for visualization
    graph_image = g.compile().get_graph().draw_mermaid_png()
    with open("screenshots/03_langgraph_graph.png", "wb") as f:
        f.write(graph_image)
    return g.compile()


def run_agent(question, country, language, store) -> AgentState:
    if "default" not in _graph_cache:
        _graph_cache["default"] = build_graph(store)
    return _graph_cache["default"].invoke(
        {
            "question": question,
            "country": country,
            "language": language,
            "hits": [],
            "answer": "",
            "citations": [],
            "language_used": language,
            "retrieval_count": 0,
            "latency_ms": 0,
            "model": LLM_MODEL_NAME,
            "error": None,
            "_start_ms": time.time() * 1000,
        }
    )
