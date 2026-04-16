"""LangGraph agent."""

from __future__ import annotations
import json, logging, time, concurrent.futures
from typing import Any, Dict, List, Literal, Optional, TypedDict


from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables.config import ContextThreadPoolExecutor


from app.config import (
    LLM_MODEL_NAME,
    LLM_API_KEY,
)
from app.vector_store import ContentVectorStore
from app.agent.prompt import (
    SUFFICIENCY_PROMPT,
    GRADE_DOCS_PROMPT,
    SYNTHESIZE_PROMPT,
    GRADE_ANSWER_PROMPT,
    TRANSLATE__PROMPT,
)

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    # Inputs
    question: str
    country: str
    language: str
    # Retrieval
    # all weighted hits of retrived context
    hits: List[Dict[str, Any]]
    # same-language hits of retrived context
    primary_hits: List[Dict[str, Any]]
    # other-language hits of retrived context
    fallback_hits: List[Dict[str, Any]]
    # graded-relevant hits (pre-translation) of retrived context
    relevant_hits: List[Dict[str, Any]]
    # after translation hots (ready for synthesis) of retrived context
    translated_hits: List[Dict[str, Any]]
    retrieval_count: int
    fallback_used: bool
    # same language document grading decision
    primary_grade_decision: Optional[
        Literal["sufficient", "insufficient", "no_primary_docs"]
    ]
    # Generation
    answer: str
    citations: List[Dict[str, Any]]
    language_used: str
    # Answer grading
    answer_grade: Optional[Literal["useful", "not_grounded", "not_useful"]]
    answer_retry_count: int
    # Trace
    latency_ms: int
    model: str
    _start_ms: float


def _make_llm(temperature):
    return ChatGoogleGenerativeAI(
        model=LLM_MODEL_NAME,
        max_retries=3,
        temperature=temperature,
        google_api_key=LLM_API_KEY,
    )


def _grade_single_hit(
    hit: Dict[str, Any], question: str, llm
) -> Optional[Dict[str, Any]]:
    prompt = (
        f"Question: {question}\n\n"
        f"Content [{hit['content_id']}] (lang={hit['language']}):\n"
        f"Title: {hit['title']}\n\nBody:{hit['body']}"
    )
    try:
        raw = llm.invoke(
            [
                SystemMessage(content=GRADE_DOCS_PROMPT),
                HumanMessage(content=prompt),
            ]
        ).content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1].lstrip("json").strip()
        grade = json.loads(raw).get("grade", "irrelevant")
    except Exception as exc:
        logger.warning("Grade failed for %s: %s", hit["content_id"], exc)
        grade = "irrelevant"

    if grade in ("relevant", "partial"):
        return {**hit, "grade": grade}
    else:
        logger.warning(f"No relevant or partial hit for {question}.")
        return None


def _grade_hits(
    hits: List[Dict[str, Any]],
    question: str,
    llm,
) -> List[Dict[str, Any]]:
    """Grades a list of Contents. Returns only relevant/partial ones with grade attached."""
    kept = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(_grade_single_hit, hit, question, llm) for hit in hits
        ]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                kept.append(result)

    return kept


def retrieve_weighted_node(state: AgentState, store: ContentVectorStore) -> AgentState:
    query = state.get("rewritten_query") or state["question"]
    hits = store.query(query, state["country"], state["language"])
    primary_hits = [h for h in hits if not h["is_fallback"]]
    fallback_hits = [h for h in hits if h["is_fallback"]]
    fallback_used = len(fallback_hits) > 0

    logger.info(
        "Retrieved: %d primary (lang=%s) + %d fallback (other langs) for country=%s",
        len(primary_hits),
        state["language"],
        len(fallback_hits),
        state["country"],
    )
    return {
        **state,
        "hits": hits,
        "primary_hits": primary_hits,
        "fallback_hits": fallback_hits,
        "retrieval_count": len(hits),
        "fallback_used": fallback_used,
    }


def grade_primary_docs_node(state: AgentState, llm) -> AgentState:
    question = state.get("rewritten_query") or state["question"]

    # if no primary docs found at all
    if not state["primary_hits"]:
        logger.info("No primary-language docs found → routing to fallback grading")
        return {
            **state,
            "relevant_hits": [],
            "primary_grade_decision": "no_primary_docs",
        }

    # Eval each primary hit document
    relevant_primary = _grade_hits(state["primary_hits"], question, llm)
    logger.info(
        "Primary grading: %d/%d kept", len(relevant_primary), len(state["primary_hits"])
    )

    if not relevant_primary:
        return {
            **state,
            "relevant_hits": [],
            "primary_grade_decision": "insufficient",
        }

    # sufficiency check across all kept primary docs
    context = "\n\n".join(
        f"[{h['content_id']}] grade={h['grade']}\n Title: {h['title']}\n\n Body: {h['body']}"
        for h in relevant_primary
    )
    prompt = f"Question: {question}\n\nRelevant documents found:\n{context}"
    try:
        raw = llm.invoke(
            [
                SystemMessage(content=SUFFICIENCY_PROMPT),
                HumanMessage(content=prompt),
            ]
        ).content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1].lstrip("json").strip()
        decision = json.loads(raw).get("decision", "sufficient")
    except Exception as exc:
        logger.warning("Sufficiency check failed: %s — defaulting to sufficient", exc)
        decision = "sufficient"

    logger.info("Primary sufficiency decision: %s", decision)
    return {
        **state,
        "relevant_hits": relevant_primary,
        "primary_grade_decision": decision,
    }


def decide_after_primary_grading(state: AgentState) -> str:
    decision = state.get("primary_grade_decision")
    if decision == "sufficient":
        # if Primary docs are good then skip translation entirely
        return "synthesize"
    # if "insufficient" or "no_primary_docs" then try fallback language docs
    return "grade_fallback_docs"


def grade_fallback_docs_node(state: AgentState, llm) -> AgentState:
    question = state.get("rewritten_query") or state["question"]

    if not state["fallback_hits"]:
        logger.info("No fallback docs available")
        # relevant_hits stays as-it is
        return {**state}

    relevant_fallback = _grade_hits(state["fallback_hits"], question, llm)
    logger.info(
        "Fallback grading: %d/%d kept",
        len(relevant_fallback),
        len(state["fallback_hits"]),
    )

    # Merge: keep existing relevant primaries + add relevant fallbacks
    existing_primary = state.get("relevant_hits", [])
    merged = existing_primary + relevant_fallback

    # Re-sort by weighted match_score
    merged.sort(key=lambda x: x["match_score"], reverse=True)

    return {**state, "relevant_hits": merged}


def decide_after_fallback_grading(state: AgentState) -> str:
    if state["relevant_hits"]:
        # Check if any fallback (non-target-language) docs need translation
        needs_translation = any(h.get("is_fallback") for h in state["relevant_hits"])
        return "translate_and_fill" if needs_translation else "synthesize"
    # Nothing relevant found at all
    return "empty_response"


def translate_and_fill_node(state: AgentState, llm) -> AgentState:
    target_lang = state["language"]

    # Translate only non-English fallback docs to English
    # We will translate the answer to required target language during synthesis.
    to_translate = [
        h
        for h in state["relevant_hits"]
        if h.get("is_fallback") and h.get("language") != "en"
    ]
    pass_through = [
        h
        for h in state["relevant_hits"]
        if not (h.get("is_fallback") and h.get("language") != "en")
    ]

    def _translate_one(hit: Dict[str, Any]) -> Dict[str, Any]:
        prompt = f"Target language: english\n\n Content to translate: {hit['excerpt']}"
        try:
            raw = llm.invoke(
                [
                    SystemMessage(content=TRANSLATE__PROMPT),
                    HumanMessage(content=prompt),
                ]
            ).content.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1].lstrip("json").strip()
            translated_text = json.loads(raw).get("translated", hit["excerpt"])
            logger.info(
                "Translated %s (%s → en)",
                hit["content_id"],
                hit["language"],
            )
        except Exception as exc:
            logger.warning("Translation failed for %s: %s", hit["content_id"], exc)
            translated_text = hit["excerpt"]

        return {
            **hit,
            "excerpt": translated_text,
            "original_language": hit["language"],
            "language": "en",
        }

    translated = []
    if to_translate:
        max_workers = min(len(to_translate), 8)
        with ContextThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_translate_one, hit): hit for hit in to_translate}
            for future in concurrent.futures.as_completed(futures):
                try:
                    translated.append(future.result())
                except Exception as exc:
                    original_hit = futures[future]
                    logger.error(
                        "Unhandled translation error for %s: %s",
                        original_hit["content_id"],
                        exc,
                    )
                    translated.append(
                        {
                            **original_hit,
                            "excerpt": original_hit["body"],
                            "original_language": original_hit["language"],
                        }
                    )

    passed = [
        {**h, "original_language": h.get("language", target_lang)} for h in pass_through
    ]

    translated_hits = translated + passed
    translated_hits.sort(key=lambda x: x["match_score"], reverse=True)

    return {**state, "translated_hits": translated_hits}


def empty_response_node(state: AgentState) -> AgentState:
    return {
        **state,
        "answer": (
            f"I could not find any relevant information for your question "
            f"in country={state['country']} (language={state['language']}). "
            f"Please contact support for assistance."
        ),
        "citations": [],
        "language_used": state["language"],
        "answer_grade": "not_useful",
        "translated_hits": [],
    }


def synthesize_node(state: AgentState, llm) -> AgentState:
    # Use translated hits if the translate node ran, else use relevant hits directly
    docs = state.get("translated_hits") or state.get("relevant_hits", [])

    context = "\n\n".join(
        f"[{h['content_id']}] ({h['type']}, grade={h.get('grade','relevant')})\n{h['excerpt']}"
        for h in docs
    )
    prompt = (
        f"Answer language: {state['language']}\n"
        f"Question: {state.get('rewritten_query') or state['question']}\n\n"
        f"Content:\n{context}"
    )
    try:
        raw = llm.invoke(
            [
                SystemMessage(content=SYNTHESIZE_PROMPT),
                HumanMessage(content=prompt),
            ]
        ).content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1].lstrip("json").strip()
        parsed = json.loads(raw)
        answer = parsed.get("answer", "")
        used_ids = set(parsed.get("used_content_ids", []))
    except Exception as exc:
        logger.warning("Synthesis failed: %s", exc)
        answer, used_ids = "Could not generate an answer at this time.", set()

    citations = [
        {
            "content_id": h["content_id"],
            "type": h["type"],
            "excerpt": h["excerpt"],
            "body": h.get("body"),
            "match_score": h["match_score"],
            "original_language": h.get(
                "original_language", h.get("language", state["language"])
            ),
        }
        for h in docs
        if h["content_id"] in used_ids
    ]
    if not citations and docs:
        citations = [
            {
                "content_id": h["content_id"],
                "type": h["type"],
                "excerpt": h["excerpt"],
                "match_score": h["match_score"],
                "original_language": h.get("original_language", state["language"]),
                "body": h.get("body"),
            }
            for h in docs[:2]
        ]

    return {
        **state,
        "answer": answer,
        "citations": sorted(citations, key=lambda x: x["match_score"], reverse=True),
        "language_used": state["language"],
        "model": LLM_MODEL_NAME,
    }


def grade_answer_node(state: AgentState, llm) -> AgentState:
    if not state.get("citations"):
        return {**state, "answer_grade": "not_useful"}

    context = "\n".join(
        f"[{c['content_id']}]: {c['excerpt']}" for c in state["citations"]
    )
    prompt = (
        f"Question: {state['question']}\n\n"
        f"Source excerpts:\n{context}\n\n"
        f"Generated answer:\n{state['answer']}"
    )
    try:
        raw = llm.invoke(
            [
                SystemMessage(content=GRADE_ANSWER_PROMPT),
                HumanMessage(content=prompt),
            ]
        ).content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1].lstrip("json").strip()
        grade = json.loads(raw).get("grade", "useful")
    except Exception as exc:
        logger.warning("Answer grading failed: %s", exc)
        # assuming OK on parse failure
        grade = "useful"

    logger.info("Answer grade: %s", grade)
    state["citations"] = [{**c, "excerpt": c["body"][:250]} for c in state["citations"]]
    return {**state, "answer_grade": grade}


def decide_after_answer_grading(state: AgentState) -> str:
    grade = state.get("answer_grade", "useful")
    if grade == "useful":
        return "format"
    if grade == "not_grounded" and state.get("answer_retry_count", 0) < 1:
        return "synthesize"  # retry synthesis once
    return "format"  # give up gracefully


def format_node(state: AgentState) -> AgentState:
    elapsed = int((time.time() * 1000) - state["_start_ms"])
    answer_retry_count = state.get("answer_retry_count", 0)
    if state.get("answer_grade") == "not_grounded":
        answer_retry_count += 1
    return {**state, "latency_ms": elapsed, "answer_retry_count": answer_retry_count}


# Graph builder
_graph_cache: Dict[str, Any] = {}


def build_graph(store: ContentVectorStore):
    llm_precise = _make_llm(temperature=0.0)
    llm_creative = _make_llm(temperature=0.3)

    g = StateGraph(AgentState)

    g.add_node("retrieve_weighted", lambda s: retrieve_weighted_node(s, store))
    g.add_node("grade_primary_docs", lambda s: grade_primary_docs_node(s, llm_precise))
    g.add_node(
        "grade_fallback_docs", lambda s: grade_fallback_docs_node(s, llm_precise)
    )
    g.add_node("translate_and_fill", lambda s: translate_and_fill_node(s, llm_precise))
    g.add_node("empty_response", empty_response_node)
    g.add_node("synthesize", lambda s: synthesize_node(s, llm_creative))
    g.add_node("grade_answer", lambda s: grade_answer_node(s, llm_precise))
    g.add_node("format", format_node)

    g.set_entry_point("retrieve_weighted")
    g.add_edge("retrieve_weighted", "grade_primary_docs")

    # After primary grading: sufficient → synthesize, else → grade fallback
    g.add_conditional_edges(
        "grade_primary_docs",
        decide_after_primary_grading,
        {
            "synthesize": "synthesize",
            "grade_fallback_docs": "grade_fallback_docs",
        },
    )

    # After fallback grading: has useful → translate if needed, else rewrite or give up
    g.add_conditional_edges(
        "grade_fallback_docs",
        decide_after_fallback_grading,
        {
            "translate_and_fill": "translate_and_fill",
            "synthesize": "synthesize",
            "empty_response": "empty_response",
        },
    )

    g.add_edge("translate_and_fill", "synthesize")
    g.add_edge("empty_response", "format")
    g.add_edge("synthesize", "grade_answer")

    g.add_conditional_edges(
        "grade_answer",
        decide_after_answer_grading,
        {
            "format": "format",
            "synthesize": "synthesize",
        },
    )

    g.add_edge("format", END)

    # # Save graph for visualization
    # graph_image = g.compile().get_graph().draw_mermaid_png()
    # with open("screenshots/03_langgraph_graph.png", "wb") as f:
    #     f.write(graph_image)
    return g.compile()


def run_agent(
    question: str, country: str, language: str, store: ContentVectorStore
) -> AgentState:
    if "default" not in _graph_cache:
        _graph_cache["default"] = build_graph(store)
    return _graph_cache["default"].invoke(
        {
            "question": question,
            "country": country,
            "language": language,
            "hits": [],
            "primary_hits": [],
            "fallback_hits": [],
            "relevant_hits": [],
            "translated_hits": [],
            "retrieval_count": 0,
            "fallback_used": False,
            "retry_count": 0,
            "rewrites": 0,
            "rewritten_query": None,
            "primary_grade_decision": None,
            "answer": "",
            "citations": [],
            "language_used": language,
            "answer_grade": None,
            "answer_retry_count": 0,
            "latency_ms": 0,
            "model": LLM_MODEL_NAME,
            "_start_ms": time.time() * 1000,
        }
    )
