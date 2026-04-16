import pytest
from unittest.mock import patch, MagicMock

from app.agent.graph import (
    retrieve_weighted_node,
    decide_after_primary_grading,
    empty_response_node,
    decide_after_fallback_grading,
    decide_after_answer_grading,
    format_node,
    AgentState,
)


def create_base_state() -> AgentState:
    return {
        "question": "test question",
        "country": "US",
        "language": "en",
        "hits": [],
        "primary_hits": [],
        "fallback_hits": [],
        "relevant_hits": [],
        "translated_hits": [],
        "retrieval_count": 0,
        "fallback_used": False,
        "primary_grade_decision": None,
        "answer": "",
        "citations": [],
        "language_used": "",
        "answer_grade": None,
        "answer_retry_count": 0,
        "latency_ms": 0,
        "model": "test-model",
        "_start_ms": 0.0,
    }


def test_retrieve_weighted_node():
    store_mock = MagicMock()
    # Returnss 2 primary hits and 1 fallback hit
    store_mock.query.return_value = [
        {"content_id": "1", "is_fallback": False},
        {"content_id": "2", "is_fallback": False},
        {"content_id": "3", "is_fallback": True},
    ]

    state = create_base_state()
    result_state = retrieve_weighted_node(state, store_mock)

    assert result_state["retrieval_count"] == 3
    assert len(result_state["primary_hits"]) == 2
    assert len(result_state["fallback_hits"]) == 1
    assert result_state["fallback_used"] is True
    store_mock.query.assert_called_once_with("test question", "US", "en")


def test_decide_after_primary_grading():
    state = create_base_state()

    # assert decisions
    state["primary_grade_decision"] = "sufficient"
    assert decide_after_primary_grading(state) == "synthesize"

    state["primary_grade_decision"] = "insufficient"
    assert decide_after_primary_grading(state) == "grade_fallback_docs"

    state["primary_grade_decision"] = "no_primary_docs"
    assert decide_after_primary_grading(state) == "grade_fallback_docs"


def test_empty_response_node():
    state = create_base_state()
    state["country"] = "US"
    state["language"] = "en"

    result = empty_response_node(state)
    assert "could not find any relevant information" in result["answer"]
    assert result["citations"] == []
    assert result["answer_grade"] == "not_useful"


def test_decide_after_fallback_grading():
    state = create_base_state()

    # No relevant hits
    state["relevant_hits"] = []
    assert decide_after_fallback_grading(state) == "empty_response"

    # Relevant hits, 0 are fallback
    state["relevant_hits"] = [{"is_fallback": False}]
    assert decide_after_fallback_grading(state) == "synthesize"

    # Relevant hits, some are fallback
    state["relevant_hits"] = [{"is_fallback": True}]
    assert decide_after_fallback_grading(state) == "translate_and_fill"


def test_decide_after_answer_grading():
    state = create_base_state()

    state["answer_grade"] = "useful"
    assert decide_after_answer_grading(state) == "format"

    state["answer_grade"] = "not_grounded"
    state["answer_retry_count"] = 0
    assert decide_after_answer_grading(state) == "synthesize"

    state["answer_grade"] = "not_grounded"
    state["answer_retry_count"] = 1
    assert decide_after_answer_grading(state) == "format"


@patch("app.agent.graph.time.time")
def test_format_node(mock_time):
    mock_time.return_value = 1000.5
    state = create_base_state()
    state["_start_ms"] = 1000000.0  # 1000.0 sec in ms

    result = format_node(state)
    assert result["latency_ms"] == 500  # 1000500 - 1000000
