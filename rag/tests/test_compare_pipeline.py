import pytest

from app.pipelines.compare import (
    _COMPARE_SYSTEM_PROMPT,
    _build_compare_messages,
    _build_compare_prompt,
    _parse_llm_comparison,
)


def test_build_compare_messages_splits_static_and_dynamic():
    system_prompt, user_message = _build_compare_messages(
        "What is RAG?",
        dense=[{"index": 0, "content": "dense doc"}],
        sparse=[{"index": 0, "content": "sparse doc", "score": 1.5}],
    )
    assert system_prompt == _COMPARE_SYSTEM_PROMPT
    assert "dense-0" in user_message
    assert "sparse-0" in user_message
    assert "What is RAG?" in user_message
    assert _COMPARE_SYSTEM_PROMPT not in user_message


def test_build_compare_prompt_includes_both_sources():
    prompt = _build_compare_prompt(
        "What is RAG?",
        dense=[{"index": 0, "content": "dense doc"}],
        sparse=[{"index": 0, "content": "sparse doc", "score": 1.5}],
    )
    assert "dense-0" in prompt
    assert "sparse-0" in prompt
    assert "What is RAG?" in prompt
    assert "bm25_score" in prompt


def test_parse_llm_comparison_extracts_json_object():
    raw = 'Here is the result:\n{"document_scores": [], "summary": "ok"}'
    parsed = _parse_llm_comparison(raw)
    assert parsed["summary"] == "ok"


def test_parse_llm_comparison_rejects_invalid_payload():
    with pytest.raises(ValueError):
        _parse_llm_comparison("no json here")
