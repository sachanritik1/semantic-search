import pytest

from app.services.compare_llm_service import (
    _build_compare_prompt,
    _parse_llm_comparison,
)


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
