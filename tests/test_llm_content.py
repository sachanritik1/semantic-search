from app.services.re_ranker import _parse_ranked_entries
from app.utils.llm_content import (
    extract_rerank_payload,
    normalize_llm_content,
    try_parse_json,
)


def test_normalize_structured_json_string():
    raw = '[{"type":"text","text":"[{\\"id\\": 3, \\"relevance\\": 10}]"}]'
    assert normalize_llm_content(raw) == '[{"id": 3, "relevance": 10}]'


def test_normalize_block_list():
    blocks = [{"type": "text", "text": '[{"id": 1, "relevance": 8}]'}]
    assert normalize_llm_content(blocks) == '[{"id": 1, "relevance": 8}]'


def test_try_parse_json_python_repr():
    raw = """[{'type': 'text', 'text': '[{"id": 1, "relevance": 8}]'}]"""
    parsed = try_parse_json(raw)
    assert isinstance(parsed, list)


def test_extract_rerank_payload_nested():
    raw = '[{"type":"text","text":"[{\\"id\\": 3, \\"relevance\\": 10}]"}]'
    payload = extract_rerank_payload(raw)
    assert payload == [{"id": 3, "relevance": 10}]


def test_parse_after_normalize_structured_response():
    raw = '[{"type":"text","text":"[{\\"id\\": 3, \\"relevance\\": 10}]"}]'
    assert _parse_ranked_entries(raw, 5) == [(3, 10.0)]


def test_parse_broken_wrapper_json_unescaped_inner():
    """Provider returns invalid outer JSON when inner array quotes are not escaped."""
    raw = '[{"type":"text","text":"[{"id": 3, "relevance": 10}]"}]'
    assert _parse_ranked_entries(raw, 5) == [(3, 10.0)]
