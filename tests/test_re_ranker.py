import pytest

from app.services.re_ranker import (
    RerankResult,
    _apply_ranking,
    _entries_degenerate,
    _entries_from_fusion,
    _parse_ranked_entries,
    _select_relevant_entries,
)
from langchain_core.documents import Document


def test_parse_ranked_entries_integer_array():
    assert _parse_ranked_entries("[2, 5, 1]", 5) == [(2, 5.0), (5, 4.0), (1, 3.0)]


def test_parse_ranked_entries_score_objects_sorted_by_relevance():
    raw = '[{"id": 3, "relevance": 9}, {"id": 1, "score": 4}]'
    assert _parse_ranked_entries(raw, 5) == [(3, 9.0), (1, 4.0)]


def test_parse_ranked_entries_structured_wrapper():
    raw = '[{"type":"text","text":"[{\\"id\\": 3, \\"relevance\\": 10}]"}]'
    assert _parse_ranked_entries(raw, 5) == [(3, 10.0)]


def test_parse_ranked_entries_python_repr_wrapper():
    raw = """[{'type': 'text', 'text': '[{"id": 2, "relevance": 8}, {"id": 4, "relevance": 6}]'}]"""
    entries = _parse_ranked_entries(raw, 5)
    assert (2, 8.0) in entries
    assert (4, 6.0) in entries


def test_apply_ranking_selected_only():
    docs = [Document(page_content=f"doc-{i}") for i in range(1, 6)]
    ranked = _apply_ranking(docs, [(2, 9.0), (5, 7.0)], top_n=5)
    assert len(ranked) == 2
    assert [d.page_content for d in ranked] == ["doc-2", "doc-5"]


def test_select_relevant_entries_filters_by_threshold():
    entries = [(1, 3.0), (2, 8.0), (3, 6.0), (4, 2.0)]
    selected = _select_relevant_entries(entries, top_n=5, min_relevance=4.0)
    assert selected == [(2, 8.0), (3, 6.0)]


def test_select_relevant_entries_fallback_to_top_scores():
    entries = [(1, 3.0), (2, 2.0), (3, 1.0)]
    selected = _select_relevant_entries(entries, top_n=2, min_relevance=4.0)
    assert selected == [(1, 3.0), (2, 2.0)]


def test_parse_ranked_entries_empty_raises():
    with pytest.raises(ValueError, match="Invalid rerank response"):
        _parse_ranked_entries("[]", 5)


def test_parse_ranked_entries_invalid_raises():
    with pytest.raises(ValueError, match="Invalid rerank response"):
        _parse_ranked_entries("no json here", 5)


def test_entries_degenerate_all_zero():
    assert _entries_degenerate([(1, 0.0), (2, 0.0)]) is True


def test_entries_degenerate_has_signal():
    assert _entries_degenerate([(1, 0.0), (2, 7.0)]) is False


def test_entries_from_fusion_orders_by_score():
    docs = [
        Document(page_content="a", metadata={"fusion_score": 0.2}),
        Document(page_content="b", metadata={"fusion_score": 0.9}),
        Document(page_content="c", metadata={"fusion_score": 0.5}),
    ]
    entries = _entries_from_fusion(docs)
    assert entries[0] == (2, 9.0)
    assert entries[1] == (3, 5.0)


def test_rerank_result_dataclass():
    doc = Document(page_content="x")
    result = RerankResult(docs=[doc], failed=False)
    assert result.docs == [doc]
    assert result.failed is False
