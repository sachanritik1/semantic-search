from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from app.services.re_ranker import (
    RerankResult,
    _apply_ranking,
    _docs_to_trace_list,
    _normalize_scores,
    _select_relevant_entries,
    re_rank_docs,
)


def test_normalize_scores_spreads_batch_to_zero_ten():
    assert _normalize_scores([-5.0, 8.0]) == [0.0, 10.0]


def test_normalize_scores_identical_logits_are_neutral():
    assert _normalize_scores([1.0, 1.0, 1.0]) == [5.0, 5.0, 5.0]


def test_normalize_scores_empty():
    assert _normalize_scores([]) == []


def test_apply_ranking_selected_only():
    docs = [Document(page_content=f"doc-{i}") for i in range(1, 6)]
    ranked = _apply_ranking(
        docs,
        [(2, 9.0), (5, 7.0)],
        top_n=5,
        raw_logits={2: 2.0, 5: 1.0},
    )
    assert len(ranked) == 2
    assert [d.page_content for d in ranked] == ["doc-2", "doc-5"]
    assert ranked[0].metadata["rerank_logit"] == 2.0


def test_select_relevant_entries_filters_by_threshold():
    entries = [(1, 3.0), (2, 8.0), (3, 6.0), (4, 2.0)]
    selected = _select_relevant_entries(entries, top_n=5, min_relevance=4.0)
    assert selected == [(2, 8.0), (3, 6.0)]


def test_select_relevant_entries_fallback_to_top_scores():
    entries = [(1, 3.0), (2, 2.0), (3, 1.0)]
    selected = _select_relevant_entries(entries, top_n=2, min_relevance=4.0)
    assert selected == [(1, 3.0), (2, 2.0)]


def test_rerank_result_dataclass():
    doc = Document(page_content="x")
    result = RerankResult(docs=[doc], failed=False)
    assert result.docs == [doc]
    assert result.failed is False


def test_re_rank_docs_ranks_by_cross_encoder_logits():
    docs = [
        Document(page_content="low", metadata={"fusion_score": 0.1}),
        Document(page_content="high", metadata={"fusion_score": 0.2}),
    ]
    mock_model = MagicMock()
    mock_model.predict.return_value = [-5.0, 8.0]

    with (
        patch("app.services.re_ranker._get_cross_encoder", return_value=mock_model),
        patch("app.services.re_ranker.settings.RERANK_MIN_RELEVANCE", 4.0),
    ):
        result = re_rank_docs("query", docs, top_n=2)

    assert result.failed is False
    assert len(result.docs) == 1
    assert result.docs[0].page_content == "high"
    assert result.docs[0].metadata["rerank_score"] == 10.0
    assert result.docs[0].metadata["rerank_logit"] == 8.0


def test_re_rank_docs_returns_failed_on_predict_error():
    mock_model = MagicMock()
    mock_model.predict.side_effect = RuntimeError("model error")

    with patch("app.services.re_ranker._get_cross_encoder", return_value=mock_model):
        result = re_rank_docs(
            "query",
            [Document(page_content="a")],
            top_n=1,
        )

    assert result.failed is True
    assert result.docs == []


def test_re_rank_docs_empty_input():
    result = re_rank_docs("query", [])
    assert result.failed is False
    assert result.docs == []


def test_docs_to_trace_list_includes_retrieval_methods():
    docs = [
        Document(
            page_content="both",
            metadata={
                "chunk_id": "c1",
                "retrieval_methods": ["dense", "sparse"],
            },
        ),
        Document(
            page_content="dense only",
            metadata={"chunk_id": "c2", "dense_score": 0.5},
        ),
    ]
    traced = _docs_to_trace_list(docs)
    assert traced[0]["retrieval_methods"] == ["dense", "sparse"]
    assert traced[1]["retrieval_methods"] == ["dense"]
