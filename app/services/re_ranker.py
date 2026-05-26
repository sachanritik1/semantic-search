import logging
from dataclasses import dataclass
from typing import Any

from huggingface_hub import InferenceClient
from langchain_core.documents import Document
from langfuse import get_client, observe

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class RerankResult:
    docs: list[Document]
    failed: bool


class HuggingFaceApiReranker:
    def __init__(
        self,
        model_name: str,
        token: str | None,
        provider: str = "hf-inference",
    ) -> None:
        self._model_name = model_name
        self._client = InferenceClient(
            provider=provider,
            api_key=token or None,
        )

    @staticmethod
    def _score_from_item(item: Any) -> float:
        if isinstance(item, dict):
            if "score" in item:
                return float(item["score"])
            scores = item.get("scores")
            if isinstance(scores, list) and scores:
                return float(max(scores))
            if "similarity" in item:
                return float(item["similarity"])
        if isinstance(item, list):
            if item and isinstance(item[0], dict):
                return float(max(entry.get("score", 0.0) for entry in item))
            if item and isinstance(item[0], (int, float)):
                return float(max(item))
        if isinstance(item, (int, float)):
            return float(item)
        return 0.0

    @classmethod
    def _parse_scores(cls, payload: Any, expected: int) -> list[float]:
        if isinstance(payload, list):
            if not payload:
                return []
            if len(payload) == expected and isinstance(payload[0], (int, float)):
                return [float(value) for value in payload]
            if len(payload) == expected and isinstance(payload[0], list):
                return [cls._score_from_item(item) for item in payload]
            if len(payload) == expected and isinstance(payload[0], dict):
                return [cls._score_from_item(item) for item in payload]
            if isinstance(payload[0], dict):
                return [cls._score_from_item(payload)]
        return [cls._score_from_item(payload)]

    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []
        payload = [{"text": query, "text_pair": passage} for query, passage in pairs]
        output = self._client.text_classification(
            payload,
            model=self._model_name,
        )
        return self._parse_scores(output, expected=len(pairs))


_model: HuggingFaceApiReranker | None = None


def _retrieval_methods(meta: dict[str, Any]) -> list[str]:
    methods = meta.get("retrieval_methods")
    if methods:
        return list(methods)
    inferred: list[str] = []
    if meta.get("dense_score") is not None:
        inferred.append("dense")
    if meta.get("sparse_score") is not None:
        inferred.append("sparse")
    return inferred


def _docs_to_trace_list(docs: list[Document]) -> list[dict[str, Any]]:
    return [
        {
            "chunk_id": (doc.metadata or {}).get("chunk_id"),
            "document_id": (doc.metadata or {}).get("document_id"),
            "chunk_index": (doc.metadata or {}).get("chunk_index"),
            "source": (doc.metadata or {}).get("source"),
            "fusion_score": (doc.metadata or {}).get("fusion_score"),
            "retrieval_methods": _retrieval_methods(doc.metadata or {}),
            "content": doc.page_content,
        }
        for doc in docs
    ]


def _trace_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    docs = inputs.get("docs") or []
    return {
        "query": inputs.get("query"),
        "top_n": inputs.get("top_n"),
        "candidate_count": len(docs),
        "min_relevance": settings.RERANK_MIN_RELEVANCE,
        "model": settings.RERANK_MODEL_NAME,
        "input_docs": _docs_to_trace_list(docs),
    }


def _chunk_label(doc: Document, doc_id: int) -> str:
    chunk_id = (doc.metadata or {}).get("chunk_id")
    return str(chunk_id) if chunk_id is not None else f"doc-{doc_id}"


def _build_rerank_trace(
    *,
    query: str,
    docs: list[Document],
    entries: list[tuple[int, float]],
    raw_logits: dict[int, float],
    selected_ids: set[int],
    ranked: list[Document],
    status: str,
    top_n: int,
    min_relevance: float,
    error: str | None = None,
) -> dict[str, Any]:
    candidates = []
    if entries:
        for doc_id, score in entries:
            source = docs[doc_id - 1]
            meta = source.metadata or {}
            candidates.append(
                {
                    "chunk_id": meta.get("chunk_id"),
                    "document_id": meta.get("document_id"),
                    "chunk_index": meta.get("chunk_index"),
                    "source": meta.get("source"),
                    "fusion_score": meta.get("fusion_score"),
                    "retrieval_methods": _retrieval_methods(meta),
                    "rerank_logit": raw_logits.get(doc_id),
                    "rerank_score": score,
                    "selected": doc_id in selected_ids,
                    "content": source.page_content,
                }
            )
    else:
        candidates = [
            {**entry, "rerank_score": None, "selected": False}
            for entry in _docs_to_trace_list(docs)
        ]

    return {
        "status": status,
        "query": query,
        "top_n": top_n,
        "min_relevance": min_relevance,
        "candidate_count": len(docs),
        "selected_count": len(ranked),
        "candidates": candidates,
        "selected_chunk_ids": [(doc.metadata or {}).get("chunk_id") for doc in ranked],
        "error": error,
    }


def _record_rerank_trace(payload: dict[str, Any]) -> None:
    get_client().update_current_span(output=payload)


def _get_reranker() -> HuggingFaceApiReranker:
    global _model
    if _model is None:
        logger.info(
            "Loading reranker via Hugging Face Inference API: %s",
            settings.RERANK_MODEL_NAME,
        )
        _model = HuggingFaceApiReranker(
            settings.RERANK_MODEL_NAME,
            token=settings.HF_TOKEN.strip() or None,
        )
    return _model


def preload_reranker() -> None:
    _get_reranker()


def _normalize_scores(raw_logits: list[float]) -> list[float]:
    """Map cross-encoder logits to 0–10 relative to this candidate batch."""
    if not raw_logits:
        return []
    lo = min(raw_logits)
    hi = max(raw_logits)
    if hi == lo:
        return [5.0] * len(raw_logits)
    return [round(10.0 * (value - lo) / (hi - lo), 2) for value in raw_logits]


def _select_relevant_entries(
    entries: list[tuple[int, float]],
    top_n: int,
    min_relevance: float,
) -> list[tuple[int, float]]:
    qualified = sorted(
        [(doc_id, score) for doc_id, score in entries if score >= min_relevance],
        key=lambda pair: pair[1],
        reverse=True,
    )
    if qualified:
        return qualified[:top_n]
    return sorted(entries, key=lambda pair: pair[1], reverse=True)[:top_n]


def _apply_ranking(
    candidates: list[Document],
    entries: list[tuple[int, float]],
    top_n: int,
    raw_logits: dict[int, float],
) -> list[Document]:
    seen: set[int] = set()
    ranked: list[Document] = []

    for doc_id, score in entries:
        if doc_id in seen or not (1 <= doc_id <= len(candidates)):
            continue
        seen.add(doc_id)
        source = candidates[doc_id - 1]
        metadata = dict(source.metadata or {})
        metadata["rerank_logit"] = raw_logits.get(doc_id)
        metadata["rerank_score"] = score
        ranked.append(Document(page_content=source.page_content, metadata=metadata))
        if len(ranked) >= top_n:
            break

    return ranked


def _score_candidates(
    query: str,
    candidates: list[Document],
) -> tuple[list[tuple[int, float]], dict[int, float]]:
    pairs = [(query, doc.page_content) for doc in candidates]
    raw_list = [float(value) for value in _get_reranker().score_pairs(pairs)]
    normalized = _normalize_scores(raw_list)
    raw_by_id = {doc_id: raw for doc_id, raw in enumerate(raw_list, start=1)}
    entries = sorted(
        enumerate(normalized, start=1),
        key=lambda item: raw_by_id[item[0]],
        reverse=True,
    )
    return [(doc_id, score) for doc_id, score in entries], raw_by_id


@observe(name="cross_encoder_rerank", capture_input=False)
def re_rank_docs(
    query: str,
    docs: list[Document],
    top_n: int = 5,
) -> RerankResult:
    get_client().update_current_span(
        input=_trace_inputs({"query": query, "docs": docs, "top_n": top_n})
    )
    if not docs:
        logger.warning("Rerank skipped: no candidates")
        _record_rerank_trace(
            _build_rerank_trace(
                query=query,
                docs=[],
                entries=[],
                raw_logits={},
                selected_ids=set(),
                ranked=[],
                status="skipped",
                top_n=top_n,
                min_relevance=settings.RERANK_MIN_RELEVANCE,
            )
        )
        return RerankResult(docs=[], failed=False)

    min_relevance = settings.RERANK_MIN_RELEVANCE
    logger.info(
        "Reranking %d candidates (top_n=%d, min_relevance=%.1f)",
        len(docs),
        top_n,
        min_relevance,
    )

    entries: list[tuple[int, float]] = []
    raw_logits: dict[int, float] = {}
    selected_ids: set[int] = set()
    try:
        entries, raw_logits = _score_candidates(query, docs)
        raw_values = list(raw_logits.values())
        logger.info(
            "Rerank raw logits: min=%.3f max=%.3f mean=%.3f",
            min(raw_values),
            max(raw_values),
            sum(raw_values) / len(raw_values),
        )
        selected = _select_relevant_entries(
            entries,
            top_n=top_n,
            min_relevance=min_relevance,
        )
        selected_ids = {doc_id for doc_id, _ in selected}

        used_fallback = not any(score >= min_relevance for _, score in entries)
        if used_fallback:
            logger.warning(
                "Rerank fallback: no scores >= %.1f, taking top %d",
                min_relevance,
                len(selected),
            )

        ranked = _apply_ranking(docs, selected, top_n, raw_logits)
        if not ranked:
            logger.warning("Rerank failed: no documents after ranking")
            _record_rerank_trace(
                _build_rerank_trace(
                    query=query,
                    docs=docs,
                    entries=entries,
                    raw_logits=raw_logits,
                    selected_ids=selected_ids,
                    ranked=[],
                    status="failed",
                    top_n=top_n,
                    min_relevance=min_relevance,
                    error="no_documents_after_ranking",
                )
            )
            return RerankResult(docs=[], failed=True)

        logger.info("Rerank ok: %d/%d candidates selected", len(ranked), len(docs))
        _record_rerank_trace(
            _build_rerank_trace(
                query=query,
                docs=docs,
                entries=entries,
                raw_logits=raw_logits,
                selected_ids=selected_ids,
                ranked=ranked,
                status="fallback" if used_fallback else "success",
                top_n=top_n,
                min_relevance=min_relevance,
            )
        )
        return RerankResult(docs=ranked, failed=False)
    except Exception as exc:
        logger.exception("Rerank failed: cross-encoder error")
        _record_rerank_trace(
            _build_rerank_trace(
                query=query,
                docs=docs,
                entries=entries,
                raw_logits=raw_logits,
                selected_ids=selected_ids,
                ranked=[],
                status="error",
                top_n=top_n,
                min_relevance=min_relevance,
                error=str(exc),
            )
        )
        return RerankResult(docs=[], failed=True)
