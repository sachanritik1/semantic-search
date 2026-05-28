import logging

import numpy as np
from langchain_core.embeddings import Embeddings

from app.adapters.cache_store import AskCacheRow, AskCacheStore

logger = logging.getLogger(__name__)


def _normalize_question(question: str) -> str:
    return question.strip().casefold()


def _normalize_vector(vector: list[float]) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float32)
    norm = np.linalg.norm(arr)
    if norm == 0:
        return arr
    return arr / norm


def _embedding_from_row(row: AskCacheRow) -> np.ndarray:
    return _normalize_vector(row.embedding)


class SemanticAskCache:
    def __init__(
        self,
        embeddings: Embeddings,
        *,
        cache_store: AskCacheStore | None = None,
        enabled: bool = True,
        threshold: float = 0.92,
        ttl_seconds: int = 3600,
    ) -> None:
        self._embeddings = embeddings
        self._cache_store = cache_store or AskCacheStore()
        self._enabled = enabled
        self._threshold = threshold
        self._ttl_seconds = ttl_seconds

    def lookup(self, question: str, document_id: str) -> dict | None:
        if not self._enabled:
            return None

        now = self._cache_store._utc_now()
        normalized = _normalize_question(question)
        live = self._cache_store.list_rows(
            document_id=document_id,
            ttl_seconds=self._ttl_seconds,
            now=now,
        )

        for row in live:
            if _normalize_question(row.original_question) == normalized:
                logger.info(
                    "Semantic cache exact hit document_id=%s",
                    document_id,
                )
                return self._to_result(row)

        if not live:
            return None

        query_embedding = _normalize_vector(self._embeddings.embed_query(question))
        best_row: AskCacheRow | None = None
        best_score = -1.0

        for row in live:
            score = float(np.dot(query_embedding, _embedding_from_row(row)))
            if score > best_score:
                best_score = score
                best_row = row

        if best_row is None or best_score < self._threshold:
            logger.info(
                "Semantic cache miss document_id=%s best_score=%.4f threshold=%.4f",
                document_id,
                best_score,
                self._threshold,
            )
            return None

        logger.info(
            "Semantic cache hit document_id=%s similarity=%.4f",
            document_id,
            best_score,
        )
        return self._to_result(best_row)

    def store(self, question: str, document_id: str, result: dict) -> None:
        if not self._enabled:
            return

        embedding = _normalize_vector(self._embeddings.embed_query(question))
        vector = embedding.tolist()
        self._cache_store.insert(
            document_id=document_id,
            original_question=result["original_question"],
            enhanced_question=result["enhanced_question"],
            response=result["response"],
            embedding=vector,
        )
        self._cache_store.prune_expired(self._ttl_seconds)

    @staticmethod
    def _to_result(row: AskCacheRow) -> dict:
        return {
            "response": row.response,
            "original_question": row.original_question,
            "enhanced_question": row.enhanced_question,
        }
