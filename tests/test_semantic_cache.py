from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import select

from app.db import ask_cache_store
from app.db.ask_cache_store import AskCacheRow
from app.services.semantic_cache import SemanticAskCache

_UTC = timezone.utc
_T0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=_UTC)


def _mock_embeddings(
    vectors: dict[str, list[float]],
    *,
    default: list[float] | None = None,
) -> MagicMock:
    embeddings = MagicMock()
    fallback = default or [0.0, 1.0]

    def embed_query(text: str) -> list[float]:
        return vectors.get(text, fallback)

    embeddings.embed_query.side_effect = embed_query
    return embeddings


@pytest.fixture
def isolated_ask_cache(isolated_ask_cache_db):
    return SemanticAskCache(
        _mock_embeddings(
            {
                "What is the refund policy?": [1.0, 0.0],
                "Tell me about refunds": [0.99, 0.01],
                "Unrelated topic": [0.0, 1.0],
            }
        ),
        threshold=0.92,
        ttl_seconds=3600,
    )


def test_lookup_miss_then_semantic_hit(isolated_ask_cache: SemanticAskCache):
    result = {
        "response": "Refunds within 30 days.",
        "original_question": "What is the refund policy?",
        "enhanced_question": "refund policy details",
    }
    isolated_ask_cache.store("What is the refund policy?", "doc-1", result)

    assert isolated_ask_cache.lookup("Something else", "doc-1") is None

    hit = isolated_ask_cache.lookup("Tell me about refunds", "doc-1")
    assert hit == result


def test_exact_match_hit_without_embedding_scan(isolated_ask_cache: SemanticAskCache):
    result = {
        "response": "answer",
        "original_question": "  What is the refund policy?  ",
        "enhanced_question": "enhanced",
    }
    isolated_ask_cache.store("What is the refund policy?", "doc-1", result)

    hit = isolated_ask_cache.lookup("what is the refund policy?", "doc-1")
    assert hit == result
    assert isolated_ask_cache._embeddings.embed_query.call_count == 1


def test_no_cross_document_hit(isolated_ask_cache: SemanticAskCache):
    result = {
        "response": "answer",
        "original_question": "What is the refund policy?",
        "enhanced_question": "enhanced",
    }
    isolated_ask_cache.store("What is the refund policy?", "doc-1", result)

    assert isolated_ask_cache.lookup("What is the refund policy?", "doc-2") is None


def test_below_threshold_is_miss(isolated_ask_cache: SemanticAskCache):
    result = {
        "response": "answer",
        "original_question": "What is the refund policy?",
        "enhanced_question": "enhanced",
    }
    isolated_ask_cache.store("What is the refund policy?", "doc-1", result)

    assert isolated_ask_cache.lookup("Unrelated topic", "doc-1") is None


def test_ttl_expiry(isolated_ask_cache: SemanticAskCache, monkeypatch):
    monkeypatch.setattr(isolated_ask_cache, "_ttl_seconds", 60)
    result = {
        "response": "answer",
        "original_question": "What is the refund policy?",
        "enhanced_question": "enhanced",
    }

    with patch("app.db.ask_cache_store.utc_now", return_value=_T0):
        isolated_ask_cache.store("What is the refund policy?", "doc-1", result)

    with patch(
        "app.db.ask_cache_store.utc_now",
        return_value=_T0 + timedelta(seconds=120),
    ):
        assert isolated_ask_cache.lookup("What is the refund policy?", "doc-1") is None


def test_persists_across_cache_instances(isolated_ask_cache: SemanticAskCache):
    result = {
        "response": "answer",
        "original_question": "What is the refund policy?",
        "enhanced_question": "enhanced",
    }
    isolated_ask_cache.store("What is the refund policy?", "doc-1", result)

    reloaded = SemanticAskCache(
        isolated_ask_cache._embeddings,
        threshold=0.92,
        ttl_seconds=3600,
    )
    hit = reloaded.lookup("What is the refund policy?", "doc-1")
    assert hit == result


def test_disabled_cache(isolated_ask_cache: SemanticAskCache):
    cache = SemanticAskCache(
        _mock_embeddings({"q": [1.0, 0.0]}),
        enabled=False,
    )
    result = {
        "response": "answer",
        "original_question": "q",
        "enhanced_question": "enhanced",
    }
    cache.store("q", "doc-1", result)
    assert cache.lookup("q", "doc-1") is None


def test_prune_removes_expired_entries_on_store(
    isolated_ask_cache: SemanticAskCache,
    monkeypatch,
):
    monkeypatch.setattr(isolated_ask_cache, "_ttl_seconds", 60)
    result = {
        "response": "answer",
        "original_question": "q",
        "enhanced_question": "enhanced",
    }

    with patch("app.db.ask_cache_store.utc_now", return_value=_T0):
        isolated_ask_cache.store("q", "doc-1", result)

    with ask_cache_store.SessionLocal() as session:
        assert len(session.scalars(select(AskCacheRow)).all()) == 1

    with patch(
        "app.db.ask_cache_store.utc_now",
        return_value=_T0 + timedelta(seconds=120),
    ):
        isolated_ask_cache.store("q", "doc-1", result)

    with ask_cache_store.SessionLocal() as session:
        rows = session.scalars(select(AskCacheRow)).all()
        assert len(rows) == 1
