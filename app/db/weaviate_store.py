import json
import logging
from typing import Optional
from urllib.parse import urlparse

import weaviate
from langchain_core.documents import Document
from weaviate.classes.config import Configure, DataType, Property, Tokenization, VectorDistances
from weaviate.classes.query import Filter, MetadataQuery
from weaviate.client import WeaviateClient

from app.config import settings

logger = logging.getLogger(__name__)

DOCUMENT_ID_PROPERTY = "document_id"

client: WeaviateClient | None = None
_grpc_available: bool | None = None


def _parse_weaviate_url() -> tuple[str, int, bool]:
    parsed = urlparse(settings.WEAVIATE_URL)
    host = parsed.hostname or "localhost"
    port = parsed.port or (443 if parsed.scheme == "https" else 8080)
    secure = parsed.scheme == "https"
    return host, port, secure


def get_weaviate_client() -> WeaviateClient:
    global client, _grpc_available
    if client is not None:
        return client

    host, port, secure = _parse_weaviate_url()

    if settings.WEAVIATE_GRPC_ENABLED and _grpc_available is not False:
        grpc_host = host
        grpc_port = settings.WEAVIATE_GRPC_PORT
        grpc_secure = secure
    else:
        grpc_host = ""
        grpc_port = 50051
        grpc_secure = False

    client = weaviate.connect_to_custom(
        http_host=host,
        http_port=port,
        http_secure=secure,
        grpc_host=grpc_host,
        grpc_port=grpc_port,
        grpc_secure=grpc_secure,
    )
    return client


def ensure_collection() -> WeaviateClient:
    w = get_weaviate_client()
    if w.collections.exists(settings.WEAVIATE_COLLECTION_NAME):
        return w

    w.collections.create(
        settings.WEAVIATE_COLLECTION_NAME,
        description="Document chunks for RAG pipeline",
        properties=[
            Property(
                name="content",
                data_type=DataType.TEXT,
                tokenization=Tokenization.WORD,
                indexFilterable=False,
                indexSearchable=True,
            ),
            Property(
                name=DOCUMENT_ID_PROPERTY,
                data_type=DataType.TEXT,
                tokenization=Tokenization.FIELD,
                indexFilterable=True,
                indexSearchable=False,
            ),
            Property(
                name="chunk_id",
                data_type=DataType.TEXT,
                tokenization=Tokenization.FIELD,
                indexFilterable=True,
                indexSearchable=False,
            ),
            Property(
                name="source",
                data_type=DataType.TEXT,
                tokenization=Tokenization.FIELD,
                indexFilterable=False,
                indexSearchable=False,
            ),
            Property(
                name="chunk_index",
                data_type=DataType.INT,
                indexFilterable=False,
                indexSearchable=False,
            ),
            Property(
                name="tenant_id",
                data_type=DataType.TEXT,
                tokenization=Tokenization.FIELD,
                indexFilterable=True,
                indexSearchable=False,
            ),
            Property(
                name="meta",
                data_type=DataType.TEXT,
                tokenization=Tokenization.FIELD,
                indexFilterable=False,
                indexSearchable=False,
            ),
        ],
        vector_config=Configure.Vectors.self_provided(
            name="default",
            vector_index_config=Configure.VectorIndex.hnsw(
                distance_metric=VectorDistances.COSINE,
            ),
        ),
    )
    return w


def _doc_props(doc: Document) -> dict:
    meta = dict(doc.metadata or {})
    return {
        "content": doc.page_content,
        DOCUMENT_ID_PROPERTY: meta.get("document_id", ""),
        "chunk_id": meta.get("chunk_id", ""),
        "source": meta.get("source", ""),
        "chunk_index": meta.get("chunk_index", 0),
        "tenant_id": meta.get("tenant_id", settings.DEFAULT_TENANT_ID),
        "meta": json.dumps(meta),
    }


def _upsert_batch(
    w: WeaviateClient,
    embeddings: list[list[float]],
    documents: list[Document],
) -> None:
    global _grpc_available
    try:
        with w.batch.dynamic() as batch:
            for i, doc in enumerate(documents):
                chunk_id = (doc.metadata or {}).get("chunk_id", "")
                batch.add_object(
                    collection=settings.WEAVIATE_COLLECTION_NAME,
                    properties=_doc_props(doc),
                    vector=embeddings[i],
                    uuid=chunk_id if chunk_id else None,
                )
        if len(w.batch.failed_objects) > 0:
            raise RuntimeError(
                f"Batch upsert: {len(w.batch.failed_objects)} failed objects"
            )
        _grpc_available = True
    except Exception as exc:
        if not _grpc_available:  # already on fallback
            raise
        logger.warning("gRPC batch failed (%s), falling back to REST inserts", exc)
        _grpc_available = False
        _upsert_rest(w, embeddings, documents)


def _upsert_rest(
    w: WeaviateClient,
    embeddings: list[list[float]],
    documents: list[Document],
) -> None:
    col = w.collections.use(settings.WEAVIATE_COLLECTION_NAME)
    for i, doc in enumerate(documents):
        chunk_id = (doc.metadata or {}).get("chunk_id", "")
        col.data.insert(
            properties=_doc_props(doc),
            vector=embeddings[i],
            uuid=chunk_id if chunk_id else None,
        )


def upsert_documents(
    embeddings: list[list[float]],
    documents: list[Document],
) -> None:
    if not documents:
        return

    w = ensure_collection()

    if settings.WEAVIATE_GRPC_ENABLED and _grpc_available is not False:
        _upsert_batch(w, embeddings, documents)
    else:
        _upsert_rest(w, embeddings, documents)


def _scored_to_doc(obj) -> tuple[Document, float]:
    props = obj.properties
    meta_raw = props.get("meta", "{}")
    try:
        metadata = json.loads(meta_raw) if isinstance(meta_raw, str) else meta_raw
    except (json.JSONDecodeError, TypeError):
        metadata = {}
    doc = Document(
        page_content=props.get("content", ""),
        metadata=metadata,
    )
    return doc, obj.metadata.score or 0.0


def document_has_chunks(document_id: str) -> bool:
    w = get_weaviate_client()
    if not w.collections.exists(settings.WEAVIATE_COLLECTION_NAME):
        return False
    col = w.collections.use(settings.WEAVIATE_COLLECTION_NAME)
    resp = col.query.fetch_objects(
        filters=Filter.by_property(DOCUMENT_ID_PROPERTY).equal(document_id),
        limit=1,
    )
    return len(resp.objects) > 0


def any_chunks_exist() -> bool:
    w = get_weaviate_client()
    if not w.collections.exists(settings.WEAVIATE_COLLECTION_NAME):
        return False
    col = w.collections.use(settings.WEAVIATE_COLLECTION_NAME)
    resp = col.query.fetch_objects(limit=1)
    return len(resp.objects) > 0


def hybrid_search(
    query_text: str,
    query_embedding: list[float],
    *,
    document_id: str | None = None,
    limit: int = 10,
) -> list[tuple[Document, float]]:
    w = get_weaviate_client()
    col = w.collections.use(settings.WEAVIATE_COLLECTION_NAME)

    filters = (
        Filter.by_property(DOCUMENT_ID_PROPERTY).equal(document_id)
        if document_id
        else None
    )

    response = col.query.hybrid(
        query=query_text,
        vector=query_embedding,
        alpha=settings.HYBRID_ALPHA,
        limit=limit,
        filters=filters,
        return_metadata=MetadataQuery(score=True),
    )

    return [_scored_to_doc(obj) for obj in response.objects]


def dense_search(
    query_embedding: list[float],
    *,
    document_id: str | None = None,
    limit: int = 10,
) -> list[tuple[Document, float]]:
    w = get_weaviate_client()
    col = w.collections.use(settings.WEAVIATE_COLLECTION_NAME)

    filters = (
        Filter.by_property(DOCUMENT_ID_PROPERTY).equal(document_id)
        if document_id
        else None
    )

    response = col.query.near_vector(
        near_vector=query_embedding,
        limit=limit,
        filters=filters,
        return_metadata=MetadataQuery(score=True),
    )

    return [_scored_to_doc(obj) for obj in response.objects]


def bm25_search(
    query_text: str,
    *,
    document_id: str | None = None,
    limit: int = 10,
) -> list[tuple[Document, float]]:
    w = get_weaviate_client()
    col = w.collections.use(settings.WEAVIATE_COLLECTION_NAME)

    filters = (
        Filter.by_property(DOCUMENT_ID_PROPERTY).equal(document_id)
        if document_id
        else None
    )

    response = col.query.bm25(
        query=query_text,
        limit=limit,
        filters=filters,
        return_metadata=MetadataQuery(score=True),
    )

    return [_scored_to_doc(obj) for obj in response.objects]


__all__ = [
    "any_chunks_exist",
    "bm25_search",
    "dense_search",
    "document_has_chunks",
    "ensure_collection",
    "get_weaviate_client",
    "hybrid_search",
    "upsert_documents",
]
