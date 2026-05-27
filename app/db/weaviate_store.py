import json
from typing import Optional

import weaviate
from langchain_core.documents import Document
from weaviate.classes.config import Configure, DataType, Property, Tokenization, VectorDistances
from weaviate.classes.query import Filter, MetadataQuery
from weaviate.client import WeaviateClient

from app.config import settings

DOCUMENT_ID_PROPERTY = "document_id"

client: WeaviateClient | None = None


def get_weaviate_client() -> WeaviateClient:
    global client
    if client is None:
        parts = settings.WEAVIATE_URL.split(":")
        host = parts[0]
        port = int(parts[1]) if len(parts) > 1 else 8080
        grpc_parts = settings.WEAVIATE_GRPC_URL.split(":")
        grpc_port = int(grpc_parts[1]) if len(grpc_parts) > 1 else 50051
        client = weaviate.connect_to_local(
            host=host,
            port=port,
            grpc_port=grpc_port,
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


def upsert_documents(
    embeddings: list[list[float]],
    documents: list[Document],
) -> None:
    if not documents:
        return

    w = ensure_collection()
    col = w.collections.use(settings.WEAVIATE_COLLECTION_NAME)

    with w.batch.dynamic() as batch:
        for i, doc in enumerate(documents):
            meta = dict(doc.metadata or {})
            chunk_id = meta.get("chunk_id", "")
            props = {
                "content": doc.page_content,
                DOCUMENT_ID_PROPERTY: meta.get("document_id", ""),
                "chunk_id": chunk_id,
                "source": meta.get("source", ""),
                "chunk_index": meta.get("chunk_index", 0),
                "tenant_id": meta.get("tenant_id", settings.DEFAULT_TENANT_ID),
                "meta": json.dumps(meta),
            }
            batch.add_object(
                collection=settings.WEAVIATE_COLLECTION_NAME,
                properties=props,
                vector=embeddings[i],
                uuid=chunk_id if chunk_id else None,
            )

    if len(w.batch.failed_objects) > 0:
        raise RuntimeError(
            f"Failed to upsert {len(w.batch.failed_objects)} objects"
        )


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
    "bm25_search",
    "dense_search",
    "ensure_collection",
    "get_weaviate_client",
    "hybrid_search",
    "upsert_documents",
]
