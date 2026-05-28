import json
import logging
import threading
from typing import Optional
from urllib.parse import urlparse

import weaviate
from langchain_core.documents import Document
from weaviate.classes.config import Configure, DataType, Property, Tokenization, VectorDistances
from weaviate.classes.query import Filter, MetadataQuery
from weaviate.client import WeaviateClient

logger = logging.getLogger(__name__)

DOCUMENT_ID_PROPERTY = "document_id"


class VectorStore:
    """Adapter for Weaviate vector storage of document chunks."""

    def __init__(
        self,
        url: str = "http://localhost:8080",
        grpc_port: int = 50051,
        grpc_enabled: bool = True,
        collection_name: str = "DocumentChunk",
        default_tenant_id: str = "default",
        hybrid_alpha: float = 0.5,
    ) -> None:
        self._url = url
        self._grpc_port = grpc_port
        self._grpc_enabled = grpc_enabled
        self._collection_name = collection_name
        self._default_tenant_id = default_tenant_id
        self._hybrid_alpha = hybrid_alpha

        self._client: WeaviateClient | None = None
        self._grpc_available: bool | None = None
        self._lock = threading.Lock()

    def _parse_url(self) -> tuple[str, int, bool]:
        parsed = urlparse(self._url)
        host = parsed.hostname or "localhost"
        port = parsed.port or (443 if parsed.scheme == "https" else 8080)
        secure = parsed.scheme == "https"
        return host, port, secure

    def _connect(self) -> WeaviateClient:
        with self._lock:
            if self._client is not None:
                return self._client

            host, port, secure = self._parse_url()

            if self._grpc_enabled and self._grpc_available is not False:
                grpc_host = host
                grpc_port = self._grpc_port
                grpc_secure = secure
            else:
                grpc_host = ""
                grpc_port = 50051
                grpc_secure = False

            self._client = weaviate.connect_to_custom(
                http_host=host,
                http_port=port,
                http_secure=secure,
                grpc_host=grpc_host,
                grpc_port=grpc_port,
                grpc_secure=grpc_secure,
            )
            return self._client

    @property
    def _w(self) -> WeaviateClient:
        if self._client is None:
            return self._connect()
        return self._client

    def _doc_props(self, doc: Document) -> dict:
        meta = dict(doc.metadata or {})
        return {
            "content": doc.page_content,
            DOCUMENT_ID_PROPERTY: meta.get("document_id", ""),
            "chunk_id": meta.get("chunk_id", ""),
            "source": meta.get("source", ""),
            "chunk_index": meta.get("chunk_index", 0),
            "tenant_id": meta.get("tenant_id", self._default_tenant_id),
            "meta": json.dumps(meta),
        }

    def _scored_to_doc(self, obj) -> tuple[Document, float]:
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

    def ensure_collection(self) -> WeaviateClient:
        w = self._w
        if w.collections.exists(self._collection_name):
            return w

        w.collections.create(
            self._collection_name,
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

    def _upsert_rest(
        self,
        embeddings: list[list[float]],
        documents: list[Document],
    ) -> None:
        col = self._w.collections.use(self._collection_name)
        for i, doc in enumerate(documents):
            chunk_id = (doc.metadata or {}).get("chunk_id", "")
            col.data.insert(
                properties=self._doc_props(doc),
                vector=embeddings[i],
                uuid=chunk_id if chunk_id else None,
            )

    def _upsert_batch(
        self,
        embeddings: list[list[float]],
        documents: list[Document],
    ) -> None:
        try:
            with self._w.batch.dynamic() as batch:
                for i, doc in enumerate(documents):
                    chunk_id = (doc.metadata or {}).get("chunk_id", "")
                    batch.add_object(
                        collection=self._collection_name,
                        properties=self._doc_props(doc),
                        vector=embeddings[i],
                        uuid=chunk_id if chunk_id else None,
                    )
            if len(self._w.batch.failed_objects) > 0:
                raise RuntimeError(
                    f"Batch upsert: {len(self._w.batch.failed_objects)} failed objects"
                )
            self._grpc_available = True
        except Exception as exc:
            if self._grpc_available is False:
                raise
            logger.warning("gRPC batch failed (%s), falling back to REST inserts", exc)
            self._grpc_available = False
            self._upsert_rest(embeddings, documents)

    def upsert(
        self,
        embeddings: list[list[float]],
        documents: list[Document],
    ) -> None:
        if not documents:
            return

        self.ensure_collection()

        if self._grpc_enabled and self._grpc_available is not False:
            self._upsert_batch(embeddings, documents)
        else:
            self._upsert_rest(embeddings, documents)

    def document_has_chunks(self, document_id: str) -> bool:
        if not self._w.collections.exists(self._collection_name):
            return False
        col = self._w.collections.use(self._collection_name)
        resp = col.query.fetch_objects(
            filters=Filter.by_property(DOCUMENT_ID_PROPERTY).equal(document_id),
            limit=1,
        )
        return len(resp.objects) > 0

    def any_chunks_exist(self) -> bool:
        if not self._w.collections.exists(self._collection_name):
            return False
        col = self._w.collections.use(self._collection_name)
        resp = col.query.fetch_objects(limit=1)
        return len(resp.objects) > 0

    def hybrid_search(
        self,
        query_text: str,
        query_embedding: list[float],
        *,
        document_id: str | None = None,
        limit: int = 10,
    ) -> list[tuple[Document, float]]:
        col = self._w.collections.use(self._collection_name)

        filters = (
            Filter.by_property(DOCUMENT_ID_PROPERTY).equal(document_id)
            if document_id
            else None
        )

        response = col.query.hybrid(
            query=query_text,
            vector=query_embedding,
            alpha=self._hybrid_alpha,
            limit=limit,
            filters=filters,
            return_metadata=MetadataQuery(score=True),
        )

        return [self._scored_to_doc(obj) for obj in response.objects]

    def dense_search(
        self,
        query_embedding: list[float],
        *,
        document_id: str | None = None,
        limit: int = 10,
    ) -> list[tuple[Document, float]]:
        col = self._w.collections.use(self._collection_name)

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

        return [self._scored_to_doc(obj) for obj in response.objects]

    def bm25_search(
        self,
        query_text: str,
        *,
        document_id: str | None = None,
        limit: int = 10,
    ) -> list[tuple[Document, float]]:
        col = self._w.collections.use(self._collection_name)

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

        return [self._scored_to_doc(obj) for obj in response.objects]


# Backward-compatible module-level functions delegate to a default instance
_default_store: VectorStore | None = None


def _default_vector_store() -> VectorStore:
    global _default_store
    if _default_store is None:
        from app.config import settings

        _default_store = VectorStore(
            url=settings.WEAVIATE_URL,
            grpc_port=settings.WEAVIATE_GRPC_PORT,
            grpc_enabled=settings.WEAVIATE_GRPC_ENABLED,
            collection_name=settings.WEAVIATE_COLLECTION_NAME,
            default_tenant_id=settings.DEFAULT_TENANT_ID,
            hybrid_alpha=settings.HYBRID_ALPHA,
        )
    return _default_store


def ensure_collection() -> WeaviateClient:
    return _default_vector_store().ensure_collection()


def upsert_documents(
    embeddings: list[list[float]],
    documents: list[Document],
) -> None:
    return _default_vector_store().upsert(embeddings, documents)


def document_has_chunks(document_id: str) -> bool:
    return _default_vector_store().document_has_chunks(document_id)


def any_chunks_exist() -> bool:
    return _default_vector_store().any_chunks_exist()


def hybrid_search(
    query_text: str,
    query_embedding: list[float],
    *,
    document_id: str | None = None,
    limit: int = 10,
) -> list[tuple[Document, float]]:
    return _default_vector_store().hybrid_search(
        query_text, query_embedding, document_id=document_id, limit=limit
    )


def dense_search(
    query_embedding: list[float],
    *,
    document_id: str | None = None,
    limit: int = 10,
) -> list[tuple[Document, float]]:
    return _default_vector_store().dense_search(
        query_embedding, document_id=document_id, limit=limit
    )


def bm25_search(
    query_text: str,
    *,
    document_id: str | None = None,
    limit: int = 10,
) -> list[tuple[Document, float]]:
    return _default_vector_store().bm25_search(
        query_text, document_id=document_id, limit=limit
    )


def get_weaviate_client() -> WeaviateClient:
    return _default_vector_store()._w


__all__ = [
    "any_chunks_exist",
    "bm25_search",
    "dense_search",
    "document_has_chunks",
    "ensure_collection",
    "get_weaviate_client",
    "hybrid_search",
    "upsert_documents",
    "VectorStore",
]
