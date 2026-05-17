from dataclasses import dataclass
from typing import List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.db.vector_store import get_vector_store


@dataclass
class DenseRetriever:
    embeddings: Embeddings
    default_k: int = 20

    def retrieve(self, query: str, k: Optional[int] = None) -> List[Document]:
        return [doc for doc, _ in self.retrieve_with_scores(query, k=k)]

    def retrieve_with_scores(
        self, query: str, k: Optional[int] = None
    ) -> List[Tuple[Document, float]]:
        vector_store = get_vector_store(self.embeddings)
        return vector_store.similarity_search_with_relevance_scores(
            query, k=k or self.default_k
        )


__all__ = ["DenseRetriever"]
