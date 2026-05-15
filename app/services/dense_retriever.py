from dataclasses import dataclass
from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.services.vector_store import get_vector_store


@dataclass
class DenseRetriever:
    embeddings: Embeddings
    default_k: int = 20

    def retrieve(self, query: str, k: Optional[int] = None) -> List[Document]:
        vector_store = get_vector_store(self.embeddings)
        retriever = vector_store.as_retriever(search_kwargs={"k": k or self.default_k})
        return list(retriever.invoke(query))


__all__ = ["DenseRetriever"]
