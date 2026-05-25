from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

from rank_bm25 import BM25Okapi  # type: ignore


Tokenizer = Callable[[str], List[str]]


def simple_tokenize(text: str) -> List[str]:
    """Very small default tokenizer: splits on whitespace."""
    return text.split()


@dataclass
class SparseRetriever:
    """Lightweight sparse retriever wrapper around BM25Okapi.

    Usage:
    - Create an instance.
    - Call `build_index(documents)` with an iterable of strings.
    - Call `query(q, top_k)` to get top results as (idx, score, doc).
    """

    tokenizer: Tokenizer = simple_tokenize
    docs: List[str] = field(default_factory=list)
    tokenized_docs: List[List[str]] = field(default_factory=list)
    bm25: Optional[BM25Okapi] = None

    def build_index(
        self, documents: Iterable[str], tokenizer: Optional[Tokenizer] = None
    ) -> None:
        """Builds the BM25 index from the provided documents.

        If `tokenizer` is provided it will be used instead of the default.
        """
        if tokenizer is not None:
            self.tokenizer = tokenizer

        self.docs = list(documents)
        self.tokenized_docs = [self.tokenizer(d) for d in self.docs]
        self.bm25 = BM25Okapi(self.tokenized_docs)

    def add_documents(self, documents: Iterable[str]) -> None:
        """Append new documents and rebuild the index."""
        self.docs.extend(documents)
        self.tokenized_docs = [self.tokenizer(d) for d in self.docs]
        self.bm25 = BM25Okapi(self.tokenized_docs)

    def query(self, query: str, top_k: int = 5) -> List[Tuple[int, float, str]]:
        """Return top_k results as (doc_index, score, document_text).

        If the index hasn't been built yet, raises a RuntimeError.
        """
        if self.bm25 is None:
            raise RuntimeError("BM25 index not built. Call `build_index()` first.")

        tokenized_query = self.tokenizer(query)
        scores: Sequence[float] = self.bm25.get_scores(tokenized_query)
        indexed = list(enumerate(scores))
        indexed.sort(key=lambda x: x[1], reverse=True)
        top = indexed[:top_k]
        return [(i, float(score), self.docs[i]) for i, score in top]


__all__ = ["SparseRetriever", "simple_tokenize"]
