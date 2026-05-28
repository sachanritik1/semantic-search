import logging
from collections.abc import Iterable
from typing import Any

from huggingface_hub import InferenceClient
from langchain_core.embeddings import Embeddings

from app.config import settings
from app.infrastructure.utils.huggingface import configure_hf_hub

configure_hf_hub()

logger = logging.getLogger(__name__)


class HuggingFaceApiEmbeddings(Embeddings):
	def __init__(self, model_name: str, token: str | None = None) -> None:
		self._model_name = model_name
		self._client = InferenceClient(model=model_name, token=token or None)

	@staticmethod
	def _as_list(value: Any) -> list[float] | list[list[float]]:
		if hasattr(value, "tolist"):
			return value.tolist()
		return list(value) if isinstance(value, Iterable) else [value]

	def _normalize_vector(self, value: Any) -> list[float]:
		vector = self._as_list(value)
		if vector and isinstance(vector[0], list):
			vector = vector[0]
		return [float(item) for item in vector]

	def _normalize_vectors(self, value: Any) -> list[list[float]]:
		vectors = self._as_list(value)
		if not vectors:
			return []
		if isinstance(vectors[0], (float, int)):
			return [self._normalize_vector(vectors)]
		return [self._normalize_vector(vector) for vector in vectors]

	def embed_query(self, text: str) -> list[float]:
		output = self._client.feature_extraction(text)
		return self._normalize_vector(output)

	def embed_documents(self, texts: list[str]) -> list[list[float]]:
		if not texts:
			return []
		output = self._client.feature_extraction(texts)
		return self._normalize_vectors(output)


_embeddings: Embeddings | None = None


def get_embeddings() -> Embeddings:
	global _embeddings
	if _embeddings is None:
		logger.info(
			"Loading embeddings model via Hugging Face Inference API: %s",
			settings.EMBEDDING_MODEL_NAME,
		)
		_embeddings = HuggingFaceApiEmbeddings(
			model_name=settings.EMBEDDING_MODEL_NAME,
			token=settings.HF_TOKEN.strip() or None,
		)
	return _embeddings
