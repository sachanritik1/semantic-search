import logging

from langchain_huggingface import HuggingFaceEmbeddings

from app.config import settings
from app.utils.huggingface import configure_hf_hub

configure_hf_hub()

logger = logging.getLogger(__name__)

_embeddings: HuggingFaceEmbeddings | None = None


def get_embeddings() -> HuggingFaceEmbeddings:
	global _embeddings
	if _embeddings is None:
		logger.info("Loading embeddings model: %s", settings.EMBEDDING_MODEL_NAME)
		_embeddings = HuggingFaceEmbeddings(
			model_name=settings.EMBEDDING_MODEL_NAME,
		)
	return _embeddings
