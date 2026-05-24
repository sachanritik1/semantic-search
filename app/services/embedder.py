from langchain_huggingface import HuggingFaceEmbeddings

from app.config import settings
from app.utils.huggingface import configure_hf_hub

configure_hf_hub()

embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL_NAME)
