import os

from app.config import settings


def configure_hf_hub() -> None:
    """Set HF token from settings so Hub and Inference API calls can authenticate."""
    token = settings.HF_TOKEN.strip()
    if token:
        os.environ["HF_TOKEN"] = token
