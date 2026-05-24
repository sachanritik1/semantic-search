import os

from app.config import settings


def configure_hf_hub() -> None:
    """Set HF Hub token from settings so model downloads use authenticated requests."""
    token = settings.HF_TOKEN.strip()
    if token:
        os.environ["HF_TOKEN"] = token
