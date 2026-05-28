from __future__ import annotations

from pathlib import Path
import importlib
import re
from typing import Any, Protocol, Sequence, cast

from app.config import settings


class DocumentProcessor:
    def __init__(self, source: str | Path):
        self.source = str(source)
        self._doc: Any | None = None
        self._cleaned_text: str | None = None

    def parse(self) -> Any:
        parser = _get_llamaparse_parser()
        self._doc = parser.load_data(self.source)
        return self._doc

    def clean(self) -> str:
        doc = self._doc or self.parse()
        raw_text = _export_text(doc)
        self._cleaned_text = _clean_text(raw_text)
        return self._cleaned_text

    def extract_structure(self) -> dict[str, Any]:
        doc = self._doc or self.parse()
        return _export_structure(doc)


class _ParseAdapter(Protocol):
    def load_data(self, file_path: str) -> Any: ...


def _get_llamaparse_parser() -> _ParseAdapter:
    try:
        module = importlib.import_module("llama_cloud")
        client_cls = getattr(module, "LlamaCloud")
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "llama-cloud>=2.1 is required. Install it with: pip install llama-cloud>=2.1"
        ) from exc

    api_key = _get_llamaparse_api_key()
    kwargs: dict[str, Any] = {}
    if api_key:
        kwargs["api_key"] = api_key
    client = client_cls(**kwargs)
    return cast(_ParseAdapter, _CloudParseAdapter(client))


class _CloudParseAdapter:
    def __init__(self, client: Any) -> None:
        self._client = client

    def load_data(self, file_path: str) -> Any:
        file = self._client.files.create(file=file_path, purpose="parse")
        return self._client.parsing.parse(
            file_id=file.id,
            tier="agentic",
            version="latest",
            expand=["markdown"],
        )


def _get_llamaparse_api_key() -> str | None:
    return settings.LLAMAPARSE_API_KEY or settings.LLAMA_CLOUD_API_KEY


def _export_text(doc: Any) -> str:
    markdown = _extract_markdown(doc)
    if markdown:
        return markdown
    if isinstance(doc, (list, tuple)):
        doc_items = cast(Sequence[Any], doc)
        parts = [_extract_text(item) for item in doc_items]
        return "\n\n".join(part for part in parts if part)
    for method_name in ("export_to_markdown", "export_to_text"):
        method = getattr(doc, method_name, None)
        if callable(method):
            return str(method())
    return str(doc)


def _extract_markdown(doc: Any) -> str | None:
    m = getattr(doc, "markdown", None)
    if m is None:
        return None
    pages = getattr(m, "pages", None)
    if isinstance(pages, (list, tuple)):
        parts = [
            getattr(p, "markdown", None)
            for p in pages
            if getattr(p, "markdown", None) and isinstance(getattr(p, "markdown"), str)
        ]
        if parts:
            return "\n\n".join(parts)
    raw = getattr(m, "markdown", None)
    if isinstance(raw, str):
        return raw
    return None


def _export_structure(doc: Any) -> dict[str, Any]:
    markdown = _extract_markdown(doc)
    if markdown is not None:
        if hasattr(doc, "model_dump") and callable(doc.model_dump):
            return cast(dict[str, Any], doc.model_dump())
        return {"text": markdown, "source": "llama_cloud"}
    if isinstance(doc, (list, tuple)):
        doc_items = cast(Sequence[Any], doc)
        items: list[dict[str, Any]] = []
        for item in doc_items:
            metadata = getattr(item, "metadata", None)
            if not isinstance(metadata, dict):
                metadata = {}
            items.append(
                {
                    "text": _extract_text(item),
                    "metadata": metadata,
                }
            )
        return {"pages": items}
    for method_name in ("model_dump", "export_to_dict", "to_dict", "dict"):
        method = getattr(doc, method_name, None)
        if callable(method):
            result = method()
            if isinstance(result, dict):
                return cast(dict[str, Any], result)

    return {"raw": str(doc)}


def _extract_text(item: Any) -> str:
    text = getattr(item, "text", None)
    if isinstance(text, str) and text:
        return text
    get_text = getattr(item, "get_text", None)
    if callable(get_text):
        value = get_text()
        if isinstance(value, str) and value:
            return value
    return str(item).strip()


def _clean_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()
