from __future__ import annotations

import ast
import json
import re
from typing import Any

# Match {"id": N, "relevance": M} anywhere in provider output (handles broken wrappers).
_RANK_OBJECT_RE = re.compile(
    r'\{\s*"id"\s*:\s*(\d+)\s*,\s*"(?:relevance|score|rating)"\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*\}',
    re.IGNORECASE,
)
_RANK_OBJECT_RE_ALT = re.compile(
    r'\{\s*"(?:relevance|score|rating)"\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*,\s*"id"\s*:\s*(\d+)\s*\}',
    re.IGNORECASE,
)
_TEXT_FIELD_ARRAY_RE = re.compile(r'["\']text["\']\s*:\s*["\'](\[.*\])["\']', re.DOTALL)


def strip_markdown_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def try_parse_json(text: str) -> Any | None:
    text = strip_markdown_fences(text)
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    try:
        return ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return None


def normalize_llm_content(content: Any) -> str:
    """Flatten LangChain / provider content blocks into plain text."""
    if content is None:
        return ""

    if isinstance(content, list):
        return _flatten_blocks(content)

    if isinstance(content, str):
        text = content.strip()
        if text.startswith("[") or text.startswith("{"):
            parsed = try_parse_json(text)
            if isinstance(parsed, list):
                flattened = _flatten_blocks(parsed)
                if flattened:
                    return flattened
            if isinstance(parsed, dict):
                inner = _flatten_blocks([parsed])
                if inner:
                    return inner
            # Broken wrapper JSON: extract inner array from "text":"[...]"
            inner = _extract_text_field_array(text)
            if inner:
                return inner
        return text

    if isinstance(content, dict):
        return _flatten_blocks([content])

    return str(content).strip()


def _extract_text_field_array(text: str) -> str | None:
    match = _TEXT_FIELD_ARRAY_RE.search(text)
    if match:
        return match.group(1).strip()
    return None


def _flatten_blocks(blocks: list[Any]) -> str:
    parts: list[str] = []
    for block in blocks:
        if isinstance(block, str):
            parts.append(block)
            continue
        if not isinstance(block, dict):
            continue

        if block.get("type") == "text" and block.get("text") is not None:
            parts.append(str(block["text"]))
        elif block.get("content") is not None:
            parts.append(str(block["content"]))
        elif block.get("text") is not None:
            parts.append(str(block["text"]))

    return "\n".join(part.strip() for part in parts if part.strip()).strip()


def _is_content_block(value: Any) -> bool:
    return isinstance(value, dict) and value.get("type") in {
        "text",
        "output_text",
        "message",
    }


def _is_rank_item(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    doc_id = value.get("id", value.get("document_id", value.get("doc_id")))
    if doc_id is None:
        return False
    return any(
        key in value
        for key in ("relevance", "score", "rating", "id", "document_id", "doc_id")
    )


def _looks_like_rank_array(values: list[Any]) -> bool:
    if not values:
        return False
    if all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in values):
        return True
    return all(_is_rank_item(item) for item in values)


def regex_extract_rank_items(text: str) -> list[dict[str, Any]]:
    """Last-resort: pull id/relevance objects from any string shape."""
    items: list[dict[str, Any]] = []
    seen: set[int] = set()

    for doc_id, score in _RANK_OBJECT_RE.findall(text):
        idx = int(doc_id)
        if idx in seen:
            continue
        seen.add(idx)
        items.append({"id": idx, "relevance": float(score)})

    for score, doc_id in _RANK_OBJECT_RE_ALT.findall(text):
        idx = int(doc_id)
        if idx in seen:
            continue
        seen.add(idx)
        items.append({"id": idx, "relevance": float(score)})

    return items


def extract_rerank_payload(content: Any) -> list[Any]:
    """Recursively unwrap provider wrappers until we reach a rank array."""
    if content is None:
        return []

    if isinstance(content, str):
        text = strip_markdown_fences(content)

        # Try regex on raw string first for broken wrapper JSON.
        regex_items = regex_extract_rank_items(text)
        if regex_items:
            return regex_items

        text = normalize_llm_content(content)
        if not text:
            return []

        regex_items = regex_extract_rank_items(text)
        if regex_items:
            return regex_items

        parsed = try_parse_json(text)
        if parsed is None:
            return []
        return extract_rerank_payload(parsed)

    if isinstance(content, dict):
        if _is_content_block(content):
            inner = content.get("text") or content.get("content")
            return extract_rerank_payload(inner)
        if _is_rank_item(content):
            return [content]
        for key in ("text", "content", "data", "result", "output"):
            if key in content:
                inner = extract_rerank_payload(content[key])
                if inner:
                    return inner
        return []

    if isinstance(content, list):
        if not content:
            return []
        if _looks_like_rank_array(content):
            return content
        if all(isinstance(item, dict) and _is_content_block(item) for item in content):
            return extract_rerank_payload(normalize_llm_content(content))
        if len(content) == 1:
            inner = extract_rerank_payload(content[0])
            if inner:
                return inner
        for item in content:
            inner = extract_rerank_payload(item)
            if _looks_like_rank_array(inner):
                return inner
        return []

    return []
