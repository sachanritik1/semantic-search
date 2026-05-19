# app/services/query_enhancer.py

import json
import re

from app.services.llm_service import LLMService

_QUERY_COUNT = 3


def _parse_queries(text: str, original: str) -> list[str]:
    """Parse LLM output into up to three retrieval queries."""
    stripped = text.strip()
    if not stripped:
        return [original, original, original]

    queries: list[str] = []

    # Try JSON array first
    candidate = stripped
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.IGNORECASE)
        candidate = re.sub(r"\s*```$", "", candidate).strip()

    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, list):
            queries = [str(item).strip() for item in parsed if str(item).strip()]
    except json.JSONDecodeError:
        pass

    # Fallback: numbered lines or plain lines
    if not queries:
        numbered = re.findall(
            r"^\s*\d+[\.\)]\s*(.+)$",
            stripped,
            flags=re.MULTILINE,
        )
        if numbered:
            queries = [line.strip() for line in numbered if line.strip()]
        else:
            queries = [line.strip() for line in stripped.splitlines() if line.strip()]

    queries = [q.strip("\"'") for q in queries if q.strip()][: _QUERY_COUNT]

    if not queries:
        queries = [original]

    while len(queries) < _QUERY_COUNT:
        queries.append(original)

    return queries


class QueryEnhancer:
    """LLM-based query enhancer that produces multiple retrieval queries."""

    def __init__(self, llm_service: LLMService, enhancer_model: str | None = None):
        self.llm_service = llm_service
        self.enhancer_model = enhancer_model

    def enhance(self, query: str) -> list[str]:
        prompt = (
            "Generate exactly 3 diverse search queries to improve document retrieval.\n"
            "Rules:\n"
            "- Keep the same intent and entities as the original.\n"
            "- Do not invent facts or assumptions.\n"
            "- Each query should use different wording, synonyms, or neutral clarifiers.\n"
            "- Queries must be useful for keyword and semantic search.\n\n"
            f"Original query: \"{query}\"\n\n"
            'Return ONLY a JSON array of 3 strings, e.g. ["query one", "query two", "query three"].'
        )

        response = self.llm_service.generate_text(
            prompt,
            temperature=0.0,
            max_tokens=256,
            model=self.enhancer_model,
        )

        text = response.content or ""
        return _parse_queries(text, query)
