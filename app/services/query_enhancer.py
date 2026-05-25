# app/services/query_enhancer.py

import json
import re

from langfuse import get_client, observe

from app.services.llm_service import LLMService
from app.utils.prompt_cache import cache_key

_ENHANCER_SYSTEM_PROMPT = """Generate exactly 3 diverse search queries to improve document retrieval.
Rules:
- Keep the same intent and entities as the original.
- Do not invent facts or assumptions.
- Each query should use different wording, synonyms, or neutral clarifiers.
- Queries must be useful for keyword and semantic search.

Return ONLY a JSON array of 3 strings, e.g. ["query one", "query two", "query three"]."""

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

    queries = [q.strip("\"'") for q in queries if q.strip()][:_QUERY_COUNT]

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

    @observe(name="query_enhancer.enhance", capture_input=False)
    def enhance(self, query: str) -> list[str]:
        get_client().update_current_span(input=query)
        user_message = f'Original query: "{query}"'

        response = self.llm_service.generate_text(
            user_message,
            temperature=0.0,
            max_tokens=256,
            model=self.enhancer_model,
            system_prompt=_ENHANCER_SYSTEM_PROMPT,
            cache_key=cache_key("enhance"),
        )

        text = response.content or ""
        queries = _parse_queries(text, query)
        get_client().update_current_span(output=queries)
        return queries
