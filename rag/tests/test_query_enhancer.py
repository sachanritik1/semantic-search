import json

from app.adapters.llm.base import LLMResponse
from app.domain.query_enhancer import QueryEnhancer, _parse_queries


class DummyLLMService:
    def __init__(self, content: str):
        self.content = content
        self.called = False

    def generate_text(
        self,
        prompt: str,
        *,
        temperature=0.0,
        max_tokens=None,
        model=None,
        system_prompt=None,
        cache_key=None,
    ):
        self.called = True
        self.system_prompt = system_prompt
        self.cache_key = cache_key
        return LLMResponse(content=self.content)


def test_enhancer_parses_json_array():
    queries = ["synonym query", "entity focused", "broader terms"]
    dummy = DummyLLMService(json.dumps(queries))
    enhancer = QueryEnhancer(dummy)
    result = enhancer.enhance("original query")
    assert dummy.called is True
    assert result == queries


def test_enhancer_parses_numbered_list_fallback():
    content = "1. first variant\n2. second variant\n3. third variant"
    enhancer = QueryEnhancer(DummyLLMService(content))
    assert enhancer.enhance("original") == [
        "first variant",
        "second variant",
        "third variant",
    ]


def test_enhancer_pads_to_three_when_fewer_parsed():
    enhancer = QueryEnhancer(DummyLLMService(json.dumps(["only one"])))
    assert enhancer.enhance("original") == ["only one", "original", "original"]


def test_parse_queries_empty_pads_to_three():
    assert _parse_queries("", "original") == [
        "original",
        "original",
        "original",
    ]


def test_parse_queries_strips_markdown_fence():
    payload = json.dumps(["a", "b", "c"])
    text = f"```json\n{payload}\n```"
    assert _parse_queries(text, "original") == ["a", "b", "c"]


def test_enhancer_falls_back_to_original_when_llm_returns_empty():
    class EmptyLLMService:
        def generate_text(
            self,
            prompt: str,
            *,
            temperature=0.0,
            max_tokens=None,
            model=None,
            system_prompt=None,
            cache_key=None,
        ):
            return LLMResponse(content="")

    enhancer = QueryEnhancer(EmptyLLMService())
    assert enhancer.enhance("original query") == [
        "original query",
        "original query",
        "original query",
    ]
