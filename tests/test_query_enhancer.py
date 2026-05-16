from app.services.query_enhancer import QueryEnhancer
from app.llm.base import LLMResponse


class DummyLLMService:
    def __init__(self):
        self.called = False

    def generate_text(self, prompt: str, *, temperature=0.0, max_tokens=None, model=None):
        self.called = True
        return LLMResponse(content="enhanced query")


def test_enhancer_calls_llm_and_returns_enhanced_query():
    dummy = DummyLLMService()
    enhancer = QueryEnhancer(dummy)
    enhanced = enhancer.enhance("original query")
    assert dummy.called is True
    assert enhanced == "enhanced query"


def test_enhancer_falls_back_to_original_when_llm_returns_empty():
    class EmptyLLMService:
        def generate_text(self, prompt: str, *, temperature=0.0, max_tokens=None, model=None):
            return LLMResponse(content="")

    enhancer = QueryEnhancer(EmptyLLMService())
    assert enhancer.enhance("original query") == ""
