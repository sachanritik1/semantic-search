from unittest.mock import MagicMock

from app.llm.openrouter_llm import OpenRouterLLM


def _make_llm() -> OpenRouterLLM:
    llm = OpenRouterLLM.__new__(OpenRouterLLM)
    llm.model = "test-model"
    llm.client = MagicMock()
    return llm


def test_stream_generate_preserves_inter_token_spaces():
    llm = _make_llm()
    bound = MagicMock()
    bound.stream.return_value = [
        MagicMock(content="Hello"),
        MagicMock(content=" world"),
        MagicMock(content=" from"),
        MagicMock(content=" stream"),
    ]
    llm.client.bind.return_value = bound

    chunks = list(llm.stream_generate("prompt"))

    assert chunks == ["Hello", " world", " from", " stream"]
    assert "".join(chunks) == "Hello world from stream"


def test_stream_generate_skips_empty_chunks_but_keeps_whitespace_only():
    llm = _make_llm()
    bound = MagicMock()
    bound.stream.return_value = [
        MagicMock(content="a"),
        MagicMock(content=""),
        MagicMock(content=" "),
        MagicMock(content="b"),
    ]
    llm.client.bind.return_value = bound

    chunks = list(llm.stream_generate("prompt"))

    assert chunks == ["a", " ", "b"]


def test_stream_generate_handles_list_content_blocks():
    llm = _make_llm()
    bound = MagicMock()
    bound.stream.return_value = [
        MagicMock(content=[{"type": "text", "text": "Hi"}]),
        MagicMock(content=[{"type": "text", "text": " there"}]),
    ]
    llm.client.bind.return_value = bound

    chunks = list(llm.stream_generate("prompt"))

    assert chunks == ["Hi", " there"]
