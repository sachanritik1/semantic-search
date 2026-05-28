from unittest.mock import MagicMock, patch

from langchain_core.messages import HumanMessage, SystemMessage

from app.adapters.llm.gemini import GeminiLLM
from app.adapters.llm.openai import OpenaiLLM
from app.adapters.llm.openrouter import OpenRouterLLM, _build_messages
from app.infrastructure.utils.prompt_cache import cache_key


def test_cache_key_includes_version():
    with patch("app.utils.prompt_cache.settings") as mock_settings:
        mock_settings.PROMPT_CACHE_ENABLED = True
        mock_settings.PROMPT_CACHE_VERSION = "v1"
        assert cache_key("ask") == "ask:v1"


def test_cache_key_disabled_returns_none():
    with patch("app.utils.prompt_cache.settings") as mock_settings:
        mock_settings.PROMPT_CACHE_ENABLED = False
        mock_settings.PROMPT_CACHE_VERSION = "v1"
        assert cache_key("ask") is None


def test_openai_adapter_forwards_system_role_and_cache_key():
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.output_text = "answer"
    mock_response.usage = None
    mock_client.responses.create.return_value = mock_response

    llm = OpenaiLLM(api_key="test", model="gpt-4o")
    llm.client = mock_client

    with patch("app.llm.openai_llm.settings") as mock_settings:
        mock_settings.PROMPT_CACHE_ENABLED = True
        llm.generate(
            "user body",
            system_prompt="system rules",
            cache_key="ask:v1",
        )

    kwargs = mock_client.responses.create.call_args.kwargs
    assert kwargs["prompt_cache_key"] == "ask:v1"
    assert kwargs["input"] == [
        {"role": "system", "content": "system rules"},
        {"role": "user", "content": "user body"},
    ]


def test_openai_adapter_omits_cache_key_when_disabled():
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.output_text = "answer"
    mock_response.usage = None
    mock_client.responses.create.return_value = mock_response

    llm = OpenaiLLM(api_key="test", model="gpt-4o")
    llm.client = mock_client

    with patch("app.llm.openai_llm.settings") as mock_settings:
        mock_settings.PROMPT_CACHE_ENABLED = False
        llm.generate("user body", system_prompt="system rules", cache_key="ask:v1")

    kwargs = mock_client.responses.create.call_args.kwargs
    assert "prompt_cache_key" not in kwargs


def test_gemini_adapter_forwards_system_instruction():
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "answer"
    mock_response.usage_metadata = None
    mock_client.models.generate_content.return_value = mock_response

    llm = GeminiLLM(api_key="test", model="gemini-2.5-flash")
    llm.client = mock_client

    llm.generate("user body", system_prompt="system rules")

    kwargs = mock_client.models.generate_content.call_args.kwargs
    config = kwargs["config"]
    assert config.system_instruction == "system rules"
    assert kwargs["contents"] == "user body"


def test_openrouter_build_messages_adds_cache_control_when_enabled():
    with patch("app.llm.openrouter_llm.settings") as mock_settings:
        mock_settings.PROMPT_CACHE_ENABLED = True
        messages = _build_messages("user body", "system rules")

    assert len(messages) == 2
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)
    assert messages[0].content == [
        {
            "type": "text",
            "text": "system rules",
            "cache_control": {"type": "ephemeral"},
        }
    ]
    assert messages[1].content == "user body"


def test_openrouter_adapter_invokes_with_messages():
    mock_bound = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "answer"
    mock_response.response_metadata = {}
    mock_bound.invoke.return_value = mock_response

    mock_client = MagicMock()
    mock_client.bind.return_value = mock_bound

    llm = OpenRouterLLM(api_key="test", model="anthropic/claude-3.5-sonnet")
    llm.client = mock_client

    with patch("app.llm.openrouter_llm.settings") as mock_settings:
        mock_settings.PROMPT_CACHE_ENABLED = True
        llm.generate("user body", system_prompt="system rules")

    messages = mock_bound.invoke.call_args.args[0]
    assert isinstance(messages[0], SystemMessage)
    assert messages[0].content[0]["cache_control"] == {"type": "ephemeral"}
