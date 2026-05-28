import logging
from unittest.mock import MagicMock

from app.adapters.llm.base import LLMResponse
from app.adapters.llm.decorators import CostTrackingLLM
from app.infrastructure.utils.llm_usage import annotate_cost, log_llm_usage


def test_annotate_cost_adds_cost_to_usage():
    usage = {"input_tokens": 1000, "output_tokens": 1000, "cached_tokens": 500}
    annotate_cost(usage, model="gpt-4o")
    assert "cost" in usage
    assert usage["cost"]["currency"] == "USD"
    assert usage["cost"]["total_cost"] > 0


def test_annotate_cost_no_op_for_unknown_model():
    usage = {"input_tokens": 100, "output_tokens": 100}
    annotate_cost(usage, model="totally-made-up")
    assert "cost" not in usage


def test_annotate_cost_handles_none_usage():
    assert annotate_cost(None, model="gpt-4o") is None


def test_log_llm_usage_handles_provider_numeric_cost(caplog):
    usage = {
        "input_tokens": 100,
        "output_tokens": 50,
        "cost": 0.00123,
    }

    with caplog.at_level(logging.INFO, logger="app.infrastructure.utils.llm_usage"):
        log_llm_usage(usage, context="generate", model="unknown/model")

    messages = [r.getMessage() for r in caplog.records]
    assert any("provider-reported" in m and "0.001230" in m for m in messages)


def test_annotate_cost_preserves_provider_numeric_cost():
    usage = {"input_tokens": 100, "output_tokens": 50, "cost": 0.00123}
    annotate_cost(usage, model="gpt-4o")
    assert usage["cost"] == 0.00123
    assert "estimated_cost" in usage
    assert usage["estimated_cost"]["currency"] == "USD"


def test_log_llm_usage_emits_cost(caplog):
    usage = {
        "input_tokens": 1000,
        "output_tokens": 500,
        "cached_tokens": 800,
    }
    annotate_cost(usage, model="gpt-4o")

    with caplog.at_level(logging.INFO, logger="app.infrastructure.utils.llm_usage"):
        log_llm_usage(usage, context="generate", model="gpt-4o")

    messages = [r.getMessage() for r in caplog.records]
    assert any("LLM cost generate" in m and "model=gpt-4o" in m for m in messages)
    assert any("cached_tokens=800" in m for m in messages)


def test_cost_tracking_llm_attaches_cost_to_response_usage():
    mock_llm = MagicMock()
    mock_llm.generate.return_value = LLMResponse(
        content="hi",
        model="gpt-4o",
        usage={"input_tokens": 1000, "output_tokens": 500, "cached_tokens": 800},
    )

    llm = CostTrackingLLM(mock_llm)
    response = llm.generate("prompt")

    assert response.usage is not None
    assert "cost" in response.usage
    assert response.usage["cost"]["total_cost"] > 0
