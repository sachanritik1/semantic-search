import json

import pytest

from app.utils.llm_pricing import (
    _lookup_rates,
    _normalize_usage,
    estimate_cost,
    register_pricing,
)


def test_normalize_usage_handles_openai_keys():
    n = _normalize_usage(
        {"input_tokens": 1000, "output_tokens": 200, "cached_tokens": 800}
    )
    assert n == {
        "prompt_tokens": 1000,
        "completion_tokens": 200,
        "cached_tokens": 800,
        "cache_write_tokens": 0,
    }


def test_normalize_usage_handles_gemini_keys():
    n = _normalize_usage(
        {
            "prompt_tokens": 500,
            "completion_tokens": 100,
            "cached_content_token_count": 300,
        }
    )
    assert n["prompt_tokens"] == 500
    assert n["cached_tokens"] == 300


def test_normalize_usage_handles_anthropic_keys():
    n = _normalize_usage(
        {
            "prompt_tokens": 1000,
            "completion_tokens": 50,
            "cache_read_input_tokens": 800,
            "cache_creation_input_tokens": 200,
        }
    )
    assert n["cached_tokens"] == 800
    assert n["cache_write_tokens"] == 200


def test_estimate_cost_returns_none_for_unknown_model():
    assert estimate_cost("totally-made-up-model", {"prompt_tokens": 100}) is None


def test_estimate_cost_returns_none_for_missing_usage():
    assert estimate_cost("gpt-4o", None) is None
    assert estimate_cost("gpt-4o", {}) is None  # empty usage is treated as missing


def test_estimate_cost_handles_partial_usage():
    cost = estimate_cost("gpt-4o", {"input_tokens": 1000})
    assert cost is not None
    assert cost["input_cost"] == pytest.approx(0.0025, rel=1e-6)
    assert cost["output_cost"] == 0.0


def test_estimate_cost_charges_fresh_input_minus_cached():
    # gpt-4o: input=$2.50/M, cached_input=$1.25/M, output=$10/M
    usage = {
        "input_tokens": 10_000,
        "output_tokens": 1_000,
        "cached_tokens": 8_000,
    }
    cost = estimate_cost("gpt-4o", usage)
    assert cost is not None
    # fresh input = 2000 tokens @ $2.50/M = $0.005
    assert cost["input_cost"] == pytest.approx(0.005, rel=1e-6)
    # cached = 8000 tokens @ $1.25/M = $0.010
    assert cost["cached_input_cost"] == pytest.approx(0.010, rel=1e-6)
    # output = 1000 tokens @ $10/M = $0.010
    assert cost["output_cost"] == pytest.approx(0.010, rel=1e-6)
    assert cost["total_cost"] == pytest.approx(0.025, rel=1e-6)
    # savings = 8000 * ($2.50 - $1.25) / 1M = $0.010
    assert cost["savings_from_cache"] == pytest.approx(0.010, rel=1e-6)
    assert cost["currency"] == "USD"


def test_estimate_cost_resolves_openrouter_prefix():
    cost_a = estimate_cost(
        "openai/gpt-4o", {"input_tokens": 1000, "output_tokens": 1000}
    )
    cost_b = estimate_cost("gpt-4o", {"input_tokens": 1000, "output_tokens": 1000})
    assert cost_a == cost_b


def test_estimate_cost_handles_anthropic_cache_write():
    # claude-3.5-sonnet: input=$3, cached=$0.30, cache_write=$3.75, output=$15
    usage = {
        "prompt_tokens": 1_000,
        "completion_tokens": 500,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 1_000,
    }
    cost = estimate_cost("anthropic/claude-3.5-sonnet", usage)
    assert cost is not None
    # Note: cache_write tokens are billed separately; prompt_tokens here is the
    # non-cache portion. Fresh input = 1000 - 0 cached = 1000 @ $3/M = $0.003.
    assert cost["input_cost"] == pytest.approx(0.003, rel=1e-6)
    # Cache write = 1000 @ $3.75/M = $0.00375
    assert cost["cache_write_cost"] == pytest.approx(0.00375, rel=1e-6)
    # Output = 500 @ $15/M = $0.0075
    assert cost["output_cost"] == pytest.approx(0.0075, rel=1e-6)


def test_register_pricing_overrides_existing():
    register_pricing("custom-test-model", input=1.0, output=2.0, cached_input=0.5)
    cost = estimate_cost(
        "custom-test-model",
        {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000},
    )
    assert cost is not None
    assert cost["input_cost"] == pytest.approx(1.0)
    assert cost["output_cost"] == pytest.approx(2.0)


def test_pricing_overrides_loaded_from_env(tmp_path, monkeypatch):
    override_path = tmp_path / "pricing.json"
    override_path.write_text(
        json.dumps(
            {"env-loaded-model": {"input": 0.5, "output": 1.5, "cached_input": 0.1}}
        )
    )
    monkeypatch.setenv("LLM_PRICING_JSON_PATH", str(override_path))
    # Force re-load by toggling the loaded flag.
    from app.utils import llm_pricing

    monkeypatch.setattr(llm_pricing, "_OVERRIDES_LOADED", False)

    cost = estimate_cost(
        "env-loaded-model",
        {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000},
    )
    assert cost is not None
    assert cost["input_cost"] == pytest.approx(0.5)
    assert cost["output_cost"] == pytest.approx(1.5)


def test_lookup_rates_resolves_versioned_model_suffix():
    # "gpt-4o-2024-11-20" should fall back to "gpt-4o" pricing.
    rates = _lookup_rates("gpt-4o-2024-11-20")
    assert rates is not None
    assert rates["input"] == 2.50
