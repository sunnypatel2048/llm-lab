from pathlib import Path

import pytest

from llm_lab.registry import OLLAMA_PLACEHOLDER_KEY, Registry
from llm_lab.types import ConfigError

ROOT = Path(__file__).resolve().parent.parent


def minimal(**model_overrides):
    """A valid one-model registry dict; override fields to break it in specific ways."""
    model = {
        "provider": "gemini",
        "model_id": "gemini-x",
        "price_in_per_m": 1.0,
        "price_out_per_m": 2.0,
        "rpm": 5,
        "rpd": 20,
    } | model_overrides
    return {
        "providers": {"gemini": {"base_url": "https://example.test/", "api_key_env": "TEST_KEY"}},
        "models": {"m": {k: v for k, v in model.items() if v is not None}},
    }


# --- the real models.toml -------------------------------------------------------------


def test_real_registry_loads_all_aliases():
    reg = Registry.load(ROOT / "models.toml")
    assert reg.aliases() == ["gemini-flash", "gemini-lite", "groq-oss", "local-qwen"]


def test_real_registry_merges_provider_settings():
    spec = Registry.load(ROOT / "models.toml").get("gemini-flash")
    assert spec.model_id == "gemini-3.8-flash"
    assert spec.base_url.startswith("https://generativelanguage.googleapis.com")
    assert spec.api_key_env == "GEMINI_API_KEY"
    assert (spec.rpm, spec.rpd) == (5, 20)


def test_local_model_has_no_limits_and_needs_no_key():
    spec = Registry.load(ROOT / "models.toml").get("local-qwen")
    assert spec.rpm is None and spec.rpd is None
    assert spec.api_key() == OLLAMA_PLACEHOLDER_KEY


# --- cost -----------------------------------------------------------------------------


def test_cost_uses_per_million_prices():
    spec = Registry.from_dict(minimal(price_in_per_m=0.75, price_out_per_m=3.75)).get("m")
    # 1000 * 0.75/1M + 500 * 3.75/1M = 0.00075 + 0.001875
    assert spec.cost(1000, 500) == pytest.approx(0.002625)


def test_cost_is_none_when_usage_missing():
    spec = Registry.from_dict(minimal()).get("m")
    assert spec.cost(None, 10) is None
    assert spec.cost(10, None) is None


# --- validation -----------------------------------------------------------------------


def test_unknown_alias_lists_known_ones():
    with pytest.raises(ConfigError, match="Known: m"):
        Registry.from_dict(minimal()).get("nope")


def test_unknown_provider_rejected():
    with pytest.raises(ConfigError, match="unknown provider 'openai'"):
        Registry.from_dict(minimal(provider="openai"))


def test_missing_field_rejected():
    with pytest.raises(ConfigError, match="missing: model_id"):
        Registry.from_dict(minimal(model_id=None))


@pytest.mark.parametrize("bad", [0, -1, 2.5, "5"])
def test_bad_rate_limit_rejected(bad):
    with pytest.raises(ConfigError, match="rpm must be a positive integer"):
        Registry.from_dict(minimal(rpm=bad))


def test_negative_price_rejected():
    with pytest.raises(ConfigError, match="price_in_per_m"):
        Registry.from_dict(minimal(price_in_per_m=-1))


def test_missing_file_is_a_config_error(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        Registry.load(tmp_path / "missing.toml")


def test_invalid_toml_is_a_config_error(tmp_path):
    bad = tmp_path / "models.toml"
    bad.write_text("[models.m\n")
    with pytest.raises(ConfigError, match="not valid TOML"):
        Registry.load(bad)


# --- API keys -------------------------------------------------------------------------


def test_missing_key_names_the_env_var(monkeypatch):
    monkeypatch.delenv("TEST_KEY", raising=False)
    with pytest.raises(ConfigError, match="needs TEST_KEY"):
        Registry.from_dict(minimal()).check_keys()


def test_blank_key_counts_as_missing(monkeypatch):
    monkeypatch.setenv("TEST_KEY", "   ")
    with pytest.raises(ConfigError):
        Registry.from_dict(minimal()).get("m").api_key()


def test_key_is_read_from_env(monkeypatch):
    monkeypatch.setenv("TEST_KEY", "abc123")
    assert Registry.from_dict(minimal()).get("m").api_key() == "abc123"


def test_check_keys_only_checks_requested_aliases(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    reg = Registry.load(ROOT / "models.toml")
    reg.check_keys(["local-qwen"])  # must not demand the Gemini key
