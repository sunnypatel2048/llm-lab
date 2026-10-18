"""Loads models.toml into ModelSpec objects that the client looks up by alias.

Validation happens in two stages:
  1. Structure, at load time: unknown provider, missing field, bad value -> ConfigError now.
  2. API keys, via check_keys(): only for the aliases a run will actually use. This keeps tests
     and CI key-free, and an Ollama-only run doesn't demand a Gemini key.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm_lab.types import ConfigError

OLLAMA_PLACEHOLDER_KEY = "ollama"  # the SDK requires a key; Ollama ignores it
_REQUIRED_MODEL_FIELDS = ("provider", "model_id", "price_in_per_m", "price_out_per_m")


@dataclass(frozen=True, slots=True)
class ModelSpec:
    alias: str
    provider: str
    model_id: str
    base_url: str
    api_key_env: str | None  # None = no key needed (local)
    price_in_per_m: float
    price_out_per_m: float
    rpm: int | None = None  # None = no per-minute limit
    rpd: int | None = None  # None = no per-day limit

    def api_key(self) -> str:
        if self.api_key_env is None:
            return OLLAMA_PLACEHOLDER_KEY
        key = os.environ.get(self.api_key_env, "").strip()
        if not key:
            raise ConfigError(
                f"Model '{self.alias}' needs {self.api_key_env}, which is not set. "
                f"Add it to .env (see .env.example)."
            )
        return key

    def cost(self, input_tokens: int | None, output_tokens: int | None) -> float | None:
        """USD at paid rates. None if either count is missing: we report unknowns, not guesses."""
        if input_tokens is None or output_tokens is None:
            return None
        return (
            input_tokens * self.price_in_per_m + output_tokens * self.price_out_per_m
        ) / 1_000_000


class Registry:
    def __init__(self, specs: dict[str, ModelSpec]):
        self._specs = specs

    @classmethod
    def load(cls, path: str | Path = "models.toml") -> Registry:
        path = Path(path)
        if not path.is_file():
            raise ConfigError(f"Model registry not found at {path.resolve()}")
        try:
            data = tomllib.loads(path.read_text())
        except tomllib.TOMLDecodeError as e:
            raise ConfigError(f"{path} is not valid TOML: {e}") from e
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Registry:
        providers = data.get("providers") or {}
        models = data.get("models") or {}
        if not models:
            raise ConfigError("Registry defines no models (expected [models.<alias>] tables).")

        specs: dict[str, ModelSpec] = {}
        for alias, m in models.items():
            missing = [f for f in _REQUIRED_MODEL_FIELDS if f not in m]
            if missing:
                raise ConfigError(f"Model '{alias}' is missing: {', '.join(missing)}")

            provider_name = m["provider"]
            provider = providers.get(provider_name)
            if provider is None:
                raise ConfigError(
                    f"Model '{alias}' uses unknown provider '{provider_name}'. "
                    f"Known providers: {', '.join(sorted(providers)) or 'none'}"
                )
            if "base_url" not in provider:
                raise ConfigError(f"Provider '{provider_name}' is missing base_url")

            price_in, price_out = m["price_in_per_m"], m["price_out_per_m"]
            for name, value in (("price_in_per_m", price_in), ("price_out_per_m", price_out)):
                if not isinstance(value, int | float) or value < 0:
                    raise ConfigError(f"Model '{alias}': {name} must be a number >= 0")

            for name in ("rpm", "rpd"):
                value = m.get(name)
                if value is not None and (not isinstance(value, int) or value <= 0):
                    raise ConfigError(f"Model '{alias}': {name} must be a positive integer")

            specs[alias] = ModelSpec(
                alias=alias,
                provider=provider_name,
                model_id=m["model_id"],
                base_url=provider["base_url"],
                api_key_env=provider.get("api_key_env"),
                price_in_per_m=float(price_in),
                price_out_per_m=float(price_out),
                rpm=m.get("rpm"),
                rpd=m.get("rpd"),
            )
        return cls(specs)

    def get(self, alias: str) -> ModelSpec:
        try:
            return self._specs[alias]
        except KeyError:
            raise ConfigError(
                f"Unknown model alias '{alias}'. Known: {', '.join(self.aliases())}"
            ) from None

    def aliases(self) -> list[str]:
        return sorted(self._specs)

    def check_keys(self, aliases: list[str] | None = None) -> None:
        """Fail fast before any work starts if a needed API key is missing."""
        for alias in aliases if aliases is not None else self.aliases():
            self.get(alias).api_key()
