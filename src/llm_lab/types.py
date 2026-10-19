"""Shared types: the response every model call returns, and the errors callers can catch.

Callers never see raw `openai` SDK exceptions; the client translates them into these.
Each error says whether retrying could help (`retryable`), so retry logic reads one
attribute instead of matching exception types.
"""

from __future__ import annotations

from dataclasses import dataclass

# A chat message, e.g. {"role": "user", "content": "Hi"}. Kept as a plain dict so it
# passes straight through to the OpenAI-compatible API without conversion.
Message = dict[str, str]


@dataclass(frozen=True, slots=True)
class LLMResponse:
    text: str
    alias: str  # our name for the model, e.g. "gemini-flash"
    model: str  # the provider's model ID, e.g. "gemini-3.8-flash"
    provider: str  # "gemini" | "groq" | "ollama"
    input_tokens: int | None  # None = provider didn't report usage (never estimated)
    output_tokens: int | None
    reasoning_tokens: int | None  # hidden "thinking" tokens, billed as output; None if not reported
    latency_ms: float  # wall time of the successful attempt
    cost_usd: float | None  # at paid rates; None when token counts are missing
    cached: bool = False  # True = served from the local cache, no API call made
    attempts: int = 1  # 1 = succeeded first try; >1 = retries happened
    # Why generation stopped: "stop" = finished; "length" = hit max_tokens (answer is cut off,
    # common with thinking models whose reasoning uses up the budget); None = not reported.
    finish_reason: str | None = None


class LLMError(Exception):
    """Base class for every error the client raises."""

    retryable: bool = False


class ConfigError(LLMError):
    """models.toml is malformed, an alias is unknown, or an API key is missing."""


class ProviderUnavailable(LLMError):
    """Couldn't connect at all, e.g. Ollama isn't running."""


class RateLimited(LLMError):
    """HTTP 429 from the provider. Waiting and retrying should succeed."""

    retryable = True

    def __init__(self, message: str, *, retry_after: float | None = None):
        super().__init__(message)
        # Seconds the provider asked us to wait (Retry-After header), if it said.
        self.retry_after = retry_after


class ProviderError(LLMError):
    """HTTP 5xx from the provider. Usually transient."""

    retryable = True


class LLMTimeout(LLMError):
    """The request took longer than the timeout."""

    retryable = True


class DailyQuotaExhausted(LLMError):
    """Our local counter says today's request limit is used up. Retrying today won't help."""


class BadRequest(LLMError):
    """Any other 4xx (bad model ID, invalid parameters, auth failure). Retrying won't help."""
