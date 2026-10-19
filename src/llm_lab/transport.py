"""The only module that imports the `openai` SDK.

Gemini, Groq and Ollama all expose an OpenAI-compatible Chat Completions endpoint, so one SDK
reaches all three by changing base_url. This module makes the HTTP call, reads the answer and
token usage, and translates SDK exceptions into llm_lab error types. Everything else in the
package depends on llm_lab types only, so swapping SDKs (or adding a native Gemini adapter)
touches this file alone.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import openai

from llm_lab.registry import ModelSpec
from llm_lab.types import (
    BadRequest,
    DailyQuotaExhausted,
    LLMError,
    LLMTimeout,
    Message,
    ProviderError,
    ProviderUnavailable,
    RateLimited,
)


@dataclass(frozen=True, slots=True)
class RawCompletion:
    """What one successful HTTP call returned, before the client adds timing and cost."""

    text: str
    input_tokens: int | None
    output_tokens: int | None
    reasoning_tokens: int | None
    finish_reason: str | None


class Transport(Protocol):
    """Anything that can make one chat call.

    Tests pass a fake; real runs use OpenAICompatTransport.
    """

    def __call__(
        self,
        spec: ModelSpec,
        messages: list[Message],
        *,
        temperature: float,
        max_tokens: int,
        timeout: float,
    ) -> RawCompletion: ...


def make_openai_client(spec: ModelSpec) -> openai.OpenAI:
    # max_retries=0: the SDK retries 429s and 5xx on its own by default. Our client already
    # retries (and counts attempts against rate limits), so SDK retries underneath would
    # multiply the attempts and hide them from our numbers.
    return openai.OpenAI(base_url=spec.base_url, api_key=spec.api_key(), max_retries=0)


class OpenAICompatTransport:
    def __init__(self, client_factory: Callable[[ModelSpec], Any] = make_openai_client):
        self._factory = client_factory
        self._clients: dict[str, Any] = {}  # one SDK client (and connection pool) per provider
        self._lock = threading.Lock()

    def _client_for(self, spec: ModelSpec) -> Any:
        with self._lock:
            client = self._clients.get(spec.provider)
            if client is None:
                client = self._factory(spec)
                self._clients[spec.provider] = client
            return client

    def __call__(
        self,
        spec: ModelSpec,
        messages: list[Message],
        *,
        temperature: float,
        max_tokens: int,
        timeout: float,
    ) -> RawCompletion:
        client = self._client_for(spec)
        try:
            completion = client.chat.completions.create(
                model=spec.model_id,
                messages=messages,
                temperature=temperature,
                # max_tokens rather than the newer max_completion_tokens: it's the one all
                # three compatible APIs (Gemini, Groq, Ollama) accept.
                max_tokens=max_tokens,
                timeout=timeout,
            )
        except openai.OpenAIError as e:
            raise translate_error(e, spec) from e
        return parse_completion(completion)


def parse_completion(completion: Any) -> RawCompletion:
    choice = completion.choices[0] if completion.choices else None
    text = (choice.message.content or "") if choice is not None else ""
    usage = completion.usage
    details = getattr(usage, "completion_tokens_details", None) if usage is not None else None
    return RawCompletion(
        text=text,
        input_tokens=usage.prompt_tokens if usage is not None else None,
        output_tokens=usage.completion_tokens if usage is not None else None,
        reasoning_tokens=getattr(details, "reasoning_tokens", None),
        finish_reason=getattr(choice, "finish_reason", None),
    )


def translate_error(e: openai.OpenAIError, spec: ModelSpec) -> LLMError:
    """Map an SDK exception to an llm_lab error. ORDER MATTERS: APITimeoutError is a subclass
    of APIConnectionError, so it must be checked first or every timeout would be reported as
    "can't connect" and never retried."""
    if isinstance(e, openai.APITimeoutError):
        return LLMTimeout(f"{spec.alias}: request timed out")

    if isinstance(e, openai.APIConnectionError):
        hint = (
            "Is Ollama running? Start it with `ollama serve`."
            if spec.provider == "ollama"
            else "Check your network connection."
        )
        return ProviderUnavailable(f"{spec.alias}: couldn't connect to {spec.base_url}. {hint}")

    if isinstance(e, openai.APIStatusError):
        status = e.status_code
        detail = _short(e.message)
        message = f"{spec.alias}: HTTP {status}: {detail}"
        if status == 429:
            # A 429 can mean "slow down" (retry helps) or "out of quota for today" (it won't).
            # Providers word the daily case differently; this catches the common phrasings.
            lowered = detail.lower().replace("_", " ")
            if "per day" in lowered or "perday" in lowered or "daily" in lowered:
                return DailyQuotaExhausted(message)
            return RateLimited(message, retry_after=_retry_after_seconds(e.response))
        if status >= 500:
            return ProviderError(message)
        return BadRequest(message)

    return LLMError(f"{spec.alias}: {e}")


def _retry_after_seconds(response: Any) -> float | None:
    """Seconds from a Retry-After header, if present as a number (HTTP-date form is ignored)."""
    value = response.headers.get("retry-after") if response is not None else None
    try:
        seconds = float(value) if value is not None else None
    except ValueError:
        return None
    return seconds if seconds is not None and seconds >= 0 else None


def _short(text: str, limit: int = 300) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"
