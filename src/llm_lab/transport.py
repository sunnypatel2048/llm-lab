"""The only module that imports the `openai` SDK.

Gemini, Groq and Ollama all expose an OpenAI-compatible Chat Completions endpoint, so one SDK
reaches all three by changing base_url. This module makes the HTTP call, reads the answer and
token usage, and translates SDK exceptions into llm_lab error types. Everything else in the
package depends on llm_lab types only, so swapping SDKs (or adding a native Gemini adapter)
touches this file alone.
"""

from __future__ import annotations

import json
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
    ToolCall,
)


@dataclass(frozen=True, slots=True)
class RawCompletion:
    """What one successful HTTP call returned, before the client adds timing and cost."""

    text: str
    input_tokens: int | None
    output_tokens: int | None
    reasoning_tokens: int | None
    finish_reason: str | None
    tool_calls: tuple[ToolCall, ...] = ()
    message: Message | None = None  # the assistant turn to send back next time


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
        tools: list[dict[str, Any]] | None = None,
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
        tools: list[dict[str, Any]] | None = None,
    ) -> RawCompletion:
        client = self._client_for(spec)
        params: dict[str, Any] = dict(
            model=spec.model_id,
            messages=messages,
            temperature=temperature,
            # max_tokens rather than the newer max_completion_tokens: it's the one all
            # three compatible APIs (Gemini, Groq, Ollama) accept.
            max_tokens=max_tokens,
            timeout=timeout,
        )
        if spec.reasoning_effort is not None:
            params["reasoning_effort"] = spec.reasoning_effort
        if tools:
            # No tool_choice: the default ("auto") is the only value all three providers accept
            # (Ollama doesn't support tool_choice at all).
            params["tools"] = tools
        try:
            completion = client.chat.completions.create(**params)
        except openai.OpenAIError as e:
            raise translate_error(e, spec, timeout=timeout) from e
        return parse_completion(completion)


def parse_completion(completion: Any) -> RawCompletion:
    choice = completion.choices[0] if completion.choices else None
    text = (choice.message.content or "") if choice is not None else ""
    usage = completion.usage
    details = getattr(usage, "completion_tokens_details", None) if usage is not None else None

    sdk_calls = (getattr(choice.message, "tool_calls", None) or []) if choice is not None else []
    raw_calls = [_to_plain(c) for c in sdk_calls]
    call_dicts = [_normalize_call(c, i) for i, c in enumerate(raw_calls)]
    tool_calls = tuple(
        ToolCall(id=c["id"], name=c["function"]["name"], arguments=c["function"]["arguments"])
        for c in call_dicts
    )

    # The turn we'll send back next time. Message level: only the standard fields, because some
    # providers reject fields they don't recognise in requests (e.g. a reasoning field another
    # provider added). Tool-call level: everything the provider sent, because that's where
    # provider-specific continuity data (such as Gemini's thinking signatures) lives.
    message: Message = {"role": "assistant", "content": text}
    if call_dicts:
        message["tool_calls"] = call_dicts

    return RawCompletion(
        text=text,
        input_tokens=usage.prompt_tokens if usage is not None else None,
        output_tokens=usage.completion_tokens if usage is not None else None,
        reasoning_tokens=getattr(details, "reasoning_tokens", None),
        finish_reason=getattr(choice, "finish_reason", None),
        tool_calls=tool_calls,
        message=message,
    )


def _to_plain(obj: Any) -> Any:
    """SDK objects (pydantic models, or SimpleNamespace in tests) -> plain JSON-able data,
    dropping None values. Extra fields the SDK doesn't know about are kept."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump(exclude_none=True)
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list | tuple):
        return [_to_plain(v) for v in obj]
    if hasattr(obj, "__dict__"):
        return {k: _to_plain(v) for k, v in vars(obj).items() if v is not None}
    return obj


def _normalize_call(call: dict[str, Any], index: int) -> dict[str, Any]:
    """Fill the gaps some providers leave, so every call has an id, a type and string arguments."""
    function = dict(call.get("function") or {})
    arguments = function.get("arguments")
    if arguments is None:
        arguments = "{}"
    elif not isinstance(arguments, str):  # some servers send a parsed object instead of text
        arguments = json.dumps(arguments)
    function["arguments"] = arguments
    function.setdefault("name", "")
    return {
        **call,
        "id": call.get("id") or f"call_{index}",
        "type": "function",
        "function": function,
    }


def translate_error(
    e: openai.OpenAIError, spec: ModelSpec, *, timeout: float | None = None
) -> LLMError:
    """Map an SDK exception to an llm_lab error. ORDER MATTERS: APITimeoutError is a subclass
    of APIConnectionError, so it must be checked first or every timeout would be reported as
    "can't connect" and never retried."""
    if isinstance(e, openai.APITimeoutError):
        waited = f" within {timeout:g}s" if timeout is not None else ""
        hint = (
            " Local models can be slow, especially while thinking: raise timeout_s for this"
            " model in models.toml, or lower max_tokens."
            if spec.provider == "ollama"
            else ""
        )
        return LLMTimeout(f"{spec.alias}: no response{waited}.{hint}")

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
