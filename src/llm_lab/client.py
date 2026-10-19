"""LLMClient: one call shape for every model, with caching, rate limits, retries and cost.

    client = LLMClient.from_env()
    resp = client.chat("gemini-lite", [{"role": "user", "content": "Hi"}])

What chat() does, in order:
  1. Look up the alias; check its API key (fails before any work if it's missing).
  2. Check the cache. A hit returns immediately and uses no quota.
  3. For each attempt: take a rate-limit slot (every attempt is a real request), call the
     transport, time the call.
  4. Retry only errors marked retryable, with exponential backoff + jitter, waiting at least
     as long as the provider's Retry-After when it sends one.
  5. Compute cost, store in the cache, log one summary line, return.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path

from dotenv import load_dotenv
from tenacity import (
    RetryCallState,
    Retrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from llm_lab.cache import CACHE_MODES, CacheMode, ResponseCache, cache_key
from llm_lab.ratelimit import RateLimiter
from llm_lab.registry import ModelSpec, Registry
from llm_lab.types import LLMError, LLMResponse, LLMTimeout, Message

log = logging.getLogger(__name__)


def _should_retry(spec: ModelSpec) -> Callable[[BaseException], bool]:
    def predicate(e: BaseException) -> bool:
        if not (isinstance(e, LLMError) and e.retryable):
            return False
        # A timeout on a local model isn't transient: the same generation will be just as slow.
        return spec.retry_on_timeout or not isinstance(e, LLMTimeout)

    return predicate


class LLMClient:
    def __init__(
        self,
        registry: Registry,
        *,
        transport=None,
        limiter: RateLimiter | None = None,
        cache: ResponseCache | None = None,
        timeout: float = 60.0,
        max_attempts: int = 4,
        max_wait: float = 60.0,
        sleep: Callable[[float], None] = time.sleep,
    ):
        """Low-level constructor. limiter=None / cache=None mean "disabled", which tests use;
        normal code should call LLMClient.from_env(), which turns both on.

        timeout: default seconds per attempt, used when neither the call nor the model's
            timeout_s in models.toml sets one.
        max_wait: cap on any single backoff wait, including a provider's Retry-After.
        """
        if transport is None:
            from llm_lab.transport import OpenAICompatTransport  # keeps `openai` import lazy

            transport = OpenAICompatTransport()
        self.registry = registry
        self.timeout = timeout
        self.max_attempts = max_attempts
        self.max_wait = max_wait
        self._transport = transport
        self._limiter = limiter
        self._cache = cache
        self._sleep = sleep
        self._backoff = wait_exponential_jitter(max=30, jitter=1)

    @classmethod
    def from_env(
        cls,
        models_path: str | Path = "models.toml",
        env_path: str | Path = ".env",
        **kwargs,
    ) -> LLMClient:
        """Load .env (never overriding variables already set) and models.toml, with the default
        on-disk cache and rate limiter enabled."""
        load_dotenv(env_path, override=False)
        kwargs.setdefault("limiter", RateLimiter())
        kwargs.setdefault("cache", ResponseCache())
        return cls(Registry.load(models_path), **kwargs)

    def chat(
        self,
        alias: str,
        messages: list[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        cache: CacheMode = "use",
        timeout: float | None = None,
    ) -> LLMResponse:
        if cache not in CACHE_MODES:
            raise ValueError(f"cache must be one of {CACHE_MODES}, got {cache!r}")
        spec = self.registry.get(alias)
        spec.api_key()  # fail fast on a missing key, before cache lookups or waiting

        key = None
        if self._cache is not None:
            # reasoning_effort changes the answer, so it's part of the key. Only added when set,
            # so models without it keep the same keys as before.
            extra = {"reasoning_effort": spec.reasoning_effort} if spec.reasoning_effort else None
            key = cache_key(
                spec, messages, temperature=temperature, max_tokens=max_tokens, extra=extra
            )
            hit = self._cache.get(key, cache)
            if hit is not None:
                log.info("%s cache hit", alias)
                return hit

        # Most specific wins: this call, then the model's timeout_s, then the client default.
        if timeout is not None:
            call_timeout = timeout
        elif spec.timeout_s is not None:
            call_timeout = spec.timeout_s
        else:
            call_timeout = self.timeout
        throttled_s = 0.0
        retrying = Retrying(
            stop=stop_after_attempt(self.max_attempts),
            wait=self._wait,
            retry=retry_if_exception(_should_retry(spec)),
            sleep=self._sleep,
            before_sleep=self._log_retry,
            reraise=True,  # surface our own error type, not tenacity's RetryError
        )
        for attempt in retrying:
            with attempt:
                attempts = attempt.retry_state.attempt_number
                if self._limiter is not None:
                    throttled_s += self._limiter.acquire(spec)
                start = time.perf_counter()
                raw = self._transport(
                    spec,
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=call_timeout,
                )
                latency_ms = (time.perf_counter() - start) * 1000

        resp = LLMResponse(
            text=raw.text,
            alias=alias,
            model=spec.model_id,
            provider=spec.provider,
            input_tokens=raw.input_tokens,
            output_tokens=raw.output_tokens,
            reasoning_tokens=raw.reasoning_tokens,
            latency_ms=latency_ms,
            cost_usd=spec.cost(raw.input_tokens, raw.output_tokens),
            cached=False,
            attempts=attempts,
            finish_reason=raw.finish_reason,
        )
        if self._cache is not None and key is not None:
            self._cache.put(key, resp, cache)

        log.info(
            "%s ok attempts=%d latency=%.0fms throttled=%.1fs in=%s out=%s cost=%s finish=%s",
            alias,
            attempts,
            latency_ms,
            throttled_s,
            resp.input_tokens,
            resp.output_tokens,
            f"${resp.cost_usd:.6f}" if resp.cost_usd is not None else "n/a",
            resp.finish_reason,
        )
        if not resp.text.strip():
            log.warning(
                "%s returned an empty answer (finish=%s); not cached", alias, resp.finish_reason
            )
        return resp

    def _wait(self, state: RetryCallState) -> float:
        """Exponential backoff with jitter, but never shorter than the provider's Retry-After,
        and never longer than max_wait."""
        backoff = self._backoff(state)
        exc = state.outcome.exception() if state.outcome is not None else None
        retry_after = getattr(exc, "retry_after", None) or 0.0
        return min(max(backoff, retry_after), self.max_wait)

    def _log_retry(self, state: RetryCallState) -> None:
        exc = state.outcome.exception() if state.outcome is not None else None
        wait = state.next_action.sleep if state.next_action is not None else 0.0
        log.warning("attempt %d failed (%s); retrying in %.1fs", state.attempt_number, exc, wait)
