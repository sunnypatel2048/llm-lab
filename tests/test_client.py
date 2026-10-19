"""Client tests with a fake transport: no network, no API keys, no `openai` calls."""

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from llm_lab.cache import ResponseCache
from llm_lab.client import LLMClient
from llm_lab.ratelimit import DailyCounter, RateLimiter
from llm_lab.registry import Registry
from llm_lab.transport import RawCompletion
from llm_lab.types import (
    BadRequest,
    ConfigError,
    DailyQuotaExhausted,
    LLMTimeout,
    ProviderError,
    ProviderUnavailable,
    RateLimited,
)

MSGS = [{"role": "user", "content": "Explain BM25 in two sentences."}]
OK = RawCompletion(
    text="BM25 ranks documents...",
    input_tokens=1000,
    output_tokens=500,
    reasoning_tokens=None,
    finish_reason="stop",
)


class FakeTransport:
    """Plays back a script of results; an Exception in the script is raised instead."""

    def __init__(self, *script):
        self.script = list(script)
        self.calls = []

    def __call__(self, spec, messages, *, temperature, max_tokens, timeout):
        self.calls.append(
            dict(alias=spec.alias, temperature=temperature, max_tokens=max_tokens, timeout=timeout)
        )
        item = self.script.pop(0) if self.script else OK
        if isinstance(item, Exception):
            raise item
        return item


class CountingLimiter:
    def __init__(self):
        self.acquired = 0

    def acquire(self, spec):
        self.acquired += 1
        return 0.0


class RecordingSleep:
    def __init__(self):
        self.waits = []

    def __call__(self, seconds):
        self.waits.append(seconds)


def registry():
    return Registry.from_dict(
        {
            "providers": {
                "gemini": {"base_url": "https://g.test/", "api_key_env": "TEST_GEMINI_KEY"},
                "ollama": {"base_url": "http://localhost:11434/v1"},
            },
            "models": {
                "gemini-flash": {
                    "provider": "gemini",
                    "model_id": "gemini-3.8-flash",
                    "price_in_per_m": 0.75,
                    "price_out_per_m": 3.75,
                    "rpm": 5,
                    "rpd": 20,
                },
                "local": {
                    "provider": "ollama",
                    "model_id": "qwen3.5:4b",
                    "price_in_per_m": 0.0,
                    "price_out_per_m": 0.0,
                },
            },
        }
    )


@pytest.fixture(autouse=True)
def gemini_key(monkeypatch):
    monkeypatch.setenv("TEST_GEMINI_KEY", "test-key")


def make_client(*script, cache=None, limiter=None, **kwargs):
    transport = FakeTransport(*script)
    sleep = RecordingSleep()
    client = LLMClient(
        registry(), transport=transport, cache=cache, limiter=limiter, sleep=sleep, **kwargs
    )
    return client, transport, sleep


# --- happy path -----------------------------------------------------------------------


def test_success_fills_every_field_and_prices_at_paid_rates():
    client, transport, _ = make_client(OK)
    resp = client.chat("gemini-flash", MSGS, max_tokens=256, timeout=5)
    assert resp.text == "BM25 ranks documents..."
    assert (resp.alias, resp.model, resp.provider) == ("gemini-flash", "gemini-3.8-flash", "gemini")
    assert (resp.input_tokens, resp.output_tokens) == (1000, 500)
    assert resp.cost_usd == pytest.approx(0.002625)  # 1000*0.75/1M + 500*3.75/1M
    assert resp.attempts == 1 and resp.cached is False and resp.finish_reason == "stop"
    assert resp.latency_ms >= 0
    assert transport.calls == [
        dict(alias="gemini-flash", temperature=0.0, max_tokens=256, timeout=5)
    ]


def test_missing_usage_gives_no_cost():
    client, _, _ = make_client(RawCompletion("hi", None, None, None, "stop"))
    assert client.chat("gemini-flash", MSGS).cost_usd is None


# --- cache ----------------------------------------------------------------------------


def test_cache_hit_skips_transport_and_quota(tmp_path):
    limiter = CountingLimiter()
    client, transport, _ = make_client(
        OK, cache=ResponseCache(tmp_path / "c.sqlite"), limiter=limiter
    )
    first = client.chat("gemini-flash", MSGS)
    second = client.chat("gemini-flash", MSGS)
    assert len(transport.calls) == 1
    assert limiter.acquired == 1
    assert first.cached is False and second.cached is True
    assert second.text == first.text


def test_refresh_mode_calls_again(tmp_path):
    client, transport, _ = make_client(OK, OK, cache=ResponseCache(tmp_path / "c.sqlite"))
    client.chat("gemini-flash", MSGS)
    client.chat("gemini-flash", MSGS, cache="refresh")
    assert len(transport.calls) == 2


def test_empty_answer_is_returned_but_not_cached(tmp_path):
    empty = RawCompletion("", 50, 512, 512, "length")  # thinking used the whole budget
    client, transport, _ = make_client(empty, OK, cache=ResponseCache(tmp_path / "c.sqlite"))
    resp = client.chat("local", MSGS)
    assert resp.text == "" and resp.finish_reason == "length"
    client.chat("local", MSGS)
    assert len(transport.calls) == 2  # second call went to the model again


def test_unknown_cache_mode_rejected():
    client, _, _ = make_client()
    with pytest.raises(ValueError, match="cache must be one of"):
        client.chat("gemini-flash", MSGS, cache="sometimes")


# --- retries --------------------------------------------------------------------------


def test_429_then_success_retries_and_counts_both_attempts():
    limiter = CountingLimiter()
    client, transport, sleep = make_client(RateLimited("slow down"), OK, limiter=limiter)
    resp = client.chat("gemini-flash", MSGS)
    assert resp.attempts == 2
    assert len(transport.calls) == 2
    assert limiter.acquired == 2  # the retry was a real request, so it took a slot too
    assert len(sleep.waits) == 1


@pytest.mark.parametrize("error", [ProviderError("HTTP 503"), LLMTimeout("timed out")])
def test_5xx_and_timeouts_are_retried(error):
    client, transport, _ = make_client(error, OK)
    assert client.chat("gemini-flash", MSGS).attempts == 2


def test_retry_after_is_respected():
    client, _, sleep = make_client(RateLimited("slow down", retry_after=7), OK)
    client.chat("gemini-flash", MSGS)
    assert sleep.waits[0] >= 7


def test_waits_are_capped_at_max_wait():
    client, _, sleep = make_client(RateLimited("slow down", retry_after=3600), OK, max_wait=60)
    client.chat("gemini-flash", MSGS)
    assert sleep.waits == [60]


def test_gives_up_after_max_attempts_with_our_error_type():
    client, transport, sleep = make_client(*[RateLimited("slow down")] * 4, max_attempts=4)
    with pytest.raises(RateLimited):
        client.chat("gemini-flash", MSGS)
    assert len(transport.calls) == 4
    assert len(sleep.waits) == 3


@pytest.mark.parametrize(
    "error",
    [
        BadRequest("HTTP 400: bad model"),
        ProviderUnavailable("couldn't connect"),
        DailyQuotaExhausted("20/20 used"),
    ],
)
def test_non_retryable_errors_fail_on_the_first_attempt(error):
    client, transport, sleep = make_client(error)
    with pytest.raises(type(error)):
        client.chat("gemini-flash", MSGS)
    assert len(transport.calls) == 1
    assert sleep.waits == []


# --- fail fast ------------------------------------------------------------------------


def test_missing_api_key_fails_before_any_call(monkeypatch):
    monkeypatch.delenv("TEST_GEMINI_KEY")
    client, transport, _ = make_client()
    with pytest.raises(ConfigError, match="TEST_GEMINI_KEY"):
        client.chat("gemini-flash", MSGS)
    assert transport.calls == []


def test_local_model_needs_no_key(monkeypatch):
    monkeypatch.delenv("TEST_GEMINI_KEY")
    client, _, _ = make_client(OK)
    assert client.chat("local", MSGS).text


def test_unknown_alias_is_a_config_error():
    client, _, _ = make_client()
    with pytest.raises(ConfigError, match="Unknown model alias"):
        client.chat("gpt-9", MSGS)


class FakeMonotonic:
    """Fake clock for the real limiter: sleeping moves time forward instantly. (A no-op sleep
    on the real clock would leave the token bucket waiting forever for time to pass.)"""

    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, seconds):
        self.t += seconds


def test_daily_quota_stops_calls_with_the_real_limiter(tmp_path):
    now = lambda: datetime(2026, 10, 18, 12, tzinfo=ZoneInfo("America/Los_Angeles"))  # noqa: E731
    clock = FakeMonotonic()
    limiter = RateLimiter(
        DailyCounter(tmp_path / "q.json", now=now), clock=clock, sleep=clock.sleep
    )
    client, transport, _ = make_client(limiter=limiter)
    for _ in range(20):  # gemini-flash allows 20/day
        client.chat("gemini-flash", MSGS)
    with pytest.raises(DailyQuotaExhausted, match="20/20"):
        client.chat("gemini-flash", MSGS)
    assert len(transport.calls) == 20
    assert clock.t == pytest.approx(15 * 12)  # 5 free, then 15 more at one per 12s
