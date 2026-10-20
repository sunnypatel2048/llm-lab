import dataclasses
import json
import sqlite3
import threading

import pytest

from llm_lab.cache import ResponseCache, cache_key
from llm_lab.registry import ModelSpec
from llm_lab.types import LLMResponse, ToolCall

MSGS = [{"role": "user", "content": "Explain BM25 in two sentences."}]


def spec(alias="gemini-flash", provider="gemini", model_id="gemini-3.8-flash") -> ModelSpec:
    return ModelSpec(
        alias=alias,
        provider=provider,
        model_id=model_id,
        base_url="https://example.test/",
        api_key_env=None,
        price_in_per_m=0.75,
        price_out_per_m=3.75,
    )


def response(text="BM25 ranks documents by term frequency...", **overrides) -> LLMResponse:
    fields = dict(
        text=text,
        alias="gemini-flash",
        model="gemini-3.8-flash",
        provider="gemini",
        input_tokens=12,
        output_tokens=40,
        reasoning_tokens=None,
        latency_ms=812.5,
        cost_usd=0.000159,
    )
    return LLMResponse(**(fields | overrides))


def key(s=None, messages=MSGS, temperature=0.0, max_tokens=512, extra=None):
    return cache_key(
        s or spec(), messages, temperature=temperature, max_tokens=max_tokens, extra=extra
    )


@pytest.fixture
def cache(tmp_path):
    return ResponseCache(tmp_path / "llm.sqlite")


# --- cache_key ------------------------------------------------------------------------


def test_key_is_deterministic():
    assert key() == key()
    assert len(key()) == 64  # hex SHA-256


def test_key_ignores_dict_key_order_inside_messages():
    reordered = [{"content": MSGS[0]["content"], "role": "user"}]
    assert key(messages=reordered) == key()


def test_key_depends_on_message_order():
    a = [{"role": "user", "content": "A"}, {"role": "user", "content": "B"}]
    assert key(messages=a) != key(messages=list(reversed(a)))


def test_int_and_float_temperature_share_a_key():
    assert key(temperature=0) == key(temperature=0.0)


@pytest.mark.parametrize(
    "changed",
    [
        {"temperature": 0.7},
        {"max_tokens": 256},
        {"s": spec(model_id="gemini-3.9-flash")},
        {"s": spec(provider="other-host")},
        {"extra": {"response_format": "json"}},
    ],
)
def test_anything_that_changes_the_answer_changes_the_key(changed):
    assert key(**changed) != key()


def test_alias_is_not_part_of_the_key():
    assert key(spec(alias="renamed")) == key()


# --- ResponseCache --------------------------------------------------------------------


def test_miss_returns_none(cache):
    assert cache.get(key()) is None


def test_round_trip_marks_hit_as_cached(cache):
    original = response()
    assert cache.put(key(), original)
    hit = cache.get(key())
    assert hit.cached is True
    assert dataclasses.replace(hit, cached=False) == original  # every other field intact


def test_entries_survive_a_new_instance(tmp_path):
    ResponseCache(tmp_path / "llm.sqlite").put(key(), response())
    assert ResponseCache(tmp_path / "llm.sqlite").get(key()) is not None


def test_refresh_skips_the_read_but_still_writes(cache):
    cache.put(key(), response(text="old answer"))
    assert cache.get(key(), mode="refresh") is None
    cache.put(key(), response(text="new answer"), mode="refresh")
    assert cache.get(key()).text == "new answer"


def test_off_neither_reads_nor_writes(cache):
    assert cache.put(key(), response(), mode="off") is False
    assert len(cache) == 0
    cache.put(key(), response())
    assert cache.get(key(), mode="off") is None


@pytest.mark.parametrize("empty", ["", "   \n"])
def test_empty_answers_are_not_cached(cache, empty):
    assert cache.put(key(), response(text=empty)) is False
    assert cache.get(key()) is None


def test_unknown_mode_rejected(cache):
    with pytest.raises(ValueError, match="cache mode"):
        cache.get(key(), mode="sometimes")


def test_unicode_round_trips(cache):
    cache.put(key(), response(text="Résumé — naïve café ✓ 日本語"))
    assert cache.get(key()).text == "Résumé — naïve café ✓ 日本語"


def test_entry_with_unexpected_shape_is_a_miss(cache):
    future = dataclasses.asdict(response()) | {"field_from_the_future": 1}
    with sqlite3.connect(cache.path) as conn:
        conn.execute(
            "INSERT INTO responses VALUES (?, ?, ?, ?, ?)",
            (key(), "gemini-flash", "gemini-3.8-flash", "2026-10-18", json.dumps(future)),
        )
    assert cache.get(key()) is None


def test_clear_empties_the_cache(cache):
    cache.put(key(), response())
    cache.clear()
    assert len(cache) == 0


def test_concurrent_writes_all_land(cache):
    keys = [key(messages=[{"role": "user", "content": f"q{i}"}]) for i in range(20)]
    threads = [threading.Thread(target=cache.put, args=(k, response(text=k))) for k in keys]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(cache) == 20
    assert all(cache.get(k).text == k for k in keys)


def test_tool_call_turn_round_trips(cache):
    call = ToolCall(id="call_1", name="find_customer", arguments='{"email": "a@b.c"}')
    message = {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1"}]}
    assert cache.put(key(), response(text="", tool_calls=(call,), message=message))
    hit = cache.get(key())
    assert hit.tool_calls == (call,) and hit.message == message


def test_entries_written_before_tool_support_still_load(cache):
    old = {
        k: v
        for k, v in dataclasses.asdict(response()).items()
        if k not in ("tool_calls", "message")
    }
    with sqlite3.connect(cache.path) as conn:
        conn.execute(
            "INSERT INTO responses VALUES (?, ?, ?, ?, ?)",
            (key(), "gemini-flash", "gemini-3.8-flash", "2026-10-18", json.dumps(old)),
        )
    hit = cache.get(key())
    assert hit is not None and hit.tool_calls == () and hit.message is None
