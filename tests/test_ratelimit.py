import json
import threading
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from llm_lab.ratelimit import DailyCounter, RateLimiter, TokenBucket
from llm_lab.registry import ModelSpec
from llm_lab.types import DailyQuotaExhausted

PT = ZoneInfo("America/Los_Angeles")
ET = ZoneInfo("America/New_York")


class FakeClock:
    """Fake monotonic time: sleep() advances it instantly, so tests take milliseconds."""

    def __init__(self):
        self.t = 1000.0
        self.slept = 0.0

    def __call__(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += seconds
        self.slept += seconds


class FakeNow:
    """Settable wall clock for the daily counter."""

    def __init__(self, dt: datetime):
        self.dt = dt

    def __call__(self) -> datetime:
        return self.dt


def spec(alias="m", rpm=None, rpd=None) -> ModelSpec:
    return ModelSpec(
        alias=alias,
        provider="test",
        model_id="x",
        base_url="https://example.test/",
        api_key_env=None,
        price_in_per_m=0.0,
        price_out_per_m=0.0,
        rpm=rpm,
        rpd=rpd,
    )


# --- TokenBucket ----------------------------------------------------------------------


def test_bucket_starts_full_so_first_burst_is_free():
    clock = FakeClock()
    bucket = TokenBucket(5, clock=clock, sleep=clock.sleep)
    assert [bucket.acquire() for _ in range(5)] == [0.0] * 5
    assert clock.slept == 0.0


def test_bucket_waits_one_refill_interval_when_empty():
    clock = FakeClock()
    bucket = TokenBucket(5, clock=clock, sleep=clock.sleep)  # 5/min = one token per 12s
    for _ in range(5):
        bucket.acquire()
    assert bucket.acquire() == pytest.approx(12.0)
    assert bucket.acquire() == pytest.approx(12.0)


def test_bucket_refills_partially_while_idle():
    clock = FakeClock()
    bucket = TokenBucket(5, clock=clock, sleep=clock.sleep)
    for _ in range(5):
        bucket.acquire()
    clock.t += 6.0  # half a token refilled
    assert bucket.acquire() == pytest.approx(6.0)  # only the other half needs waiting


def test_bucket_never_holds_more_than_capacity():
    clock = FakeClock()
    bucket = TokenBucket(5, clock=clock, sleep=clock.sleep)
    for _ in range(5):
        bucket.acquire()
    clock.t += 3600  # an hour idle must not bank 300 tokens
    assert [bucket.acquire() for _ in range(5)] == [0.0] * 5
    assert bucket.acquire() == pytest.approx(12.0)


@pytest.mark.parametrize("bad", [0, -3])
def test_bucket_rejects_non_positive_rate(bad):
    with pytest.raises(ValueError):
        TokenBucket(bad)


def test_bucket_never_hands_out_a_token_twice_under_concurrency():
    # 60/min = 1 token/s, starting with 60. 70 threads => the extra 10 must wait 1s each,
    # one after another: 10s of fake time in total. If two threads ever took the same token,
    # the total would come out lower.
    clock = FakeClock()
    bucket = TokenBucket(60, clock=clock, sleep=clock.sleep)
    threads = [threading.Thread(target=bucket.acquire) for _ in range(70)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert clock.slept == pytest.approx(10.0)


# --- DailyCounter ---------------------------------------------------------------------


def make_counter(tmp_path, dt=datetime(2026, 10, 18, 12, 0, tzinfo=PT)):
    now = FakeNow(dt)
    return DailyCounter(tmp_path / "quota.json", now=now), now


def test_counter_refuses_once_limit_reached(tmp_path):
    counter, _ = make_counter(tmp_path)
    assert counter.consume("gemini-flash", 2) == 1
    assert counter.consume("gemini-flash", 2) == 2
    with pytest.raises(DailyQuotaExhausted, match="gemini-flash: 2/2 requests used today"):
        counter.consume("gemini-flash", 2)


def test_counter_tracks_each_alias_separately(tmp_path):
    counter, _ = make_counter(tmp_path)
    counter.consume("a", 1)
    counter.consume("b", 1)  # must not be blocked by "a"
    assert (counter.used("a"), counter.used("b")) == (1, 1)


def test_counter_survives_restart(tmp_path):
    first, now = make_counter(tmp_path)
    first.consume("a", 5)
    first.consume("a", 5)
    second = DailyCounter(tmp_path / "quota.json", now=now)  # new process, same file
    assert second.used("a") == 2


def test_counter_resets_at_pacific_midnight(tmp_path):
    counter, now = make_counter(tmp_path, datetime(2026, 10, 18, 23, 59, tzinfo=PT))
    counter.consume("a", 1)
    with pytest.raises(DailyQuotaExhausted):
        counter.consume("a", 1)
    now.dt = datetime(2026, 10, 19, 0, 1, tzinfo=PT)
    assert counter.consume("a", 1) == 1


def test_counter_day_follows_pacific_not_local_time(tmp_path):
    # 1:30am Eastern on Oct 19 is still 10:30pm Oct 18 in Pacific: same quota day.
    counter, now = make_counter(tmp_path, datetime(2026, 10, 18, 20, 0, tzinfo=PT))
    counter.consume("a", 1)
    now.dt = datetime(2026, 10, 19, 1, 30, tzinfo=ET)
    with pytest.raises(DailyQuotaExhausted):
        counter.consume("a", 1)


def test_counter_ignores_corrupt_file_with_warning(tmp_path, caplog):
    (tmp_path / "quota.json").write_text("{not json")
    counter, _ = make_counter(tmp_path)
    assert counter.consume("a", 5) == 1
    assert "unreadable quota file" in caplog.text


def test_counter_writes_atomically_without_leftovers(tmp_path):
    counter, _ = make_counter(tmp_path)
    counter.consume("a", 5)
    assert [p.name for p in tmp_path.iterdir()] == ["quota.json"]
    assert json.loads((tmp_path / "quota.json").read_text()) == {"2026-10-18": {"a": 1}}


def test_counter_prunes_days_older_than_a_week(tmp_path):
    old_days = {"2026-10-01": {"a": 3}, "2026-10-15": {"a": 1}}
    (tmp_path / "quota.json").write_text(json.dumps(old_days))
    counter, _ = make_counter(tmp_path)
    counter.consume("a", 5)
    assert set(json.loads((tmp_path / "quota.json").read_text())) == {"2026-10-15", "2026-10-18"}


# --- RateLimiter ----------------------------------------------------------------------


def test_limiter_skips_everything_for_unlimited_models(tmp_path):
    counter, _ = make_counter(tmp_path)
    limiter = RateLimiter(counter)
    assert limiter.acquire(spec("local-qwen")) == 0.0
    assert not (tmp_path / "quota.json").exists()


def test_limiter_checks_daily_quota_before_waiting(tmp_path):
    counter, _ = make_counter(tmp_path)
    clock = FakeClock()
    limiter = RateLimiter(counter, clock=clock, sleep=clock.sleep)
    s = spec("gemini-flash", rpm=1, rpd=1)
    limiter.acquire(s)
    with pytest.raises(DailyQuotaExhausted):
        limiter.acquire(s)
    assert clock.slept == 0.0  # refused immediately, no 60s wait first


def test_limiter_keeps_a_separate_bucket_per_model(tmp_path):
    counter, _ = make_counter(tmp_path)
    clock = FakeClock()
    limiter = RateLimiter(counter, clock=clock, sleep=clock.sleep)
    limiter.acquire(spec("a", rpm=1))
    assert limiter.acquire(spec("b", rpm=1)) == 0.0  # "a" being empty doesn't slow "b"
    assert limiter.acquire(spec("a", rpm=1)) == pytest.approx(60.0)
