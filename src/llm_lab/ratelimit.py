"""Client-side rate limiting, so we spend our free-tier quota on answers, not on 429s.

Two limits per model, matching how providers publish them:
  - TokenBucket:  requests per minute (rpm). Waits before sending instead of failing after.
  - DailyCounter: requests per day (rpd). Refuses once the day's quota is used.
RateLimiter applies both for a ModelSpec. Cached responses never reach it (the client checks
the cache first), so cache hits cost no quota.

Scope: one process. Limits shared across processes or machines would need a shared store,
e.g. Redis with an atomic Lua script.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections.abc import Callable
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from llm_lab.registry import ModelSpec
from llm_lab.types import DailyQuotaExhausted

log = logging.getLogger(__name__)

# Gemini's daily limits are commonly reported to reset at midnight Pacific. If AI Studio ever
# disagrees with our counter, change this one constant.
QUOTA_RESET_TZ = ZoneInfo("America/Los_Angeles")
DEFAULT_QUOTA_PATH = Path(".cache/quota.json")


class TokenBucket:
    """Holds up to `rpm` tokens, refilling at rpm/60 per second. One request = one token.

    Starts full, so a burst of `rpm` requests goes out immediately, then one per 60/rpm seconds.
    `clock` must be monotonic (never jumps when the system clock is adjusted); `clock` and
    `sleep` are injectable so tests run on fake time.
    """

    def __init__(
        self,
        rpm: int,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        if rpm <= 0:
            raise ValueError(f"rpm must be positive, got {rpm}")
        self.capacity = float(rpm)
        self.refill_per_sec = rpm / 60.0
        self._clock = clock
        self._sleep = sleep
        self._tokens = self.capacity
        self._last = clock()
        self._lock = threading.Lock()

    def acquire(self) -> float:
        """Block until a token is available and take it. Returns seconds spent waiting.

        The lock is held while sleeping on purpose: if this caller has to wait, nobody else
        could get a token sooner anyway, and holding it makes waiters go one at a time
        instead of all waking at once and racing for the same token.
        """
        waited = 0.0
        with self._lock:
            while True:
                self._refill()
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return waited
                wait = (1.0 - self._tokens) / self.refill_per_sec
                self._sleep(wait)
                waited += wait

    def _refill(self) -> None:
        now = self._clock()
        elapsed = max(0.0, now - self._last)
        self._tokens = min(self.capacity, self._tokens + elapsed * self.refill_per_sec)
        self._last = now


class DailyCounter:
    """Per-model request counts per day, saved to a small JSON file so restarts don't reset them.

    File shape: {"2026-10-18": {"gemini-flash": 7, "gemini-lite": 120}, ...}
    Counts are taken BEFORE each request: a request that fails may still count against the
    provider's quota, so we err on the side of stopping early.
    """

    def __init__(
        self,
        path: str | Path = DEFAULT_QUOTA_PATH,
        *,
        now: Callable[[], datetime] = lambda: datetime.now(QUOTA_RESET_TZ),
        keep_days: int = 7,
    ):
        self.path = Path(path)
        self._now = now
        self._keep_days = keep_days
        self._lock = threading.Lock()

    def today(self) -> str:
        return self._now().astimezone(QUOTA_RESET_TZ).date().isoformat()

    def used(self, alias: str) -> int:
        with self._lock:
            return self._load().get(self.today(), {}).get(alias, 0)

    def consume(self, alias: str, limit: int) -> int:
        """Count one request for `alias`. Returns the new count; raises if the limit is reached."""
        with self._lock:
            data = self._load()
            day = self.today()
            counts = data.setdefault(day, {})
            used = counts.get(alias, 0)
            if used >= limit:
                raise DailyQuotaExhausted(
                    f"{alias}: {used}/{limit} requests used today "
                    f"(resets at midnight {QUOTA_RESET_TZ.key})"
                )
            counts[alias] = used + 1
            self._prune(data, day)
            self._save(data)
            return used + 1

    def _load(self) -> dict[str, dict[str, int]]:
        if not self.path.exists():
            return {}
        try:
            data = json.loads(self.path.read_text())
            if not isinstance(data, dict):
                raise ValueError("top level is not an object")
            return data
        except (OSError, ValueError) as e:
            # Fail open: the provider still enforces the real limit (we'd see 429s, never a
            # bill), so a damaged counter file shouldn't block all work.
            log.warning("Ignoring unreadable quota file %s (%s); starting fresh.", self.path, e)
            return {}

    def _save(self, data: dict[str, dict[str, int]]) -> None:
        # Write to a temp file, then rename over the real one. os.replace is atomic, so a
        # crash mid-write leaves the old file intact instead of a half-written one.
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
        os.replace(tmp, self.path)

    def _prune(self, data: dict[str, dict[str, int]], today: str) -> None:
        cutoff = (date.fromisoformat(today) - timedelta(days=self._keep_days)).isoformat()
        for day in [d for d in data if d < cutoff]:
            del data[day]


class RateLimiter:
    """Applies a model's daily and per-minute limits. One bucket per alias, created on first use."""

    def __init__(
        self,
        counter: DailyCounter | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.counter = counter or DailyCounter()
        self._clock = clock
        self._sleep = sleep
        self._buckets: dict[str, TokenBucket] = {}
        self._lock = threading.Lock()

    def acquire(self, spec: ModelSpec) -> float:
        """Call right before sending a request. Returns seconds spent waiting for the bucket.

        The daily check runs first, so a model that's out for the day fails immediately
        instead of after waiting for a per-minute token.
        """
        if spec.rpd is not None:
            self.counter.consume(spec.alias, spec.rpd)
        if spec.rpm is None:
            return 0.0
        with self._lock:
            bucket = self._buckets.get(spec.alias)
            if bucket is None:
                bucket = TokenBucket(spec.rpm, clock=self._clock, sleep=self._sleep)
                self._buckets[spec.alias] = bucket
        return bucket.acquire()
