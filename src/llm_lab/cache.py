"""Response cache: the same request never costs quota twice.

Stored in SQLite (.cache/llm.sqlite): one file, safe to write from several threads, and
built into Python. Keys are SHA-256 hashes of everything that determines the answer.

Modes, chosen per call:
  "use"     read hits, write misses (default)
  "refresh" skip the read, still write (get a fresh answer and keep it)
  "off"     neither read nor write
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from llm_lab.registry import ModelSpec
from llm_lab.types import LLMResponse, Message

log = logging.getLogger(__name__)

CacheMode = Literal["use", "refresh", "off"]
CACHE_MODES: tuple[str, ...] = ("use", "refresh", "off")
DEFAULT_CACHE_PATH = Path(".cache/llm.sqlite")

# Bump when the key's contents change, so old entries stop matching instead of colliding.
KEY_VERSION = 1


def cache_key(
    spec: ModelSpec,
    messages: list[Message],
    *,
    temperature: float,
    max_tokens: int,
    extra: dict[str, Any] | None = None,
) -> str:
    """Hash of everything that determines the answer.

    - Provider + model ID, not the alias: renaming an alias keeps hits, a new model version
      misses. Provider is included because the same open model can answer differently on
      different hosts.
    - Canonical JSON (sorted keys, fixed separators), so {"role", "content"} in a different
      key order hashes the same. Message ORDER still matters, as it should.
    - temperature is cast to float so 0 and 0.0 don't produce two different keys.
    - extra: future request options (tools, response format) that change the answer.
    """
    payload = {
        "v": KEY_VERSION,
        "provider": spec.provider,
        "model": spec.model_id,
        "messages": messages,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "extra": extra or {},
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _check_mode(mode: str) -> None:
    if mode not in CACHE_MODES:
        raise ValueError(f"cache mode must be one of {CACHE_MODES}, got {mode!r}")


class ResponseCache:
    def __init__(self, path: str | Path = DEFAULT_CACHE_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn, conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS responses (
                       key        TEXT PRIMARY KEY,
                       alias      TEXT NOT NULL,
                       model      TEXT NOT NULL,
                       created_at TEXT NOT NULL,
                       response   TEXT NOT NULL
                   )"""
            )

    def _connect(self) -> sqlite3.Connection:
        # A short-lived connection per operation keeps this thread-safe without sharing
        # connections. WAL mode lets readers run while another thread writes; timeout makes
        # a writer wait for a lock instead of failing immediately.
        conn = sqlite3.connect(self.path, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def get(self, key: str, mode: CacheMode = "use") -> LLMResponse | None:
        """The cached response (with cached=True), or None on a miss or when mode isn't "use"."""
        _check_mode(mode)
        if mode != "use":
            return None
        with closing(self._connect()) as conn:
            row = conn.execute("SELECT response FROM responses WHERE key = ?", (key,)).fetchone()
        if row is None:
            return None
        try:
            stored = LLMResponse.from_dict(json.loads(row[0]))
        except (TypeError, ValueError) as e:
            # Written by an older/newer LLMResponse shape: treat as a miss rather than crash.
            log.debug("Ignoring unreadable cache entry %s: %s", key[:12], e)
            return None
        return dataclasses.replace(stored, cached=True)

    def put(self, key: str, response: LLMResponse, mode: CacheMode = "use") -> bool:
        """Store a response. Returns False if it was skipped (mode "off" or an empty answer).

        Empty answers usually mean the response was blocked or cut off; caching one would
        hide that problem on every later run. A turn that only calls tools has no text but
        isn't empty, so it is cached.
        """
        _check_mode(mode)
        if mode == "off" or (not response.text.strip() and not response.tool_calls):
            return False
        record = dataclasses.asdict(dataclasses.replace(response, cached=False))
        with closing(self._connect()) as conn, conn:
            conn.execute(
                "INSERT OR REPLACE INTO responses (key, alias, model, created_at, response) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    key,
                    response.alias,
                    response.model,
                    datetime.now(UTC).isoformat(timespec="seconds"),
                    json.dumps(record, ensure_ascii=False),
                ),
            )
        return True

    def __len__(self) -> int:
        with closing(self._connect()) as conn:
            return conn.execute("SELECT COUNT(*) FROM responses").fetchone()[0]

    def clear(self) -> None:
        with closing(self._connect()) as conn, conn:
            conn.execute("DELETE FROM responses")
