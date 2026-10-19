"""Send one prompt to several models and print a side-by-side table.

    uv run llm-lab "Explain BM25 in two sentences"
    uv run llm-lab "..." --models local-qwen,groq-oss --cache off

Each model runs in turn; a failure on one shows up in its row (with the full message below
the table) instead of stopping the run. Exit code is 1 if any model failed, so the script
can gate a CI job or a shell `&&` chain.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass

from llm_lab.cache import CACHE_MODES
from llm_lab.client import LLMClient
from llm_lab.types import LLMError, LLMResponse

PREVIEW_CHARS = 60
COLUMNS = (
    ("model", "<"),
    ("status", "<"),
    ("latency", ">"),
    ("tries", ">"),
    ("in", ">"),
    ("out", ">"),
    ("thinking", ">"),
    ("cost", ">"),
    ("finish", "<"),
    ("answer", "<"),
)


@dataclass(frozen=True, slots=True)
class Row:
    alias: str
    response: LLMResponse | None = None
    error: LLMError | None = None

    @property
    def status(self) -> str:
        if self.error is not None:
            return f"error: {type(self.error).__name__}"
        assert self.response is not None
        if not self.response.text.strip():
            return "empty"
        return "cached" if self.response.cached else "ok"


def run(
    client: LLMClient,
    aliases: list[str],
    prompt: str,
    *,
    max_tokens: int = 2048,
    cache: str = "use",
) -> list[Row]:
    messages = [{"role": "user", "content": prompt}]
    rows = []
    for alias in aliases:
        try:
            resp = client.chat(alias, messages, max_tokens=max_tokens, cache=cache)
            rows.append(Row(alias, response=resp))
        except LLMError as e:
            rows.append(Row(alias, error=e))
    return rows


def _num(value: int | None) -> str:
    return "–" if value is None else str(value)


def _cost(value: float | None) -> str:
    if value is None:
        return "–"
    return "$0" if value == 0 else f"${value:.6f}"


def _preview(text: str) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= PREVIEW_CHARS else flat[: PREVIEW_CHARS - 1] + "…"


def _cells(row: Row) -> list[str]:
    r = row.response
    if r is None:
        return [row.alias, row.status, "–", "–", "–", "–", "–", "–", "–", "see below"]
    return [
        row.alias,
        row.status,
        f"{r.latency_ms:,.0f} ms",
        str(r.attempts),
        _num(r.input_tokens),
        _num(r.output_tokens),
        _num(r.reasoning_tokens),
        _cost(r.cost_usd),
        r.finish_reason or "–",
        _preview(r.text),
    ]


def render(rows: list[Row]) -> str:
    table = [[name for name, _ in COLUMNS]] + [_cells(row) for row in rows]
    widths = [max(len(line[i]) for line in table) for i in range(len(COLUMNS))]

    def fmt(line: list[str]) -> str:
        parts = [
            f"{cell:{align}{width}}"
            for cell, (_, align), width in zip(line, COLUMNS, widths, strict=True)
        ]
        return "  ".join(parts).rstrip()

    out = [fmt(table[0]), "  ".join("-" * w for w in widths)]
    out += [fmt(line) for line in table[1:]]

    errors = [row for row in rows if row.error is not None]
    if errors:
        out += ["", "Errors:"]
        out += [f"  {row.error}" for row in errors]  # messages already name the model
    if any(row.response is not None and row.response.cached for row in rows):
        out += ["", "Cached rows show the latency and cost of the original call."]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Send one prompt to several models.")
    parser.add_argument("prompt")
    parser.add_argument("--models", help="comma-separated aliases (default: all in models.toml)")
    parser.add_argument("--cache", choices=CACHE_MODES, default="use")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("-v", "--verbose", action="store_true", help="log each call and retry")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    client = LLMClient.from_env()
    aliases = (
        [a.strip() for a in args.models.split(",") if a.strip()]
        if args.models
        else client.registry.aliases()
    )
    rows = run(client, aliases, args.prompt, max_tokens=args.max_tokens, cache=args.cache)
    print(render(rows))
    return 1 if any(row.error is not None for row in rows) else 0
