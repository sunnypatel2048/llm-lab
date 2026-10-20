"""`llm-lab-agent`: run the tool-calling agent on a question and print its trace.

    uv run llm-lab-agent --model groq-oss                       # the demo question
    uv run llm-lab-agent --model gemini-lite --cache off --expect 169.42
    uv run llm-lab-agent "Which customer has spent the most?" --model local-qwen-nothink

Steps print as they happen (useful with slow local models). Exit code 1 if the run stopped
early or, with --expect, if the answer doesn't contain the expected value.
"""

from __future__ import annotations

import argparse
import json
import logging
import re

from llm_lab.agent import AgentError, AgentResult, Step, run_agent
from llm_lab.cache import CACHE_MODES
from llm_lab.client import LLMClient
from llm_lab.tools import shop_tools
from llm_lab.types import LLMError

DEMO_QUESTION = (
    "How much has jane@example.com spent on delivered orders, after her loyalty discount?"
)


def _short(text: str, limit: int) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def _compact_json(text: str) -> str:
    """Re-serialize JSON without spaces so traces stay on one line; fall back to the raw text."""
    try:
        return json.dumps(json.loads(text), separators=(",", ":"))
    except ValueError:  # includes json.JSONDecodeError
        return text


def _cost(value: float | None) -> str:
    if value is None:
        return "$?"
    return "$0" if value == 0 else f"${value:.6f}"


def _stats(step: Step) -> str:
    r = step.response
    tokens = f"in {r.input_tokens if r.input_tokens is not None else '?'}"
    tokens += f" out {r.output_tokens if r.output_tokens is not None else '?'}"
    when = "cached" if r.cached else f"{r.latency_ms / 1000:.1f} s"
    return f"[{tokens} · {_cost(r.cost_usd)} · {when}]"


def format_step(step: Step) -> list[str]:
    prefix = f"step {step.number}"
    if not step.results:
        return [f"{prefix}  answer: {_short(step.response.text, 120)}  {_stats(step)}"]
    lines = []
    for i, (call, result) in enumerate(zip(step.response.tool_calls, step.results, strict=True)):
        flag = "" if result.ok else f"  !! {result.kind}"
        args = _short(_compact_json(call.arguments), 70)
        output = _short(_compact_json(result.content), 90)
        lead = prefix if i == 0 else " " * len(prefix)
        stats = f"  {_stats(step)}" if i == 0 else ""
        lines.append(f"{lead}  {call.name}({args}) -> {output}{flag}{stats}")
    return lines


def format_totals(result: AgentResult) -> str:
    cost = _cost(result.cost_usd) + ("" if result.cost_complete else " (some steps unreported)")
    return (
        f"{len(result.steps)} steps · {result.tool_calls} tool calls · {result.repairs} repairs · "
        f"{result.input_tokens:,} in / {result.output_tokens:,} out tokens · {cost} · "
        f"{result.latency_ms / 1000:.1f} s"
    )


def contains_expected(answer: str, expected: str) -> bool:
    """True if the answer states the expected number: "$169.42", "169.42 USD" and "1,169.42"
    style formatting are all fine, but 169.42 must not match inside 2169.421."""
    plain = answer.replace(",", "")
    return re.search(rf"(?<![\d.]){re.escape(expected)}(?![\d])", plain) is not None


def main(argv: list[str] | None = None, client: LLMClient | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the tool-calling agent on a question.")
    parser.add_argument("question", nargs="?", default=DEMO_QUESTION)
    parser.add_argument("--model", default="groq-oss")
    parser.add_argument("--cache", choices=CACHE_MODES, default="use")
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--expect", help="a value the answer must contain, e.g. 169.42")
    parser.add_argument("-v", "--verbose", action="store_true", help="log each call and retry")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    client = client or LLMClient.from_env()
    print(f"Q: {args.question}\nmodel: {args.model}\n")

    def show(step: Step) -> None:
        for line in format_step(step):
            print(line, flush=True)

    try:
        result = run_agent(
            client,
            args.model,
            args.question,
            shop_tools(),
            max_steps=args.max_steps,
            max_tokens=args.max_tokens,
            cache=args.cache,
            on_step=show,
        )
    except AgentError as e:
        print(f"\nstopped ({e.result.stop_reason}): {e}")
        print(format_totals(e.result))
        return 1
    except LLMError as e:  # e.g. an unknown alias or missing key, before the first step
        print(f"error: {e}")
        return 1

    print(f"\n{format_totals(result)}")
    print(f"answer: {result.answer}")
    if args.expect is not None:
        ok = contains_expected(result.answer or "", args.expect)
        print(f"check: {'PASS' if ok else 'FAIL'} (expected {args.expect})")
        return 0 if ok else 1
    return 0
