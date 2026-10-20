"""A tool-calling agent loop, written without a framework.

Each step is one model call. If the model asks for tools, run them all, append their results
to the conversation and call again; if it answers instead, that's the final answer.

Guards:
  - Repair: a model mistake (arguments that aren't valid JSON or break the schema, or a tool
    that doesn't exist) goes back to the model as an error result it can fix. A second mistake
    on the same tool before that tool has succeeded once stops the run.
  - A tool failing on valid input ("no customer with that email") is an answer from the data,
    not a model mistake, so it doesn't count toward the repair limit.
  - max_steps caps the number of model calls, so a confused model can't loop forever.
Every way of stopping early raises AgentError carrying the trace and cost so far.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from llm_lab.cache import CacheMode
from llm_lab.client import LLMClient
from llm_lab.tools import Toolbox, ToolResult
from llm_lab.types import LLMError, LLMResponse, Message

SYSTEM_PROMPT = """\
You answer questions about a shop's customers and orders.
- Get every fact from the tools. Never guess or invent data.
- Use the calculate tool for every arithmetic step, however simple.
- When you have the answer, reply in one short sentence that ends with the final amount in USD \
to 2 decimal places, like $12.34."""

MODEL_MISTAKES = ("invalid_arguments", "unknown_tool")


@dataclass(frozen=True)
class Step:
    number: int
    response: LLMResponse
    results: tuple[ToolResult, ...]  # empty on the final answer step


@dataclass(frozen=True)
class AgentResult:
    answer: str | None
    steps: tuple[Step, ...]
    stop_reason: str  # answered | repeated_mistake | max_steps | provider_error | empty_answer

    def _known(self, field: str) -> list[float]:
        values = [getattr(s.response, field) for s in self.steps]
        return [v for v in values if v is not None]

    @property
    def input_tokens(self) -> int:
        return int(sum(self._known("input_tokens")))

    @property
    def output_tokens(self) -> int:
        return int(sum(self._known("output_tokens")))

    @property
    def cost_usd(self) -> float:
        """Sum over the steps whose cost is known; check cost_complete before quoting it."""
        return sum(self._known("cost_usd"))

    @property
    def cost_complete(self) -> bool:
        return all(s.response.cost_usd is not None for s in self.steps)

    @property
    def latency_ms(self) -> float:
        return sum(s.response.latency_ms for s in self.steps)

    @property
    def tool_calls(self) -> int:
        return sum(len(s.results) for s in self.steps)

    @property
    def repairs(self) -> int:
        """Model mistakes that were sent back for fixing."""
        return sum(r.kind in MODEL_MISTAKES for s in self.steps for r in s.results)


class AgentError(Exception):
    def __init__(self, message: str, result: AgentResult):
        super().__init__(message)
        self.result = result  # the trace and cost up to the point it stopped


def run_agent(
    client: LLMClient,
    alias: str,
    question: str,
    toolbox: Toolbox,
    *,
    max_steps: int = 6,
    system: str = SYSTEM_PROMPT,
    cache: CacheMode = "use",
    max_tokens: int = 2048,
    on_step: Callable[[Step], None] | None = None,
) -> AgentResult:
    messages: list[Message] = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]
    tools = toolbox.definitions()
    steps: list[Step] = []
    mistakes: dict[str, int] = {}  # tool name -> model mistakes since that tool last succeeded

    def stop(reason: str, message: str) -> AgentError:
        return AgentError(message, AgentResult(None, tuple(steps), reason))

    def record(step: Step) -> None:
        steps.append(step)
        if on_step is not None:
            on_step(step)

    for number in range(1, max_steps + 1):
        try:
            resp = client.chat(
                alias, messages, tools=tools, temperature=0.0, max_tokens=max_tokens, cache=cache
            )
        except LLMError as e:
            raise stop("provider_error", f"step {number}: {e}") from e
        messages.append(resp.message or {"role": "assistant", "content": resp.text})

        if not resp.tool_calls:
            record(Step(number, resp, ()))
            if not resp.text.strip():
                raise stop(
                    "empty_answer",
                    f"step {number}: {alias} returned an empty answer "
                    f"(finish_reason={resp.finish_reason})",
                )
            return AgentResult(resp.text.strip(), tuple(steps), "answered")

        results = tuple(toolbox.run(call) for call in resp.tool_calls)
        messages.extend(r.message() for r in results)
        record(Step(number, resp, results))

        # Count at most one mistake per tool per step: two bad calls to the same tool in one
        # turn were made before the model saw either error, so they're one chance, not two.
        for name in {r.name for r in results if r.kind in MODEL_MISTAKES}:
            mistakes[name] = mistakes.get(name, 0) + 1
            if mistakes[name] >= 2:
                error = next(r.content for r in results if r.name == name)
                raise stop(
                    "repeated_mistake",
                    f"step {number}: {alias} called '{name}' wrongly again after being shown "
                    f"the error: {error}",
                )
        for r in results:
            if r.ok:
                mistakes.pop(r.name, None)

    raise stop("max_steps", f"no answer after {max_steps} steps")
