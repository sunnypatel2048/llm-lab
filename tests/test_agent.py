"""Agent loop tests: a fake transport plays back scripted model turns; the tools are real."""

import json

import pytest

from llm_lab.agent import AgentError, run_agent
from llm_lab.cache import ResponseCache
from llm_lab.client import LLMClient
from llm_lab.registry import Registry
from llm_lab.tools import shop_tools
from llm_lab.transport import RawCompletion
from llm_lab.types import BadRequest, ToolCall

QUESTION = "How much has jane@example.com spent on delivered orders, after her loyalty discount?"
DELIVERED_TOTAL = "(129.99 + 45.5 + 12.75) * (1 - 10 / 100)"


class ScriptedModel:
    """Fake transport: returns the next scripted turn and records what the model was sent."""

    def __init__(self, *turns):
        self.turns = list(turns)
        self.requests = []  # the messages list sent on each call

    def __call__(self, spec, messages, *, temperature, max_tokens, timeout, tools=None):
        self.requests.append([dict(m) for m in messages])
        turn = self.turns.pop(0)
        if isinstance(turn, Exception):
            raise turn
        return turn


_ids = iter(range(1, 10_000))


def calls(*specs, cost_tokens=(100, 20)):
    """A model turn that calls tools: calls(("find_customer", {...}), ...)."""
    tool_calls = []
    for name, args in specs:
        arguments = args if isinstance(args, str) else json.dumps(args)
        tool_calls.append(ToolCall(id=f"call_{next(_ids)}", name=name, arguments=arguments))
    message = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}}
            for c in tool_calls
        ],
    }
    return RawCompletion("", *cost_tokens, None, "tool_calls", tuple(tool_calls), message)


def answer(text, tokens=(300, 15)):
    return RawCompletion(text, *tokens, None, "stop", (), {"role": "assistant", "content": text})


HAPPY_PATH = [
    calls(("find_customer", {"email": "jane@example.com"})),
    calls(("list_orders", {"customer_id": "C-101", "status": "delivered"})),
    calls(("calculate", {"expression": DELIVERED_TOTAL})),
    answer("Jane has spent $169.42 on delivered orders after her discount."),
]


def make(*turns, cache=None):
    registry = Registry.from_dict(
        {
            "providers": {"p": {"base_url": "https://p.test/"}},
            "models": {
                "m": {
                    "provider": "p",
                    "model_id": "m-1",
                    "price_in_per_m": 1.0,
                    "price_out_per_m": 2.0,
                    "max_attempts": 1,
                }
            },
        }
    )
    model = ScriptedModel(*turns)
    return LLMClient(registry, transport=model, cache=cache), model


def run(client, **kwargs):
    return run_agent(client, "m", QUESTION, shop_tools(), **kwargs)


# --- the happy path ---------------------------------------------------------------------


def test_three_tool_steps_then_an_answer():
    client, model = make(*HAPPY_PATH)
    result = run(client)
    assert result.stop_reason == "answered"
    assert result.answer.endswith("$169.42 on delivered orders after her discount.")
    assert len(result.steps) == 4 and result.tool_calls == 3 and result.repairs == 0


def test_each_tool_result_reaches_the_model_on_the_next_call():
    client, model = make(*HAPPY_PATH)
    run(client)
    after_first = model.requests[1]
    assert after_first[-2]["role"] == "assistant" and after_first[-2]["tool_calls"]
    tool_message = after_first[-1]
    assert tool_message["role"] == "tool"
    assert tool_message["tool_call_id"] == after_first[-2]["tool_calls"][0]["id"]
    assert json.loads(tool_message["content"])["customer_id"] == "C-101"
    calculated = json.loads(model.requests[3][-1]["content"])
    assert calculated["result"] == pytest.approx(169.416)


def test_conversation_starts_with_system_prompt_and_question():
    client, model = make(*HAPPY_PATH)
    run(client)
    first = model.requests[0]
    assert [m["role"] for m in first] == ["system", "user"]
    assert "calculate tool for every arithmetic step" in first[0]["content"]
    assert first[1]["content"] == QUESTION


def test_totals_add_up_across_steps():
    client, _ = make(*HAPPY_PATH)
    result = run(client)
    assert result.input_tokens == 3 * 100 + 300
    assert result.output_tokens == 3 * 20 + 15
    assert result.cost_usd == pytest.approx((600 * 1.0 + 75 * 2.0) / 1_000_000)
    assert result.cost_complete


def test_several_tool_calls_in_one_turn_all_run_in_one_step():
    client, model = make(
        calls(
            ("find_customer", {"email": "jane@example.com"}),
            ("calculate", {"expression": "1 + 1"}),
        ),
        answer("Done: $2.00"),
    )
    result = run(client)
    assert len(result.steps) == 2 and len(result.steps[0].results) == 2
    assert [m["role"] for m in model.requests[1][-3:]] == ["assistant", "tool", "tool"]


def test_on_step_sees_every_step_as_it_happens():
    seen = []
    client, _ = make(*HAPPY_PATH)
    run(client, on_step=lambda step: seen.append(step.number))
    assert seen == [1, 2, 3, 4]


# --- repair -----------------------------------------------------------------------------


def test_one_bad_call_is_shown_to_the_model_and_fixed():
    client, model = make(
        calls(("find_customer", {"mail": "jane@example.com"})),  # wrong field name
        *HAPPY_PATH,
    )
    result = run(client)
    assert result.stop_reason == "answered" and result.repairs == 1
    error = json.loads(model.requests[1][-1]["content"])["error"]
    assert "missing required field 'email'" in error and "unexpected field 'mail'" in error


def test_second_mistake_on_the_same_tool_stops_the_run():
    client, _ = make(
        calls(("find_customer", {"mail": "jane@example.com"})),
        calls(("find_customer", '{"email": ')),  # still broken
        *HAPPY_PATH,
    )
    with pytest.raises(AgentError, match="called 'find_customer' wrongly again") as excinfo:
        run(client)
    partial = excinfo.value.result
    assert partial.stop_reason == "repeated_mistake" and len(partial.steps) == 2
    assert partial.cost_usd > 0  # the cost so far is still reported


def test_a_success_resets_the_mistake_count():
    client, _ = make(
        calls(("list_orders", {"customer": "C-101"})),
        calls(("list_orders", {"customer_id": "C-101"})),  # fixed
        calls(("list_orders", {"customer_id": "C-101", "status": "done"})),  # new, different slip
        calls(("list_orders", {"customer_id": "C-101", "status": "delivered"})),
        answer("$188.24"),
    )
    assert run(client).stop_reason == "answered"


def test_two_bad_calls_in_one_turn_count_as_one_mistake():
    client, _ = make(
        calls(("calculate", {"expr": "1"}), ("calculate", {"expr": "2"})),
        calls(("calculate", {"expression": "1 + 1"})),
        answer("$2.00"),
    )
    assert run(client).stop_reason == "answered"


def test_an_unknown_tool_is_a_mistake_the_model_can_fix():
    client, model = make(
        calls(("lookup_customer", {"email": "jane@example.com"})),
        *HAPPY_PATH,
    )
    result = run(client)
    assert result.stop_reason == "answered"
    assert "No tool named 'lookup_customer'" in model.requests[1][-1]["content"]


def test_tool_errors_on_valid_input_are_not_mistakes():
    # "No customer with that email" twice is the data talking, not a broken call.
    client, _ = make(
        calls(("find_customer", {"email": "jane@exmaple.com"})),
        calls(("find_customer", {"email": "jane@exmple.com"})),
        *HAPPY_PATH,
    )
    result = run(client)
    assert result.stop_reason == "answered" and result.repairs == 0


# --- other ways to stop -----------------------------------------------------------------------


def test_max_steps_stops_a_model_that_never_answers():
    looping = [calls(("calculate", {"expression": "1 + 1"})) for _ in range(10)]
    client, model = make(*looping)
    with pytest.raises(AgentError, match="no answer after 3 steps") as excinfo:
        run(client, max_steps=3)
    assert excinfo.value.result.stop_reason == "max_steps"
    assert len(model.requests) == 3


def test_provider_error_keeps_the_trace_so_far():
    client, _ = make(HAPPY_PATH[0], BadRequest("m: HTTP 400: bad request"))
    with pytest.raises(AgentError, match="step 2: m: HTTP 400") as excinfo:
        run(client)
    partial = excinfo.value.result
    assert partial.stop_reason == "provider_error" and len(partial.steps) == 1


def test_empty_final_answer_is_an_error():
    client, _ = make(*HAPPY_PATH[:3], RawCompletion("", 300, 2048, None, "length"))
    with pytest.raises(AgentError, match="empty answer") as excinfo:
        run(client)
    assert excinfo.value.result.stop_reason == "empty_answer"


def test_unknown_usage_makes_the_cost_incomplete_not_zero():
    client, _ = make(
        calls(("find_customer", {"email": "jane@example.com"})),
        RawCompletion("$0.00", None, None, None, "stop", (), None),
    )
    result = run(client)
    assert not result.cost_complete
    assert result.cost_usd == pytest.approx((100 * 1.0 + 20 * 2.0) / 1_000_000)


# --- caching ------------------------------------------------------------------------------


def test_a_rerun_replays_from_the_cache_without_calling_the_model(tmp_path):
    cache = ResponseCache(tmp_path / "c.sqlite")
    client, model = make(*HAPPY_PATH, cache=cache)
    first = run(client)
    second = run(client)
    assert len(model.requests) == 4  # only the first run reached the model
    assert second.answer == first.answer
    assert all(s.response.cached for s in second.steps)
