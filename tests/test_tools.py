import json

import pytest

from llm_lab.tools import Toolbox, ToolError, calculator_tool, safe_eval, shop_tools
from llm_lab.types import ToolCall


def call(name, arguments, id="call_1"):
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments)
    return ToolCall(id=id, name=name, arguments=arguments)


@pytest.fixture
def shop():
    return shop_tools()


# --- safe_eval --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("(129.99 + 45.5 + 12.75) * 0.9", 169.416),
        ("2 + 3 * 4", 14),
        ("(2 + 3) * 4", 20),
        ("-5 + 2", -3),
        ("10 / 4", 2.5),
        ("100 * (1 - 10 / 100)", 90),
    ],
)
def test_safe_eval_does_arithmetic(expression, expected):
    assert safe_eval(expression) == pytest.approx(expected)


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('echo pwned')",
        "open('/etc/passwd').read()",
        "x + 1",
        "(1).__class__",
        "9 ** 9 ** 9",  # exponentiation is excluded: it can hang the process
        "[1, 2, 3]",
        "True + 1",
        "'a' * 3",
    ],
)
def test_safe_eval_rejects_anything_but_arithmetic(expression):
    with pytest.raises(ToolError):
        safe_eval(expression)


def test_safe_eval_rejects_division_by_zero_long_input_and_bad_syntax():
    for bad in ["1 / 0", "1" + " + 1" * 100, "2 +* 3"]:
        with pytest.raises(ToolError):
            safe_eval(bad)


def test_calculator_hides_float_noise():
    result = Toolbox([calculator_tool()]).run(call("calculate", {"expression": "0.1 + 0.2"}))
    assert json.loads(result.content)["result"] == 0.3


# --- shop tools -----------------------------------------------------------------------------


def test_find_customer_is_case_insensitive(shop):
    result = shop.run(call("find_customer", {"email": " Jane@Example.com "}))
    assert result.ok
    assert json.loads(result.content) == {
        "customer_id": "C-101",
        "name": "Jane Doe",
        "loyalty_discount_pct": 10,
    }


def test_unknown_customer_is_a_tool_error_the_model_can_read(shop):
    result = shop.run(call("find_customer", {"email": "nobody@example.com"}))
    assert result.kind == "tool_error"
    assert "No customer with email" in json.loads(result.content)["error"]


def test_list_orders_filters_by_status(shop):
    result = shop.run(call("list_orders", {"customer_id": "C-101", "status": "delivered"}))
    orders = json.loads(result.content)["orders"]
    assert [o["order_id"] for o in orders] == ["O-5001", "O-5002", "O-5006"]


def test_list_orders_without_filter_returns_all_of_that_customers_orders(shop):
    result = shop.run(call("list_orders", {"customer_id": "C-101"}))
    orders = json.loads(result.content)["orders"]
    assert len(orders) == 6  # all 4 statuses, and never another customer's
    assert {o["status"] for o in orders} == {"placed", "shipped", "delivered", "cancelled"}


def test_the_demo_question_has_one_right_answer(shop):
    # What the agent should find: Jane's delivered orders, minus her 10% discount.
    orders = json.loads(
        shop.run(call("list_orders", {"customer_id": "C-101", "status": "delivered"})).content
    )["orders"]
    expression = f"({' + '.join(str(o['total_usd']) for o in orders)}) * (1 - 10 / 100)"
    total = json.loads(shop.run(call("calculate", {"expression": expression})).content)["result"]
    assert total == pytest.approx(169.416)


# --- Toolbox.run: everything the model can get wrong ------------------------------------


def test_unknown_tool_lists_the_real_ones(shop):
    result = shop.run(call("delete_customer", {"email": "jane@example.com"}))
    assert result.kind == "unknown_tool"
    assert "find_customer, list_orders, calculate" in result.content


def test_invalid_json_arguments(shop):
    result = shop.run(call("find_customer", '{"email": "jane@example.com"'))
    assert result.kind == "invalid_arguments" and "not valid JSON" in result.content


def test_arguments_that_are_not_an_object(shop):
    assert shop.run(call("find_customer", '["jane@example.com"]')).kind == "invalid_arguments"


def test_schema_errors_are_all_reported(shop):
    result = shop.run(call("list_orders", {"customer": "C-101", "status": "lost"}))
    assert result.kind == "invalid_arguments"
    message = json.loads(result.content)["error"]
    assert "missing required field 'customer_id'" in message
    assert "unexpected field 'customer'" in message
    assert "'lost' is not one of" in message


def test_empty_arguments_mean_an_empty_object(shop):
    result = shop.run(call("find_customer", ""))
    assert result.kind == "invalid_arguments" and "missing required field 'email'" in result.content


def test_a_crashing_tool_does_not_crash_the_agent_or_leak_details():
    from llm_lab.tools import Tool

    def broken(**kwargs):
        raise KeyError("secret internal detail")

    toolbox = Toolbox([Tool("broken", "d", {"type": "object"}, broken)])
    result = toolbox.run(call("broken", {}))
    assert result.kind == "tool_error"
    assert "KeyError" in result.content and "secret" not in result.content


def test_results_become_tool_messages_with_the_call_id(shop):
    result = shop.run(call("find_customer", {"email": "jane@example.com"}, id="call_42"))
    assert result.message() == {
        "role": "tool",
        "tool_call_id": "call_42",
        "content": result.content,
    }


# --- Toolbox setup ------------------------------------------------------------------------


def test_definitions_are_in_openai_format(shop):
    definitions = shop.definitions()
    assert [d["function"]["name"] for d in definitions] == shop.names()
    assert all(d["type"] == "function" for d in definitions)


def test_duplicate_names_and_unenforceable_schemas_fail_at_startup():
    from llm_lab.tools import Tool

    tool = calculator_tool()
    with pytest.raises(ValueError, match="duplicate"):
        Toolbox([tool, tool])
    bad = Tool("t", "d", {"type": "object", "properties": {"x": {"pattern": "a"}}}, lambda: 0)
    with pytest.raises(ValueError, match="unsupported"):
        Toolbox([bad])
