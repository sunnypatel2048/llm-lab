"""Tools the agent can call, and the Toolbox that runs them safely.

Toolbox.run() never raises for anything the model did wrong: an unknown tool, arguments that
aren't JSON, arguments that break the schema, or a tool that fails all come back as a
ToolResult with ok=False and a message written for the model. The agent loop decides what to
do with each kind (see `kind`).
"""

from __future__ import annotations

import ast
import json
import math
import operator
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from llm_lab.schema import check_schema, validate
from llm_lab.types import ToolCall

SHOP_DATA = Path(__file__).parent / "data" / "shop.json"

ResultKind = Literal["ok", "invalid_arguments", "unknown_tool", "tool_error"]


class ToolError(Exception):
    """An expected failure inside a tool (e.g. "no customer with that email"). The message is
    shown to the model, so write it for the model."""


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema for the arguments object
    fn: Callable[..., Any]

    def definition(self) -> dict[str, Any]:
        """The OpenAI-format definition sent to the model."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass(frozen=True)
class ToolResult:
    call_id: str
    name: str
    kind: ResultKind
    content: str  # JSON text sent back to the model as the tool message

    @property
    def ok(self) -> bool:
        return self.kind == "ok"

    def message(self) -> dict[str, Any]:
        """The role="tool" message to append to the conversation."""
        return {"role": "tool", "tool_call_id": self.call_id, "content": self.content}


class Toolbox:
    def __init__(self, tools: list[Tool]):
        names = [t.name for t in tools]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(f"duplicate tool names: {', '.join(duplicates)}")
        for tool in tools:
            if tool.parameters.get("type") != "object":
                raise ValueError(f"{tool.name}: parameters must be a JSON Schema of type object")
            check_schema(tool.parameters, f"{tool.name}.parameters")  # fail at startup, not mid-run
        self._tools = {t.name: t for t in tools}

    def definitions(self) -> list[dict[str, Any]]:
        return [t.definition() for t in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools)

    def run(self, call: ToolCall) -> ToolResult:
        def fail(kind: ResultKind, message: str) -> ToolResult:
            return ToolResult(call.id, call.name, kind, json.dumps({"error": message}))

        tool = self._tools.get(call.name)
        if tool is None:
            available = ", ".join(self._tools)
            return fail("unknown_tool", f"No tool named '{call.name}'. Available: {available}.")

        try:
            args = json.loads(call.arguments) if call.arguments.strip() else {}
        except json.JSONDecodeError as e:
            return fail("invalid_arguments", f"Arguments are not valid JSON ({e.msg}).")
        if not isinstance(args, dict):
            return fail("invalid_arguments", "Arguments must be a JSON object.")
        errors = validate(args, tool.parameters)
        if errors:
            return fail("invalid_arguments", "Invalid arguments: " + "; ".join(errors) + ".")

        try:
            result = tool.fn(**args)
        except ToolError as e:
            return fail("tool_error", str(e))
        except Exception as e:  # a bug in the tool: report it without leaking a traceback
            return fail("tool_error", f"{call.name} failed unexpectedly ({type(e).__name__}).")
        return ToolResult(call.id, call.name, "ok", json.dumps(result))


# --- calculator -------------------------------------------------------------------------

MAX_EXPRESSION_CHARS = 200

_BINARY_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def safe_eval(expression: str) -> float:
    """Evaluate arithmetic without eval(): numbers, + - * /, unary minus and parentheses only.

    The expression is parsed into a syntax tree and only those node types are evaluated, so
    names, function calls, attribute access and imports are rejected before anything runs.
    Exponentiation is left out on purpose: 9**9**9 is one short string that hangs the process.
    """
    if len(expression) > MAX_EXPRESSION_CHARS:
        raise ToolError(f"Expression is longer than {MAX_EXPRESSION_CHARS} characters.")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        raise ToolError(f"Could not parse {expression!r} as arithmetic.") from None

    def evaluate(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPS:
            left, right = evaluate(node.left), evaluate(node.right)
            if isinstance(node.op, ast.Div) and right == 0:
                raise ToolError("Division by zero.")
            return _BINARY_OPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
            return _UNARY_OPS[type(node.op)](evaluate(node.operand))
        raise ToolError(
            f"Unsupported syntax ({type(node).__name__}). "
            "Use numbers, + - * /, and parentheses only."
        )

    result = evaluate(tree)
    if not math.isfinite(result):
        raise ToolError("Result is not a finite number.")
    return result


def calculator_tool() -> Tool:
    def calculate(expression: str) -> dict[str, Any]:
        # Rounded to 10 places to hide float noise (0.1 + 0.2), not to round money.
        return {"expression": expression, "result": round(safe_eval(expression), 10)}

    return Tool(
        name="calculate",
        description=(
            "Evaluate an arithmetic expression exactly. Use this for every calculation instead "
            "of doing math yourself. Supports numbers, + - * / and parentheses."
        ),
        parameters={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "minLength": 1,
                    "description": "For example: (129.99 + 45.5) * 0.9",
                }
            },
            "required": ["expression"],
            "additionalProperties": False,
        },
        fn=calculate,
    )


# --- shop tools (demo data) -------------------------------------------------------------

ORDER_STATUSES = ["placed", "shipped", "delivered", "cancelled"]


def shop_tools(path: str | Path = SHOP_DATA) -> Toolbox:
    """find_customer, list_orders and calculate over the synthetic shop data."""
    data = json.loads(Path(path).read_text())
    customers = {c["customer_id"]: c for c in data["customers"]}

    def find_customer(email: str) -> dict[str, Any]:
        wanted = email.strip().lower()
        for c in customers.values():
            if c["email"].lower() == wanted:
                return {
                    "customer_id": c["customer_id"],
                    "name": c["name"],
                    "loyalty_discount_pct": c["loyalty_discount_pct"],
                }
        raise ToolError(f"No customer with email {email!r}.")

    def list_orders(customer_id: str, status: str | None = None) -> dict[str, Any]:
        if customer_id not in customers:
            raise ToolError(f"No customer with id {customer_id!r}. Look it up with find_customer.")
        orders = [
            {k: o[k] for k in ("order_id", "status", "total_usd", "placed_on")}
            for o in data["orders"]
            if o["customer_id"] == customer_id and (status is None or o["status"] == status)
        ]
        return {"customer_id": customer_id, "status_filter": status, "orders": orders}

    return Toolbox(
        [
            Tool(
                name="find_customer",
                description=(
                    "Look up a customer by email. Returns their customer_id, name and "
                    "loyalty_discount_pct (a percentage, e.g. 10 means 10% off)."
                ),
                parameters={
                    "type": "object",
                    "properties": {"email": {"type": "string", "minLength": 3}},
                    "required": ["email"],
                    "additionalProperties": False,
                },
                fn=find_customer,
            ),
            Tool(
                name="list_orders",
                description=(
                    "List a customer's orders with their totals in USD. Optionally filter by "
                    "status. Needs the customer_id from find_customer."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "customer_id": {"type": "string", "minLength": 1},
                        "status": {"type": "string", "enum": ORDER_STATUSES},
                    },
                    "required": ["customer_id"],
                    "additionalProperties": False,
                },
                fn=list_orders,
            ),
            calculator_tool(),
        ]
    )
