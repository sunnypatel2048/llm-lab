"""One real tool-calling round trip per model: ask -> tool call -> send result back -> answer.

    uv run scripts/check_tools.py                      # groq-oss, gemini-lite, local-qwen-nothink
    uv run scripts/check_tools.py --models groq-oss

Checks the two things fake transports can't: that each provider returns tool calls in the
format we parse, and that it accepts our echoed assistant turn on the follow-up request.
Uses cache="off" so every run talks to the real API (2 requests per model).
"""

import argparse
import json
import sys

from llm_lab import LLMClient, LLMError

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_order_total",
            "description": "Return the total in USD for an order ID.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        },
    }
]


def check(client: LLMClient, alias: str) -> bool:
    messages = [{"role": "user", "content": "What is the total for order A-1001? Use the tool."}]
    first = client.chat(alias, messages, tools=TOOLS, cache="off")
    if not first.tool_calls:
        print(f"{alias}: FAIL, answered without calling the tool: {first.text[:80]!r}")
        return False
    call = first.tool_calls[0]
    print(f"{alias}: called {call.name}({call.arguments})")
    messages.append(first.message)
    for c in first.tool_calls:  # answer every call, even if the model made several
        result = {"order_id": "A-1001", "total_usd": 42.5}
        messages.append({"role": "tool", "tool_call_id": c.id, "content": json.dumps(result)})
    second = client.chat(alias, messages, tools=TOOLS, cache="off")
    ok = "42.5" in second.text
    print(f"{alias}: {'OK' if ok else 'CHECK'}, answered {second.text.strip()[:80]!r}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="groq-oss,gemini-lite,local-qwen-nothink")
    args = parser.parse_args()
    client = LLMClient.from_env()
    results = []
    for alias in args.models.split(","):
        try:
            results.append(check(client, alias.strip()))
        except LLMError as e:
            print(f"{alias}: ERROR {e}")
            results.append(False)
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
