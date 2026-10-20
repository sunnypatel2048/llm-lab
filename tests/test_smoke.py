from llm_lab.cache import ResponseCache
from llm_lab.client import LLMClient
from llm_lab.registry import Registry
from llm_lab.smoke import render, run
from llm_lab.transport import RawCompletion
from llm_lab.types import ProviderUnavailable


class ScriptedTransport:
    """Returns a fixed result per alias; an Exception is raised instead."""

    def __init__(self, by_alias):
        self.by_alias = by_alias
        self.calls = 0

    def __call__(self, spec, messages, *, temperature, max_tokens, timeout, tools=None):
        self.calls += 1
        item = self.by_alias[spec.alias]
        if isinstance(item, Exception):
            raise item
        return item


def client(by_alias, cache=None):
    registry = Registry.from_dict(
        {
            "providers": {"ollama": {"base_url": "http://localhost:11434/v1"}},
            "models": {
                alias: {
                    "provider": "ollama",
                    "model_id": f"{alias}-id",
                    "price_in_per_m": 1.0,
                    "price_out_per_m": 2.0,
                }
                for alias in by_alias
            },
        }
    )
    transport = ScriptedTransport(by_alias)
    return LLMClient(registry, transport=transport, cache=cache), transport


ANSWER = RawCompletion("BM25 scores documents by term frequency. " * 5, 100, 50, 30, "stop")


def test_one_failure_does_not_stop_the_others():
    c, _ = client({"a": ProviderUnavailable("a: Is Ollama running?"), "b": ANSWER})
    rows = run(c, ["a", "b"], "q")
    assert [row.status for row in rows] == ["error: ProviderUnavailable", "ok"]


def test_table_has_header_numbers_and_full_error_messages():
    c, _ = client({"a": ProviderUnavailable("a: Is Ollama running? `ollama serve`"), "b": ANSWER})
    text = render(run(c, ["a", "b"], "q"))
    lines = text.splitlines()
    assert lines[0].split()[:3] == ["model", "status", "latency"]
    assert "$0.000200" in text  # 100*1/1M + 50*2/1M
    assert "Errors:" in text and "`ollama serve`" in text  # never truncated


def test_long_answers_are_previewed_on_one_line():
    c, _ = client({"b": ANSWER})
    row_line = render(run(c, ["b"], "q")).splitlines()[2]
    assert row_line.endswith("…")
    assert len(row_line.split("  ")[-1]) <= 60


def test_cached_rows_are_labelled(tmp_path):
    c, transport = client({"b": ANSWER}, cache=ResponseCache(tmp_path / "c.sqlite"))
    run(c, ["b"], "q")
    rows = run(c, ["b"], "q")
    assert rows[0].status == "cached" and transport.calls == 1
    assert "original call" in render(rows)


def test_empty_answer_is_flagged():
    c, _ = client({"b": RawCompletion("", 20, 512, 512, "length")})
    rows = run(c, ["b"], "q")
    assert rows[0].status == "empty"
    assert "length" in render(rows)


def test_unknown_alias_is_reported_not_raised():
    c, _ = client({"b": ANSWER})
    rows = run(c, ["nope", "b"], "q")
    assert rows[0].status == "error: ConfigError" and rows[1].status == "ok"
