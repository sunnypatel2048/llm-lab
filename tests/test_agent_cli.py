import pytest
from test_agent import HAPPY_PATH, answer, calls, make

from llm_lab.agent_cli import contains_expected, main
from llm_lab.cache import ResponseCache


@pytest.mark.parametrize(
    "text",
    [
        "Jane has spent $169.42 after her discount.",
        "The total is 169.42 USD",
        "Total: $169.42.",
        "169.42",
    ],
)
def test_expected_value_found_in_common_formats(text):
    assert contains_expected(text, "169.42")


@pytest.mark.parametrize("text", ["$169.416", "$2169.42", "$169.40", "about 169", ""])
def test_expected_value_not_found_in_lookalikes(text):
    assert not contains_expected(text, "169.42")


def test_thousands_separators_are_ignored():
    assert contains_expected("$1,169.42", "1169.42")


def test_happy_path_prints_trace_totals_and_check(capsys):
    client, _ = make(*HAPPY_PATH)
    code = main(["--model", "m", "--expect", "169.42"], client=client)
    out = capsys.readouterr().out
    assert code == 0
    assert "step 1  find_customer(" in out and "step 4  answer:" in out
    assert "4 steps · 3 tool calls · 0 repairs" in out
    assert "check: PASS (expected 169.42)" in out


def test_wrong_answer_fails_the_check(capsys):
    client, _ = make(*HAPPY_PATH[:3], answer("Jane spent $188.24."))  # forgot the discount
    assert main(["--model", "m", "--expect", "169.42"], client=client) == 1
    assert "check: FAIL" in capsys.readouterr().out


def test_model_mistakes_are_flagged_in_the_trace(capsys):
    client, _ = make(calls(("find_customer", {"mail": "jane@example.com"})), *HAPPY_PATH)
    main(["--model", "m"], client=client)
    out = capsys.readouterr().out
    assert "!! invalid_arguments" in out and "1 repairs" in out


def test_early_stop_prints_reason_and_cost_so_far(capsys):
    client, _ = make(
        calls(("find_customer", {"mail": "x"})),
        calls(("find_customer", {"mail": "x"})),
    )
    assert main(["--model", "m"], client=client) == 1
    out = capsys.readouterr().out
    assert "stopped (repeated_mistake)" in out
    assert "2 steps · 2 tool calls · 2 repairs" in out


def test_cached_steps_are_marked(capsys, tmp_path):
    client, _ = make(*HAPPY_PATH, cache=ResponseCache(tmp_path / "c.sqlite"))
    main(["--model", "m"], client=client)
    capsys.readouterr()
    main(["--model", "m"], client=client)
    assert "cached]" in capsys.readouterr().out


def test_unknown_model_is_a_one_line_error(capsys):
    client, _ = make()
    assert main(["--model", "nope"], client=client) == 1
    assert "Unknown model alias 'nope'" in capsys.readouterr().out
