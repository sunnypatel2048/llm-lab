"""Transport tests against the real `openai` SDK classes (no network: clients and errors are
constructed locally, and the HTTP call is replaced by a fake)."""

from types import SimpleNamespace

import httpx2
import openai
import pytest

from llm_lab.registry import ModelSpec
from llm_lab.transport import (
    OpenAICompatTransport,
    make_openai_client,
    parse_completion,
    translate_error,
)
from llm_lab.types import (
    BadRequest,
    DailyQuotaExhausted,
    LLMTimeout,
    ProviderError,
    ProviderUnavailable,
    RateLimited,
)

REQUEST = httpx2.Request("POST", "https://example.test/v1/chat/completions")


def spec(provider="gemini", alias="gemini-flash") -> ModelSpec:
    return ModelSpec(
        alias=alias,
        provider=provider,
        model_id="model-x",
        base_url="https://example.test/v1",
        api_key_env=None,
        price_in_per_m=0.0,
        price_out_per_m=0.0,
    )


def status_error(cls, status, message="error", headers=None):
    response = httpx2.Response(status, request=REQUEST, headers=headers or {})
    return cls(message, response=response, body=None)


# --- error translation ----------------------------------------------------------------


def test_timeout_is_a_timeout_not_a_connection_error():
    # APITimeoutError subclasses APIConnectionError; checking in the wrong order would make
    # every timeout "unavailable" and never retried.
    err = translate_error(openai.APITimeoutError(request=REQUEST), spec())
    assert isinstance(err, LLMTimeout) and err.retryable


def test_connection_error_to_ollama_suggests_starting_it():
    err = translate_error(openai.APIConnectionError(request=REQUEST), spec(provider="ollama"))
    assert isinstance(err, ProviderUnavailable) and not err.retryable
    assert "ollama serve" in str(err)


def test_429_becomes_rate_limited_with_retry_after():
    e = status_error(openai.RateLimitError, 429, "slow down", {"retry-after": "7"})
    err = translate_error(e, spec())
    assert isinstance(err, RateLimited) and err.retry_after == 7.0


def test_429_without_retry_after_header():
    err = translate_error(status_error(openai.RateLimitError, 429), spec())
    assert isinstance(err, RateLimited) and err.retry_after is None


@pytest.mark.parametrize(
    "message",
    [
        "Quota exceeded for metric: generate_content_requests_per_model_per_day",
        "You exceeded your daily request limit",
    ],
)
def test_429_for_daily_quota_is_not_retried(message):
    err = translate_error(status_error(openai.RateLimitError, 429, message), spec())
    assert isinstance(err, DailyQuotaExhausted) and not err.retryable


@pytest.mark.parametrize("status", [500, 502, 503])
def test_5xx_becomes_provider_error(status):
    err = translate_error(status_error(openai.InternalServerError, status), spec())
    assert isinstance(err, ProviderError) and err.retryable


@pytest.mark.parametrize(
    ("cls", "status"),
    [
        (openai.BadRequestError, 400),
        (openai.AuthenticationError, 401),
        (openai.NotFoundError, 404),
    ],
)
def test_other_4xx_become_bad_request(cls, status):
    err = translate_error(status_error(cls, status, "model not found"), spec())
    assert isinstance(err, BadRequest) and not err.retryable
    assert f"HTTP {status}" in str(err)


# --- response parsing -----------------------------------------------------------------


def completion(content="answer", usage=True, reasoning=None, finish="stop"):
    details = SimpleNamespace(reasoning_tokens=reasoning) if reasoning is not None else None
    u = (
        SimpleNamespace(prompt_tokens=10, completion_tokens=20, completion_tokens_details=details)
        if usage
        else None
    )
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message, finish_reason=finish)
    return SimpleNamespace(choices=[choice], usage=u)


def test_parse_reads_text_usage_and_reasoning():
    raw = parse_completion(completion(reasoning=15))
    assert (raw.text, raw.input_tokens, raw.output_tokens) == ("answer", 10, 20)
    assert raw.reasoning_tokens == 15 and raw.finish_reason == "stop"


def test_parse_missing_usage_and_null_content():
    raw = parse_completion(completion(content=None, usage=False, finish="length"))
    assert raw.text == "" and raw.input_tokens is None and raw.output_tokens is None
    assert raw.finish_reason == "length"


# --- the transport itself -------------------------------------------------------------


def test_sdk_client_has_its_own_retries_turned_off():
    client = make_openai_client(spec(provider="ollama"))
    assert client.max_retries == 0


def test_transport_sends_the_right_request_and_reuses_one_client_per_provider():
    sent, built = [], []

    def create(**kwargs):
        sent.append(kwargs)
        return completion()

    def factory(s):
        built.append(s.provider)
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    transport = OpenAICompatTransport(client_factory=factory)
    msgs = [{"role": "user", "content": "hi"}]
    for _ in range(2):
        transport(spec(), msgs, temperature=0.0, max_tokens=64, timeout=9)

    assert built == ["gemini"]
    assert sent[0] == dict(
        model="model-x", messages=msgs, temperature=0.0, max_tokens=64, timeout=9
    )


def test_transport_translates_sdk_errors():
    def create(**kwargs):
        raise status_error(openai.RateLimitError, 429, "slow down")

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    transport = OpenAICompatTransport(client_factory=lambda s: client)
    with pytest.raises(RateLimited) as excinfo:
        transport(spec(), [], temperature=0.0, max_tokens=8, timeout=1)
    assert isinstance(excinfo.value.__cause__, openai.RateLimitError)  # original kept for debugging
