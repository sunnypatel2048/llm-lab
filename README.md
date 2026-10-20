# llm-lab

One Python client for three model providers: Google Gemini, Groq and a local Ollama model. Every call goes through the same `chat()` method and returns the same response shape, with per-model rate limiting, a response cache, retries with backoff, and cost tracking at paid rates.

It's the shared foundation for the rest of my AI engineering work: an agent loop, a RAG system with an eval harness, and a local-model benchmark all build on this client instead of calling SDKs directly.

## Quick start

Requires [uv](https://docs.astral.sh/uv/) and, for the local model, [Ollama](https://ollama.com).

```bash
git clone https://github.com/sunnypatel2048/llm-lab.git && cd llm-lab
uv sync
cp .env.example .env            # add GEMINI_API_KEY and GROQ_API_KEY
ollama pull qwen3.5:4b          # optional: the local model

uv run llm-lab "Explain BM25 in two sentences"
```

Options: `--models local-qwen,groq-oss` to pick models, `--cache refresh|off` to bypass the cache, `--max-tokens N`, `-v` to log every call and retry.

From Python:

```python
from llm_lab import LLMClient

client = LLMClient.from_env()
resp = client.chat("gemini-lite", [{"role": "user", "content": "Hi"}])
print(resp.text, resp.latency_ms, resp.cost_usd, resp.cached)
```

## Models

Defined in [`models.toml`](models.toml). Code refers to models by alias, so changing a model version is a one-line edit there.

| Alias | Provider | Model | Limits | $/1M tokens in / out |
|---|---|---|---|---|
| `gemini-flash` | Gemini | `gemini-3.8-flash` | 5/min, 20/day | 0.75 / 3.75 |
| `gemini-lite` | Gemini | `gemini-3.5-flash-lite` | 15/min, 500/day | 0.30 / 2.50 |
| `groq-oss` | Groq | `openai/gpt-oss-20b` | 30/min, 1,000/day | 0.075 / 0.30 |
| `local-qwen` | Ollama | `qwen3.5:4b`, thinking on | none | 0 / 0 |
| `local-qwen-nothink` | Ollama | `qwen3.5:4b`, thinking off | none | 0 / 0 |

Prices are the providers' paid rates even though I run on free tiers, so reported cost answers "what would this cost in production?". Limits are my free-tier quotas.

Optional per-model settings: `timeout_s` (seconds per attempt), `retry_on_timeout`, `reasoning_effort` (passed to the provider as-is; `"none"` turns thinking off on Ollama), and `max_attempts` (total tries per call; `gemini-flash` gets 2 instead of the default 4).

## Design decisions

**One interface, one SDK.** All three providers expose an OpenAI-compatible Chat Completions endpoint, so a single SDK reaches them by changing `base_url`. Only [`transport.py`](src/llm_lab/transport.py) imports that SDK; the rest of the code depends on my own types. The tradeoff: the compatible API only covers features the providers share, so a provider-specific feature would need its own adapter in the transport layer.

**Throttle before sending, not after failing.** Free tiers are small (`gemini-flash` allows 20 requests a day), so a 429 wastes real quota. Each model gets a token bucket for its per-minute limit and a persisted counter for its daily limit, and the client waits *before* sending. Every retry takes a fresh slot, because a retry is a real request. The daily counter rolls over at midnight Pacific.

**Cache every answer.** Responses are stored in SQLite, keyed by a SHA-256 hash of provider, model ID, messages, temperature and max tokens. Re-running an eval or a demo costs no quota. Errors and empty answers are never cached, so a blocked or truncated response can't hide on later runs. A side effect: a repeated prompt is answered from the cache even when the provider is down.

**Retry only what can succeed.** Rate limits (429), server errors (5xx) and timeouts are retried with exponential backoff and jitter, waiting at least as long as the provider's `Retry-After`. Calls get 4 attempts by default, but every retry is a real request against the daily quota, so `gemini-flash` (20 a day) gets 2. Bad requests, missing keys, an unreachable Ollama and exhausted daily quotas fail immediately with a one-line message. Local models don't retry timeouts at all: a remote timeout is often a network blip, but a slow local generation will be just as slow the second time, so they get a long `timeout_s` instead. The SDK's own retries are turned off so attempts aren't silently multiplied underneath mine.

**Report unknowns, don't guess.** If a provider doesn't return token counts, tokens and cost are `None`, never estimated. Every response records `finish_reason`, which is how truncated answers get caught (see results).

**Fail fast on configuration.** `models.toml` is validated when it loads, and a missing API key fails before any request, naming the variable to set. Keys are checked only for the models a run uses, so CI and local-only runs need no keys.

## Results

Same prompt, every model, run on a MacBook Air (Apple Silicon, 16 GB):

```
uv run llm-lab "Explain BM25 in two sentences" --cache refresh
```

| Model | Status | Latency | Tokens in / out | Thinking tokens | Cost per call | Finish |
|---|---|---|---|---|---|---|
| `gemini-flash` | ok | 23,989 ms | 8 / 70 | not reported | $0.000269 | stop |
| `gemini-lite` | ok | 759 ms | 8 / 55 | not reported | $0.000140 | stop |
| `groq-oss` | ok | 557 ms | 77 / 172 | 109 | $0.000057 | stop |
| `local-qwen` | empty | 96,047 ms | 18 / 2,048 | not reported | $0 | length |
| `local-qwen-nothink` | ok | 3,641 ms | 20 / 69 | thinking off | $0 | stop |

One run each, so treat the latencies as a snapshot, not a benchmark.

To see how long `local-qwen` needs to finish with thinking on, I ran it directly without a token cap (`ollama run qwen3.5:4b --verbose`): 3,536 output tokens at 22.4 tokens/s, 2 min 41 s in total. Prompt processing took 0.2 s; almost all of the time was generating.

**What surprised me**

- **Thinking dominates local cost.** The same model gave a two-sentence answer in 69 tokens and 3.6 s with thinking off. With thinking on, it needed about 3,500 tokens and 2 min 41 s: roughly 50 times the tokens and 44 times the time. Capped at `max_tokens=2048`, it ran for 96 s and returned nothing (`finish_reason="length"`), because the cap covers the hidden reasoning as well as the answer. Earlier, a five-word answer needed 2048 tokens; at 512 and 1024 it also came back empty.
- **A timeout that's harmless for an API was the wrong thing to retry locally.** With the original 60 s timeout, `local-qwen` never finished, and each retry restarted the same 2.5-minute generation from scratch. Ollama logged a `500` for every attempt, but that was only my client hanging up. The fix was a per-model `timeout_s`, no timeout retries for local models, and the `local-qwen-nothink` alias to measure what thinking costs.
- **"Thinking tokens" is only as good as what the provider reports.** Groq reports them (109 of 172 output tokens were reasoning). Ollama doesn't, through the OpenAI-compatible API, so a blank in that column doesn't mean no thinking: all 2,048 of `local-qwen`'s output tokens were reasoning. `gemini-lite` reported none either, which could mean it didn't think or that the count isn't exposed.
- **Free tiers are noisy.** Groq answered in 0.6 s, the local model in 3.6 s, and `gemini-lite` took 10.8 s for 55 tokens, while `gemini-flash` returned 503 "high demand" on all 4 attempts in both of that day's runs. Every retry is a real request, so those failures used 8 of its 20 daily requests without producing an answer. So I made the number of attempts a per-model setting and gave `gemini-flash` 2: one retry still rides out a brief blip, but a sustained outage now costs 2 requests instead of 4.

## Testing

```bash
uv run pytest -q          # no network or API keys needed
uv run ruff check && uv run ruff format --check
```

Tests use fake transports and fake clocks, so retries, rate limits and quota rollovers run in milliseconds. CI runs lint, format and tests on every push and pull request.

## Layout

```
models.toml               model registry: aliases, providers, prices, limits
src/llm_lab/
  client.py               LLMClient.chat(): cache → rate limit → call → retry → cost
  transport.py            the only code that touches the OpenAI SDK; error translation
  ratelimit.py            per-minute token bucket, persisted daily counter
  cache.py                SQLite response cache
  registry.py             loads and validates models.toml
  types.py                LLMResponse and error types
  smoke.py                the `llm-lab` command
tests/                    one test file per module
```

## Limitations

- Rate limits and quota counts are per process. Several processes or machines sharing one quota would need a shared store such as Redis with an atomic Lua script.
- No streaming, so time-to-first-token isn't measured yet.
- Tool calling isn't wired through yet; that comes with the agent loop.
