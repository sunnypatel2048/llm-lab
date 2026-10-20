# llm-lab

One Python client for three model providers: Google Gemini, Groq and a local Ollama model. Every call goes through the same `chat()` method and returns the same response shape, with per-model rate limiting, a response cache, retries with backoff, and cost tracking at paid rates.

It's the shared foundation for the rest of my AI engineering work. A tool-calling agent built on it is included below, and the RAG system and local-model benchmark I'm building next will use it too, instead of calling SDKs directly.

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
- **Free tiers are noisy.** On my first run, `gemini-lite` took 10.8 s for 55 tokens (759 ms on the rerun in the table), while `gemini-flash` returned 503 "high demand" on all 4 attempts in both of that day's runs; when it did answer later, it took 24 s. Every retry is a real request, so those failures used 8 of its 20 daily requests without producing an answer. So I made the number of attempts a per-model setting and gave `gemini-flash` 2: one retry still rides out a brief blip, but a sustained outage now costs 2 requests instead of 4.

## Agent: a tool-calling loop

A small agent written without a framework, on top of the client: [`agent.py`](src/llm_lab/agent.py) is about 150 lines. It answers questions about a synthetic shop ([`data/shop.json`](src/llm_lab/data/shop.json)) using three tools:

| Tool | Does |
|---|---|
| `find_customer(email)` | customer ID, name and loyalty discount |
| `list_orders(customer_id, status?)` | a customer's orders and totals, optionally filtered by status |
| `calculate(expression)` | exact arithmetic |

```bash
uv run llm-lab-agent --model groq-oss --expect 169.42
```

The demo question, "How much has jane@example.com spent on delivered orders, after her loyalty discount?", takes three dependent tool calls: the customer ID and discount from the first feed the second and third. There's one right answer ($169.42), so `--expect` turns a run into a pass/fail check.

**How it works.** Each step is one model call with the conversation and the tool definitions. If the model asks for tools, all of them run, each result goes back as a `tool` message linked to its call ID, and the loop calls again. If the model answers instead, that's the result.

**Guards, and why each exists:**

- **Mistakes go back to the model, once.** Arguments that aren't valid JSON or break the tool's schema, and calls to a tool that doesn't exist, are returned to the model as an error it can fix. A second mistake on the same tool before it has succeeded stops the run. A tool failing on valid input ("no customer with that email") is the data answering, not a mistake, so it doesn't count.
- **A step limit** (6 by default) stops a model that never answers.
- **Stopping early keeps the evidence.** Every early stop reports why, with the trace and the cost so far.
- **No `eval()`.** `calculate` parses the expression and evaluates only numbers, `+ - * /` and parentheses, so model output can't run code. Exponentiation is excluded because `9 ** 9 ** 9` hangs the process.
- **Schemas are checked, not trusted.** A small validator ([`schema.py`](src/llm_lab/schema.py)) reports every problem with a call at once, since the model gets one chance to fix it. A tool whose schema uses a rule the validator can't enforce fails at startup instead of silently passing bad input.
- **Provider data survives the round trip.** The assistant turn sent back on the next call keeps everything the provider attached to each tool call (Gemini needs its reasoning data back), but only standard fields on the message itself, because some providers reject fields they don't recognise.

### Results

```bash
uv run llm-lab-agent --model <alias> --cache off --expect 169.42
```

| Model | Correct | Steps | Tool calls | Repairs | Tokens in / out | Cost | Time |
|---|---|---|---|---|---|---|---|
| `groq-oss` | yes | 4 | 3 | 0 | 2,169 / 434 | $0.000293 | 1.7 s |
| `gemini-lite` | yes | 5 | 4 | 0 | 2,982 / 131 | $0.001222 | 3.4 s |
| `local-qwen-nothink` | yes | 4 | 3 | 0 | 3,138 / 144 | $0 | 15.1 s |

One run each, so these are a snapshot.

**What I learned**

- **Agent cost is mostly input, and it grows with every step.** Each call resends the whole conversation plus the tool definitions (300+ tokens on their own). `gemini-lite`'s input went 373, 437, 646, 724, 802 tokens across its five steps, so input was 23 times its output. Fewer steps and shorter tool results matter more than shorter answers.
- **The model with the fewest tokens cost the most.** `gemini-lite` produced the least output but cost about 4 times what `groq-oss` did: its input price is 4 times higher, and it split the arithmetic into two `calculate` calls, adding a step that resent everything.
- **Reasoning models spend output on small decisions.** `groq-oss` was fastest and cheapest, but used 177 output tokens to decide on one `calculate` call, and it worked out `1 - 10/100 = 0.9` itself instead of passing it to the calculator, despite being told to use the tool for all arithmetic. Harmless here; a reason to check tool use, not just final answers.
- **Locally, reading the prompt is the slow part.** The local model's first step took 7.5 s, the later ones 1.5 to 3.7 s. At the roughly 96 tokens/s prompt-processing rate I measured earlier, its 593-token first prompt accounts for about 6 s. The later steps were most likely faster because Ollama reuses work for the shared start of the conversation.
- **"Tokens" aren't comparable across providers.** The same first prompt counted as 373 (Gemini), 399 (Groq) and 593 (Qwen) tokens: each has its own tokenizer and its own way of turning tool definitions into prompt text.

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
  agent.py                the tool-calling loop
  tools.py                tools, the Toolbox that runs them, the safe calculator
  schema.py               JSON Schema validation for tool arguments
  agent_cli.py            the `llm-lab-agent` command
  data/shop.json          synthetic data for the agent
tests/                    one test file per module
```

## Limitations

- Rate limits and quota counts are per process. Several processes or machines sharing one quota would need a shared store such as Redis with an atomic Lua script.
- No streaming, so time-to-first-token isn't measured yet.
- The agent runs tools one after another; tools a model requests together could run in parallel.
- The schema validator covers only the keywords these tools use; anything else is rejected at startup rather than ignored.
