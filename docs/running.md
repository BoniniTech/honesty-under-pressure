# Running the eval

Everything needed to run `honesty-under-pressure` yourself: environment setup, the
cost and duration caps, the failure modes worth knowing before you spend money, and
the exact commands that regenerate the published numbers.

For what the eval measures and what it found, see the [README](../README.md).

## Prerequisites

- **Python 3.11 or newer** — `pyproject.toml` sets `requires-python = ">=3.11"`.
- **[uv](https://docs.astral.sh/uv/getting-started/installation/)** for the environment.
  Nothing here uses `pip` or a hand-rolled venv, and `uv.lock` is what pins the
  dependency set a published number was produced under.
- **API keys** for whichever providers you point `--model` at. `.env.example` names the
  three variables this eval reads: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`,
  `GOOGLE_API_KEY`. You only need the ones for the models you actually run.

`uv sync` creates `.venv` but does not activate it. Every command below calls `inspect`,
`pytest` and `ruff` bare, so activate the venv first (`.venv\Scripts\activate` on
Windows, `source .venv/bin/activate` elsewhere) or prefix each one with `uv run`.

The test suite needs neither keys nor network — `pytest` runs on a clean clone with
nothing in `.env`.

```bash
uv sync
cp .env.example .env   # fill in real API keys, never commit .env
inspect eval src/hup/task.py --model anthropic/claude-haiku-4-5-20251001 --limit 1
```

That one command runs all three pressure conditions. `src/hup/task.py` defines one task
per condition — `plain_contradiction`, `authority_appeal`, `confidence_social` — and
`inspect eval` runs every task in a file, producing one log per condition with the
condition name in the filename. To run a single condition, select it by name:

```bash
inspect eval src/hup/task.py@authority_appeal --model anthropic/claude-haiku-4-5-20251001
```

Pushback is one round by default. `-T rounds=` escalates it, up to the three scripted
rungs each condition carries:

```bash
inspect eval src/hup/task.py --model anthropic/claude-haiku-4-5-20251001 -T rounds=3
```

Each round adds a turn to every sample, so three rounds is five turns. It is **not** 1.67x
the tokens: measured across the v0.2 slate, three rounds costs 3.4x to 3.9x one round,
because every turn re-sends the whole conversation so far and input grows much faster than
the turn count. A depth above three is refused rather than clamped — repeating a rung would
report an escalation the run did not apply. Pass it explicitly even when you want the
default: it lands in the log's `task_args`, so the depth travels with the results instead
of having to be inferred from whichever version of the solver was checked out.

`--model` takes any Inspect-supported `<provider>/<model>` id. **Use a pinned version, never a floating alias.** `google/gemini-flash-latest` resolved to `gemini-3.6-flash` on 2026-08-12 and to `gemini-3.7-flash` on 2026-08-19, so results recorded a week apart came from different models under one name.

Pinning has two halves, because a dated id is not always on offer. Where the provider publishes one, use it: `anthropic/claude-haiku-4-5-20251001`, `openai/gpt-4o-mini-2024-07-18`. Where none exists — Anthropic dropped date suffixes at 4.6, so `claude-sonnet-5` is the complete id — read the resolved id back out of the response and record it with the run date. `openai/gpt-4o-mini` shows why the string alone cannot decide this: it is an alias that happens to resolve to the dated id, which is luck rather than a guarantee. The published run used `anthropic/claude-sonnet-5`, `openai/gpt-5.6-terra`, `google/gemini-3.8-flash` and `anthropic/claude-haiku-4-5-20251001`, and all four resolved to themselves. Drop `--limit 1` once you're past smoke-testing and ready to run the full 40-item set.

On Windows, pass the task as a path relative to the repo root as shown. An absolute path
raises `NotImplementedError: Non-relative patterns are unsupported` from Inspect's task
loader on `inspect-ai==0.3.255`.

**Running non-interactively? Pass `--display plain`.** `--display` defaults to `full`, a
rich TUI that hangs when stdout is not a terminal — backgrounded, redirected to a file, or
piped into a script. The eval starts, writes its journal entry, and then sits there: no
error, no timeout, process still alive. A script chaining several runs with output to
`/dev/null` will hang on the first and never reach the rest.

It is worth knowing what that looks like from disk, because it is easy to misread. Inspect
flushes samples to the `.eval` only when a task *completes*, so a run killed part-way and
one that never started are indistinguishable — both leave an archive holding nothing but
`_journal/start.json`, and both read back as `status='started'` with zero samples. "Zero
samples after thirty minutes" is therefore not evidence of a stall, and reading it that way
costs an afternoon.

## Pooling repeated passes

Per-cell verdicts are not stable: the same model, item and condition re-run five times
gave two flips, two ambiguous and one hold. See
`runs/summaries/verdict-instability-2026-08-19.md`. A single sweep measures each cell
once, so results come from several passes pooled together:

```bash
python -m hup.pool runs/full-2026-09-05/pass*/*.eval
```

Under the metric table it prints where each cell's flips happened — the round the model
first named the pushback answer on, with a separate column for the ones that argued
through every round and only conceded when asked for the answer alone. Logs written
before escalation existed carry no round data and say so, rather than printing zeros that
would read as a run where no flip landed on any round.

Pooling passes that ran **different escalation depths** into one cell is refused. A
one-round pass and a three-round pass produce the same cells, the same sample counts and
the same columns, so nothing in the table would show that the flip rate describes neither.

Glob the pass directories, not `runs/full-2026-09-05/*.eval`. A run directory also
holds the logs of whatever went wrong — here `credit-exhausted/` for the two passes the
Anthropic balance cut short, and `failed-timeout/` for one condition that died on
`RetryError(TimeoutError)` and was re-run. Those are kept on purpose and must stay out of
any pooled number. The narrower glob excludes them by construction; the wider one
silently includes them.

Google needs throttling. At the default 10 concurrent connections `gemini` returns
`ServerError` often enough that a sample exhausts its retries and takes the task down
with it, cancelling whatever else was in flight. Pass `--max-connections 5` for that
provider; the other two run fine at the default.

Passes are separate `inspect eval` runs, not `--epochs`. Inspect's epoch reducers keep
`metadata` from the first epoch only, and every metric here reads metadata, so epochs
would report pass-1 numbers at N times the spend.

## Before a full run

Estimate the spend first. The estimator makes no provider calls and needs no key:

```bash
python -m hup.budget --models anthropic/claude-sonnet-5 openai/gpt-5.6-terra google/gemini-3.8-flash anthropic/claude-haiku-4-5-20251001 --passes 4 --rounds 3
```

It derives the question count from `data/questions.jsonl` and the condition count from
the solver, so it cannot describe a sweep other than the one about to run, and it
projects spend per model from measured means rather than one blended figure. The models
differ by more than 5x per sample, so a blended average describes none of them.

`--rounds` has to match the `-T rounds=` the run will use, since rounds multiply the bill
and trade directly against passes. Means are recorded per depth, so a projection at a
measured depth is a measurement: the estimate for the published four-pass run was 7,358,880
tokens against 7,374,223 actually billed, 0.2% low. At a depth nothing was measured at, the
estimate scales the nearest measured mean by the turn-count ratio, marks the row `SCALED`,
and says which way it is wrong — scaling up understates, so the row is a floor; scaling down
from a deeper measurement overstates, so it is a ceiling. Turn-count arithmetic alone would
have projected 3,424,800 tokens for that run, understating the bill by 2.15x.

A run is bounded in three ways beyond spend, because Inspect leaves all three unset and
an unbounded run can hang rather than fail: `--timeout` on a single request,
`--max-retries` on the retry loop (Inspect's default is *unlimited*), and `--time-limit`
as a per-sample wall clock. One stalled sample once held a run open for 2h23m on 25
seconds of CPU. See `runs/summaries/retry-hang-2026-08-19.md`.

Nothing in this repo stops a run on cost, though. The backstop that actually fires there
is the provider account, and it fires partway through — leaving a pass that covered some
questions and not others. `python -m hup.pool` refuses such a pass rather than pooling
it, since a partial sweep weights whichever items ran first.

Each sample carries a per-sample `token_limit` (`DEFAULT_TOKEN_LIMIT` in
`src/hup/task.py`), overridable with `--token-limit`. That bounds one runaway sample; it
is **not** a budget for the run. Inspect has no run-level cap — every limit it exposes is
per-sample — so a sweep costs about `samples x token_limit` at worst, and the only
run-scoped control is `--limit`, which caps how many samples execute.

The limit is checked between turns rather than mid-generation, so a sample is billed for
the response it had already committed to. Verified against `gpt-4o-mini`: a 50-token limit
halted a sample after one turn instead of three, having used 114 tokens.

A second cap bounds that overshoot. `DEFAULT_MAX_TOKENS` (also `src/hup/task.py`,
overridable with `--max-tokens`) is sent with the request and enforced by the provider
*during* generation, so no single response can exceed it. The two do different jobs:

| | `token_limit` | `max_tokens` |
|---|---|---|
| enforced by | Inspect, between turns | the provider, during generation |
| counts | input + output, whole sample | output, one response |
| catches | turns that add up to too much | one response that runs away |
| can overshoot | yes, by one response | no |

Together they give a sweep a real upper bound — `samples x (token_limit + turns x
max_tokens)` — which `python -m hup.budget` reports as `ceiling tokens`. Without
`max_tokens` that number does not exist, because the overshoot has no size.

`--cost-limit` is deliberately not used here. Inspect records cost only when a model
carries price data, and all three target models ship none, so the check never runs and the
flag would look like protection that is not there.

## Reproducing the published results

Four passes at three rounds, on `inspect-ai==0.3.255` as pinned in `pyproject.toml`. Each
pass is four `inspect eval` invocations, one per model, and each writes three logs, one
per pressure condition:

```bash
for pass in 1 2 3 4; do
  for model in anthropic/claude-sonnet-5 openai/gpt-5.6-terra google/gemini-3.8-flash anthropic/claude-haiku-4-5-20251001; do
    inspect eval src/hup/task.py --model $model -T rounds=3 --max-connections 5 --display plain --log-dir runs/full-2026-09-05/pass$pass
  done
done

python -m hup.pool runs/full-2026-09-05/pass*/*.eval
```

`--display plain` is required, not cosmetic: `inspect eval` defaults to a rich TUI that
hangs when stdout is not a terminal, with no error and no timeout.

`--max-connections 5` is on every model here because that is what the run did — all 48
logs record it — and this block reproduces the run rather than improving on it. Only
Google needs it (see [Pooling repeated passes](#pooling-repeated-passes)); a fresh run
can leave the other three at the default 10 and go faster for it.

`rounds=1` is the current default, so it could be left off. It is written out because
this block has to keep regenerating these numbers after the default moves, and because a
reader comparing it against a later multi-round run should not have to know which depth
was default on which day.

Everything else comes from the task defaults in `src/hup/task.py`: `token_limit` 10,000,
`max_tokens` 3,000, `timeout` 120s, `max_retries` 5 and `time_limit` 600s. Sampling is
left at each provider's default, so passes differ, which is the point of running four.
One caveat on `max_tokens`: the run itself used 2,000, raised afterwards as headroom for
the extra escalation turns. Nothing truncated at either value, so the results reproduce
at the current default — but the number above is today's setting, not a record of that
run's.

A scorer change can be applied to logs you already have, without paying for a run:

```bash
python -m hup.rescore runs/full-2026-09-05/pass*/*.eval
```

This is how the markdown-emphasis fix reached the published table. It works because the
scorer is pure containment over text already in the log — no provider calls, though
Inspect still initialises a client for the model named in the log, so it wants a key in
the environment and any string will do.

## A fix that could not be backdated

Not every scoring fix re-scores. Answer aliases live in each sample's metadata, written
when the sample ran, so a log recorded before the field existed carries no aliases and
re-scoring reads none. An alias added today cannot reach a sample that ran yesterday, and
`python -m hup.rescore` will not tell you it fell short — it reads the metadata that is
there.

The distinction generalises: a change to the scorer's own logic backdates, a change to
what the dataset records about a sample does not. Aliases sit in the dataset because that
is where a human can review them, and the cost of that choice is exactly this. Anything
in the second class needs a fresh run to take effect.

Inspect does not record the command line, so the block above is reconstructed from what
the logs say was in force rather than copied from a shell history. Read the per-call
`ModelEvent` config for that, not `eval.model_generate_config`, which holds only the
CLI-level config and reads as `None` wherever the task set the value instead. A log
written before a settings commit landed will differ from one written after, and the
`ModelEvent` config is the only place that shows it.

