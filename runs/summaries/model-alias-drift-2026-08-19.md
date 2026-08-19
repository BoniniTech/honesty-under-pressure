# Model alias drift, 2026-08-19

A week of recorded results describe two different Google models under one name. Found
while diagnosing why the D5 canary's gemini pass kept failing.

## What happened

Every run in this repo requested `google/gemini-flash-latest`. Reading the resolved model
back out of the `.eval` logs:

| run | date | requested | **resolved** |
|---|---|---|---|
| `pilot-stage1` | 2026-08-12 | `google/gemini-flash-latest` | `gemini-3.6-flash` |
| `pilot-stage2` | 2026-08-12 | `google/gemini-flash-latest` | `gemini-3.6-flash` |
| `pilot-stage2b` | 2026-08-12 | `google/gemini-flash-latest` | `gemini-3.6-flash` |
| `ab-control` | 2026-08-12 | `google/gemini-flash-latest` | `gemini-3.6-flash` |
| `ab-concise` | 2026-08-12 | `google/gemini-flash-latest` | `gemini-3.6-flash` |
| `d5/pass1` | 2026-08-19 | `google/gemini-flash-latest` | **`gemini-3.7-flash`** |

The alias moved between the pilot and the first D5 attempt. Nothing announced it.

The other two models did not drift. `anthropic/claude-haiku-4-5-20251001` is a pinned id
and resolved to itself throughout. `openai/gpt-4o-mini` is technically an alias but
resolved to `gpt-4o-mini-2024-07-18` on both dates. It has not moved; that is luck rather
than a guarantee, so it is now pinned to the dated id as well.

## Why it matters

Every gemini number recorded before today describes `gemini-3.6-flash`: the 2,047 mean
tokens per sample, the 39%-to-20% ambiguity movement attributed to the distractor change,
the per-model ambiguity split that motivated the model-graded-fallback issue, the
distractor-type breakdown. A results table with a "model" column reading
`gemini-flash-latest` would name a moving target, and a reader could not tell which model
produced which row.

This is a reproducibility defect against non-negotiable #3, which requires logging model
versions for every run. The `.eval` logs did capture the resolved version — that is how
this was found — but the summaries recorded only the requested alias, so nothing surfaced
the change.

## The behaviour that exposed it

`gemini-3.7-flash` is not a drop-in for 3.6. It emits reasoning tokens: a first-turn reply
to "What is the chemical symbol for gold?" used 56 output tokens, 37 of them reasoning.
Consequences measured today:

- **Slower.** Three samples took 54 seconds, about six seconds per call. `gpt-4o-mini`
  completed 120 samples in 26 seconds.
- **More expensive.** Derived rate of $4.55/1M against $2.60 for haiku and $0.23 for
  `gpt-4o-mini`, and it was already the token-hungriest of the three.
- **Unstable under concurrency.** At Inspect's default of 10 connections, a sample
  exhausted its five retries on `ServerError`, which failed the task and cancelled every
  other sample in flight. Only 67 of 120 gemini samples scored. At three concurrent
  samples it succeeded cleanly, so the errors are load-related rather than an outage.

That last point is also what turned an earlier attempt into a 2h23m hang, before retries
were bounded — see `retry-hang-2026-08-19.md`.

## Decision

Pin `google/gemini-3.6-flash`.

It is what the pilot actually ran, so every existing gemini figure in this repo stays
valid rather than needing a caveat, and cross-run comparability is restored. Both
`google/gemini-3.6-flash` and `google/gemini-3.7-flash` were verified addressable as
explicit ids, so the alias was never necessary.

`gemini-3.7-flash` is a reasonable v0.2 subject — reasoning versus non-reasoning models
under pressure is a real question — but adding it to v0.1 would make the model set a
confound rather than a comparison.

## What changed

- `src/hup/budget.py` records the measured mean against `google/gemini-3.6-flash`. The
  value is unchanged at 2,047, because that is the model it was measured on.
- `openai/gpt-4o-mini` is pinned to `openai/gpt-4o-mini-2024-07-18` for the same reason.
- A test requires every recorded id to carry a date or a dotted version. Rejecting the
  word `latest` would not have been enough, since `gpt-4o-mini` contains no such marker
  and is an alias all the same.
- `README.md` and `CLAUDE.md` require pinned versions and name the three ids v0.1 uses.
- Google runs with `--max-connections 5`. The other two are fine at the default.

## What this does not fix

The resolved version is recorded in the `.eval` logs and now in this file, but nothing
automatically checks that a run used the model a summary claims. A pinned id makes drift
much less likely rather than impossible — a provider can still retire or repoint a
version. Reading the resolved model out of the log remains a manual step at write-up time.
