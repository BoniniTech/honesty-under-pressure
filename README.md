# honesty-under-pressure

[![CI](https://github.com/BoniniTech/honesty-under-pressure/actions/workflows/ci.yml/badge.svg)](https://github.com/BoniniTech/honesty-under-pressure/actions/workflows/ci.yml)

**Status: scaffolding only — no eval has run yet.** This README will be filled in as the project moves through its 7-day plan (see `CLAUDE.md`).

## Motivation

_TODO (D6): 3–4 sentences on why this question matters._

## Method

An [Inspect AI](https://inspect.aisi.org.uk/) eval. For each of ~100 factual QA items: ask the question, apply one of three scripted pushback conditions (plain contradiction, authority appeal, confidence + social pressure) inserting a plausible wrong answer, then ask for a final answer. A custom scorer records whether the model's answer flipped from correct to incorrect (or vice versa).

Full design details, non-negotiables, and scope boundaries live in `CLAUDE.md`.

## Running it

```bash
uv sync
cp .env.example .env   # fill in real API keys, never commit .env
inspect eval src/hup/task.py --model google/gemini-flash-latest --limit 1
```

That one command runs all three pressure conditions. `src/hup/task.py` defines one task
per condition — `plain_contradiction`, `authority_appeal`, `confidence_social` — and
`inspect eval` runs every task in a file, producing one log per condition with the
condition name in the filename. To run a single condition, select it by name:

```bash
inspect eval src/hup/task.py@authority_appeal --model google/gemini-flash-latest
```

`--model` takes any Inspect-supported `<provider>/<model>` id, e.g. `openai/gpt-4o-mini` or `anthropic/claude-haiku-4-5-20251001` — check each provider's current model list before running, they deprecate names often (`google/gemini-2.0-flash` and `google/gemini-2.5-flash-lite`, for instance, both 404 as of this writing; `google/gemini-flash-latest` is confirmed working). Drop `--limit 1` once you're past smoke-testing and ready to run the full ~100-item set.

On Windows, pass the task as a path relative to the repo root as shown. An absolute path
raises `NotImplementedError: Non-relative patterns are unsupported` from Inspect's task
loader on `inspect-ai==0.3.255`.

_TODO: document the exact command(s) used for the frozen v0.1 results, including which models and the inspect-ai version pinned in `pyproject.toml`._

## Results

_TODO (D5–D6): results table + flip-rate chart (by model × pressure condition, with bootstrap CIs)._

## Limitations

**Answer matching.** Correctness is decided by normalized whole-word containment of
the target and of the sample's plausible wrong answer. An answer naming both is not
decidable that way: "no, it's Au, not Ag" (holding) and "it's Ag, not Au"
(capitulating) contain the same tokens and mean opposite things. Those samples are
reported as ambiguous and dropped from the flip-rate denominator instead of being
guessed at, so `flip_rate` should always be read next to `ambiguous_rate`. A model
whose habit is to answer by contrast will drive `ambiguous_rate` up and thin out the
base the headline number rests on, which is a property of its phrasing rather than
its honesty. Adjudicating that residue, most likely with a logged and hand-audited
model grader, is the obvious next step.

_TODO (D6): sample size, prompt-template sensitivity, grader error, construct validity — what "flipping" does and doesn't prove._

## What I'd do next

_TODO (D6): candidate v0.2 directions — see "Explicitly out of scope" in `CLAUDE.md`._
