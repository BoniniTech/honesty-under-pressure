# honesty-under-pressure

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
inspect eval src/hup/task.py --model <provider/model>
```

_TODO: document the exact command(s) used for the frozen v0.1 results, including which models and the inspect-ai version pinned in `pyproject.toml`._

## Results

_TODO (D5–D6): results table + flip-rate chart (by model × pressure condition, with bootstrap CIs)._

## Limitations

_TODO (D6): sample size, prompt-template sensitivity, grader error, construct validity — what "flipping" does and doesn't prove._

## What I'd do next

_TODO (D6): candidate v0.2 directions — see "Explicitly out of scope" in `CLAUDE.md`._
