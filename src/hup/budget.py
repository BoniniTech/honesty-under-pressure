"""Preflight spend estimate for a full sweep.

Inspect has no run-level budget cap. Every limit it exposes — `token_limit`,
`cost_limit`, `time_limit`, `working_limit` — applies to a single sample, so the
cost of a whole run is bounded by arithmetic over the configuration rather than by
anything that fires at runtime. This module does that arithmetic, so the numbers in
the README are regenerable from a clean clone instead of hand-typed.

    python -m hup.budget --models openai/gpt-4o-mini anthropic/claude-haiku-4-5-20251001
    python -m hup.budget --models ... --passes 5

No number here stops a run. The only backstop that actually fires is the provider
account: balances are prepaid and do not refill on their own, so a sweep that outruns
one fails mid-flight rather than overspending. That fails closed, which is the right
direction, but it fails closed *partway* — leaving a pass covering some questions and
not others. `hup.pool` refuses such a pass rather than pooling it, because a partial
sweep weights whichever items ran first. Estimate before running; the alternative is
discovering the ceiling by hitting it.

Three figures, and the difference between them matters. The observed case projects
from measured per-model means. The worst case is the planning figure,
`samples x token_limit`. The ceiling is the number a sweep cannot exceed:
`token_limit` is checked between turns rather than mid-generation, so a sample always
overshoots by the response it was already committed to, and `max_tokens` is what
bounds that response. Without a `max_tokens` there is no ceiling to compute.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import get_args

from hup.dataset import DEFAULT_DATA_PATH, load_questions
from hup.solvers import TURNS_PER_SAMPLE, PressureCondition
from hup.task import DEFAULT_MAX_TOKENS, DEFAULT_TOKEN_LIMIT

# Mean total tokens per sample, all three turns summed, measured over the complete
# 120-sample-per-model D5 pass of 2026-08-19 — the whole 40-question set under all three
# conditions. Raw .eval logs are gitignored, which is why these are recorded constants
# rather than recomputed here.
#
# These replace figures taken from the 2026-08-12 pilot, which ran ten questions. Two
# barely moved (haiku 512 to 511, gpt-4o-mini 360 to 367) and gemini-3.6-flash rose 9.3%,
# from 2,047 to 2,238. Ten questions were enough for the cheap models and not for the
# expensive one, which is the model whose share of a sweep the estimate most needs right.
#
# Mean, not median. A total is n x mean by definition, and these distributions are
# right-skewed enough that the difference is not cosmetic: an earlier revision of this
# module projected from a single pooled median of 536 and reported 192,960 tokens for a
# sweep whose measured figure is 350,328, an 82% understatement. Pooling models with
# means from 367 to 2,238 into one median made it worse still.
OBSERVED_MEAN_TOKENS_PER_SAMPLE: dict[str, int] = {
    "openai/gpt-4o-mini-2024-07-18": 367,
    "anthropic/claude-haiku-4-5-20251001": 511,
    "google/gemini-3.6-flash": 2_238,
}

# Model ids here are pinned versions, never floating aliases, and that is a correctness
# requirement rather than a style preference. The pilot ran `google/gemini-flash-latest`,
# which resolved to `gemini-3.6-flash` on 2026-08-12 and to `gemini-3.7-flash` a week
# later. Every gemini figure recorded from that pilot describes 3.6, so an alias in this
# table would attach a measurement to whichever model answers next.
#
# `gpt-4o-mini` is the same hazard wearing a less obvious name. It resolved to
# `gpt-4o-mini-2024-07-18` on both dates, so it had not moved yet, but nothing stops it.
# All three ids here were run on 2026-08-19 and each resolved to itself.

# An unmeasured model is assumed to behave like the most expensive one measured. Guessing
# low here produces a budget that is approved and then exceeded, which is the failure this
# module exists to prevent; guessing high produces a conversation.
UNMEASURED_MEAN_TOKENS_PER_SAMPLE = max(OBSERVED_MEAN_TOKENS_PER_SAMPLE.values())


@dataclass(frozen=True)
class SweepEstimate:
    """Token arithmetic for one sweep, optionally repeated over several passes."""

    questions: int
    conditions: int
    models: tuple[str, ...]
    passes: int
    token_limit: int
    max_tokens: int
    mean_tokens_by_model: dict[str, int]

    @property
    def samples_per_pass(self) -> int:
        return self.questions * self.conditions * len(self.models)

    @property
    def samples(self) -> int:
        return self.samples_per_pass * self.passes

    @property
    def generate_calls(self) -> int:
        return self.samples * TURNS_PER_SAMPLE

    @property
    def unmeasured_models(self) -> tuple[str, ...]:
        """Models with no recorded mean, whose share of the estimate is a stand-in."""
        return tuple(m for m in self.models if m not in OBSERVED_MEAN_TOKENS_PER_SAMPLE)

    def observed_tokens_for(self, model: str) -> int:
        """Projected spend for one model across every question, condition and pass."""
        return self.questions * self.conditions * self.passes * self.mean_tokens_by_model[model]

    @property
    def observed_case_tokens(self) -> int:
        """Projection from measured per-model means.

        Summed per model rather than scaled from one pooled figure, because the models
        differ by more than 5x and a mixed-population average describes none of them.
        """
        return sum(self.observed_tokens_for(model) for model in self.models)

    @property
    def worst_case_tokens(self) -> int:
        """Planning figure: what a sweep costs if every sample runs to its token limit.

        Not a guarantee — see `ceiling_tokens` for the number a sweep cannot exceed.
        """
        return self.samples * self.token_limit

    @property
    def overshoot_per_sample(self) -> int:
        """How far one sample can exceed `token_limit` before it is stopped.

        The limit is checked between turns, so the final response completes and is billed
        in full. That response costs at most `max_tokens` of output, and its input is the
        conversation so far, itself bounded by the earlier capped responses. `turns x
        max_tokens` covers both. Excludes the scripted prompt text, which is tens of
        tokens per call and does not scale with anything.
        """
        return TURNS_PER_SAMPLE * self.max_tokens

    @property
    def ceiling_tokens(self) -> int:
        """Upper bound a sweep cannot exceed, give or take the prompt text.

        Computable only because `max_tokens` bounds the overshoot. With `max_tokens`
        unset the provider default applies and this number does not exist.
        """
        return self.samples * (self.token_limit + self.overshoot_per_sample)


def estimate_sweep(
    *,
    models: Sequence[str],
    passes: int = 1,
    dataset_path: Path = DEFAULT_DATA_PATH,
    token_limit: int = DEFAULT_TOKEN_LIMIT,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> SweepEstimate:
    """Size a sweep from the live dataset, the declared conditions, and the model list.

    Question and condition counts are derived rather than passed, so the estimate cannot
    drift from the dataset that will actually run.
    """
    if not models:
        raise ValueError("at least one model is required")
    if len(set(models)) != len(models):
        raise ValueError(f"models must be unique, got {list(models)}")
    if passes < 1:
        raise ValueError(f"passes must be at least 1, got {passes}")
    if token_limit < 1:
        raise ValueError(f"token_limit must be at least 1, got {token_limit}")
    if max_tokens < 1:
        raise ValueError(f"max_tokens must be at least 1, got {max_tokens}")

    return SweepEstimate(
        questions=len(load_questions(dataset_path)),
        conditions=len(get_args(PressureCondition)),
        models=tuple(models),
        passes=passes,
        token_limit=token_limit,
        max_tokens=max_tokens,
        mean_tokens_by_model={
            model: OBSERVED_MEAN_TOKENS_PER_SAMPLE.get(model, UNMEASURED_MEAN_TOKENS_PER_SAMPLE)
            for model in models
        },
    )


def format_estimate(estimate: SweepEstimate) -> str:
    width = max(len(model) for model in estimate.models)
    lines = [
        f"questions          {estimate.questions}",
        f"conditions         {estimate.conditions}",
        f"models             {len(estimate.models)}",
        f"passes             {estimate.passes}",
        f"samples            {estimate.samples:,}"
        + (f"  ({estimate.samples_per_pass:,} per pass)" if estimate.passes > 1 else ""),
        f"generate calls     {estimate.generate_calls:,}  ({TURNS_PER_SAMPLE} turns per sample)",
        "",
        "projected spend, per model, from measured means:",
    ]

    for model in estimate.models:
        mean = estimate.mean_tokens_by_model[model]
        marker = "  <- no measurement, assumed" if model in estimate.unmeasured_models else ""
        lines.append(
            f"  {model:<{width}}  {estimate.observed_tokens_for(model):>10,}"
            f"  (at {mean:,}/sample){marker}"
        )

    lines += [
        "",
        f"observed tokens    {estimate.observed_case_tokens:,}  (sum of the rows above)",
        f"worst-case tokens  {estimate.worst_case_tokens:,}"
        f"  ({estimate.token_limit:,} token limit x {estimate.samples:,} samples)",
        f"ceiling tokens     {estimate.ceiling_tokens:,}"
        f"  (+{estimate.overshoot_per_sample:,}/sample overshoot, bounded by"
        f" max_tokens={estimate.max_tokens:,})",
        "",
        "Limits are checked between turns, so a sample is billed for the response it had",
        "already committed to. max_tokens bounds that response, which is what makes the",
        "ceiling a real number. It excludes the scripted prompt text, tens of tokens/call.",
        "",
        "Token counts only. Convert with current provider pricing before approving a run;",
        "this repo keeps no price table, because a stale one understates the bill silently.",
    ]

    if estimate.unmeasured_models:
        lines += [
            "",
            "Models marked above have no measured mean and are projected at the highest",
            f"observed rate ({UNMEASURED_MEAN_TOKENS_PER_SAMPLE:,}/sample). Run a small pass and",
            "record the real figure before treating their share of this number as tight.",
        ]

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m hup.budget",
        description="Estimate token spend for a sweep before running it.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        metavar="MODEL",
        help="Model ids the sweep will run against. Spend is projected per model.",
    )
    parser.add_argument(
        "--passes",
        type=int,
        default=1,
        help="Repeated passes over the whole sweep, pooled afterwards (default 1).",
    )
    parser.add_argument(
        "--dataset", type=Path, default=DEFAULT_DATA_PATH, help="Path to questions.jsonl."
    )
    parser.add_argument(
        "--token-limit",
        type=int,
        default=DEFAULT_TOKEN_LIMIT,
        help=f"Per-sample token ceiling to assume (default {DEFAULT_TOKEN_LIMIT:,}).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Per-response output ceiling to assume (default {DEFAULT_MAX_TOKENS:,}).",
    )
    args = parser.parse_args(argv)

    print(
        format_estimate(
            estimate_sweep(
                models=args.models,
                passes=args.passes,
                dataset_path=args.dataset,
                token_limit=args.token_limit,
                max_tokens=args.max_tokens,
            )
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
