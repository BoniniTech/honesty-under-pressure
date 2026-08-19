"""Preflight spend estimate for a full sweep.

Inspect has no run-level budget cap. Every limit it exposes — `token_limit`,
`cost_limit`, `time_limit`, `working_limit` — applies to a single sample, so the
cost of a whole run is bounded by arithmetic over the configuration rather than by
anything that fires at runtime. This module does that arithmetic, so the numbers in
the README are regenerable from a clean clone instead of hand-typed.

    python -m hup.budget --models openai/gpt-4o-mini anthropic/claude-haiku-4-5-20251001

Three figures, and the difference between them matters. The observed case is measured
from the pilot and carries that run's models and phrasing. The worst case is the
planning figure, `samples x token_limit`. The ceiling is the number a sample cannot
exceed: `token_limit` is checked between turns rather than mid-generation, so a sample
always overshoots by the response it was already committed to, and `max_tokens` is what
bounds that response. Without a `max_tokens` there is no ceiling to compute.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import get_args

from hup.dataset import DEFAULT_DATA_PATH, load_questions
from hup.solvers import TURNS_PER_SAMPLE, PressureCondition
from hup.task import DEFAULT_MAX_TOKENS, DEFAULT_TOKEN_LIMIT

# Median total tokens per sample over the 90-sample stage-3 concise arm, all three
# turns summed: haiku 512, gpt-4o-mini 319, gemini-flash-latest 1,939. Measured, not
# modelled — see runs/summaries/pilot-2026-08-12.md. Raw .eval logs are gitignored,
# which is why this is a recorded constant rather than something recomputed here.
OBSERVED_MEDIAN_TOKENS_PER_SAMPLE = 536


@dataclass(frozen=True)
class SweepEstimate:
    """Token arithmetic for one full sweep."""

    questions: int
    conditions: int
    models: int
    token_limit: int
    max_tokens: int
    observed_median_tokens: int

    @property
    def samples(self) -> int:
        return self.questions * self.conditions * self.models

    @property
    def generate_calls(self) -> int:
        return self.samples * TURNS_PER_SAMPLE

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

    @property
    def observed_case_tokens(self) -> int:
        """What a sweep costs if samples behave as they did in the pilot."""
        return self.samples * self.observed_median_tokens


def estimate_sweep(
    *,
    models: int,
    dataset_path: Path = DEFAULT_DATA_PATH,
    token_limit: int = DEFAULT_TOKEN_LIMIT,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    observed_median_tokens: int = OBSERVED_MEDIAN_TOKENS_PER_SAMPLE,
) -> SweepEstimate:
    """Size a full sweep from the live dataset and the declared pressure conditions.

    Question and condition counts are derived rather than passed, so the estimate
    cannot drift from the dataset that will actually run.
    """
    if models < 1:
        raise ValueError(f"models must be at least 1, got {models}")
    if token_limit < 1:
        raise ValueError(f"token_limit must be at least 1, got {token_limit}")
    if max_tokens < 1:
        raise ValueError(f"max_tokens must be at least 1, got {max_tokens}")

    return SweepEstimate(
        questions=len(load_questions(dataset_path)),
        conditions=len(get_args(PressureCondition)),
        models=models,
        token_limit=token_limit,
        max_tokens=max_tokens,
        observed_median_tokens=observed_median_tokens,
    )


def format_estimate(estimate: SweepEstimate) -> str:
    return "\n".join(
        [
            f"questions          {estimate.questions}",
            f"conditions         {estimate.conditions}",
            f"models             {estimate.models}",
            f"samples            {estimate.samples:,}",
            f"generate calls     {estimate.generate_calls:,}"
            f"  ({TURNS_PER_SAMPLE} turns per sample)",
            "",
            f"observed tokens    {estimate.observed_case_tokens:,}"
            f"  (at {estimate.observed_median_tokens:,} median/sample, pilot 2026-08-12)",
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
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m hup.budget",
        description="Estimate token spend for a full sweep before running it.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        metavar="MODEL",
        help="Model ids the sweep will run against; only the count affects the estimate.",
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

    estimate = estimate_sweep(
        models=len(args.models),
        dataset_path=args.dataset,
        token_limit=args.token_limit,
        max_tokens=args.max_tokens,
    )
    print(format_estimate(estimate))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
