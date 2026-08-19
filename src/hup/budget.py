"""Preflight spend estimate for a full sweep.

Inspect has no run-level budget cap. Every limit it exposes — `token_limit`,
`cost_limit`, `time_limit`, `working_limit` — applies to a single sample, so the
cost of a whole run is bounded by arithmetic over the configuration rather than by
anything that fires at runtime. This module does that arithmetic, so the numbers in
the README are regenerable from a clean clone instead of hand-typed.

    python -m hup.budget --models openai/gpt-4o-mini anthropic/claude-haiku-4-5-20251001

The worst case is a planning figure, not a hard ceiling. `token_limit` is checked
after a generate call returns, not mid-stream, so a sample overshoots by up to one
model response: a 50-token limit measured against gpt-4o-mini stopped a sample at
114 tokens. The overshoot is proportionally small at the default limit and large at
a tiny one. The observed figure is measured, not predicted, and carries whatever the
pilot's models and phrasing happened to do.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import get_args

from hup.dataset import DEFAULT_DATA_PATH, load_questions
from hup.solvers import TURNS_PER_SAMPLE, PressureCondition
from hup.task import DEFAULT_TOKEN_LIMIT

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
    observed_median_tokens: int

    @property
    def samples(self) -> int:
        return self.questions * self.conditions * self.models

    @property
    def generate_calls(self) -> int:
        return self.samples * TURNS_PER_SAMPLE

    @property
    def worst_case_tokens(self) -> int:
        """Planning bound, not a hard ceiling.

        `token_limit` is checked after each generate returns, so a sample can exceed it
        by one model response. Measured: a 50-token limit stopped a sample at 114 tokens.
        Treat this as the figure to budget against, not a guarantee.
        """
        return self.samples * self.token_limit

    @property
    def observed_case_tokens(self) -> int:
        """What a sweep costs if samples behave as they did in the pilot."""
        return self.samples * self.observed_median_tokens


def estimate_sweep(
    *,
    models: int,
    dataset_path: Path = DEFAULT_DATA_PATH,
    token_limit: int = DEFAULT_TOKEN_LIMIT,
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

    return SweepEstimate(
        questions=len(load_questions(dataset_path)),
        conditions=len(get_args(PressureCondition)),
        models=models,
        token_limit=token_limit,
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
            "                   limits are checked between turns, so a sample can overshoot",
            "                   by one model response; budget against this, do not treat it",
            "                   as a guarantee.",
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
    args = parser.parse_args(argv)

    estimate = estimate_sweep(
        models=len(args.models), dataset_path=args.dataset, token_limit=args.token_limit
    )
    print(format_estimate(estimate))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
