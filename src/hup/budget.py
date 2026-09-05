"""Preflight spend estimate for a full sweep.

Inspect has no run-level budget cap. Every limit it exposes — `token_limit`,
`cost_limit`, `time_limit`, `working_limit` — applies to a single sample, so the
cost of a whole run is bounded by arithmetic over the configuration rather than by
anything that fires at runtime. This module does that arithmetic, so the numbers in
the README are regenerable from a clean clone instead of hand-typed.

    python -m hup.budget --models openai/gpt-4o-mini-2024-07-18 anthropic/claude-haiku-4-5-20251001
    python -m hup.budget --models ... --passes 5
    python -m hup.budget --models ... --passes 4 --rounds 3

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
from hup.solvers import DEFAULT_ROUNDS, PressureCondition, turns_per_sample, validate_rounds
from hup.task import DEFAULT_MAX_TOKENS, DEFAULT_TOKEN_LIMIT

# Mean total tokens per sample, every turn summed, keyed by the escalation depth the
# run was made at and then by model. Raw .eval logs are gitignored, which is why these
# are recorded constants rather than recomputed here.
#
# Keyed by depth because the depth is not a multiplier on a single table. Measured, a
# three-round sample costs 3.4x to 3.9x its one-round self across the four v0.2 models,
# where the ratio of turn counts predicts 1.67x — every turn re-sends the whole
# conversation, so input grows far faster than the turn count. A single table plus
# arithmetic understated the ladder by about 2.2x, and `round_scale` still carries that
# error wherever a depth has no measurement of its own.
#
# Depth 1, v0.1 slate: the complete 120-sample-per-model pass of 2026-08-19. These
# replaced figures from the 2026-08-12 pilot, which ran ten questions. Two barely moved
# (haiku 512 to 511, gpt-4o-mini 360 to 367) and gemini-3.6-flash rose 9.3%, from 2,047
# to 2,238. Ten questions were enough for the cheap models and not for the expensive one,
# which is the model whose share of a sweep the estimate most needs right.
#
# Depths 1 and 3, v0.2 slate: the two-pass build-out of 2026-09-04, 240 samples per model
# per arm, `runs/summaries/buildout-2026-09-04.md`. Recomputed from the logs rather than
# copied from that file's spend table.
#
# Mean, not median. A total is n x mean by definition, and these distributions are
# right-skewed enough that the difference is not cosmetic: an earlier revision of this
# module projected from a single pooled median of 536 and reported 192,960 tokens for a
# sweep whose measured figure is 350,328, an 82% understatement. Pooling models with
# means from 339 to 6,902 into one median made it worse still.
OBSERVED_MEAN_TOKENS_PER_SAMPLE: dict[int, dict[str, int]] = {
    1: {
        "openai/gpt-4o-mini-2024-07-18": 367,
        "anthropic/claude-haiku-4-5-20251001": 519,
        "google/gemini-3.6-flash": 2_238,
        "openai/gpt-5.6-terra": 339,
        "anthropic/claude-sonnet-5": 1_525,
        "google/gemini-3.8-flash": 1_898,
    },
    3: {
        "openai/gpt-5.6-terra": 1_240,
        "anthropic/claude-haiku-4-5-20251001": 2_012,
        "anthropic/claude-sonnet-5": 5_177,
        "google/gemini-3.8-flash": 6_902,
    },
}

# haiku is in the depth-1 table twice over: 511 from the 2026-08-19 full run and 519 from
# the 2026-09-04 build-out. The later figure is the one recorded, because it was produced
# by the build the v0.2 run will use. The 1.6% gap between them is the size of the noise
# on a 240-sample mean, and is worth knowing before reading any of these to three digits.

# Model ids here are pinned versions, never floating aliases, and that is a correctness
# requirement rather than a style preference. The pilot ran `google/gemini-flash-latest`,
# which resolved to `gemini-3.6-flash` on 2026-08-12 and to `gemini-3.7-flash` a week
# later. Every gemini figure recorded from that pilot describes 3.6, so an alias in this
# table would attach a measurement to whichever model answers next.
#
# `gpt-4o-mini` is the same hazard wearing a less obvious name. It resolved to
# `gpt-4o-mini-2024-07-18` on both dates, so it had not moved yet, but nothing stops it.
# All three ids there were run on 2026-08-19 and each resolved to itself.


def nearest_measured_depth(rounds: int, model: str | None = None) -> int:
    """The measured depth a projection for `rounds` should be built from.

    Exact match wins. Otherwise the closest measured depth, and on a tie the deeper one:
    scaling down from a deeper measurement by the turn ratio overstates, and scaling up
    from a shallower one understates, so the tie breaks toward the safe direction.

    `model` restricts the search to depths that measured that model. Left as None it
    considers every depth that measured anything, which is what an unmeasured model's
    stand-in needs.
    """
    depths = [
        depth
        for depth, means in OBSERVED_MEAN_TOKENS_PER_SAMPLE.items()
        if model is None or model in means
    ]
    if not depths:
        raise ValueError(f"no depth in the table measured {model!r}")
    return min(depths, key=lambda depth: (abs(depth - rounds), -depth))


# An unmeasured model is assumed to behave like the most expensive one measured at the
# depth its projection is built from. Guessing low here produces a budget that is approved
# and then exceeded, which is the failure this module exists to prevent; guessing high
# produces a conversation.
def unmeasured_mean_tokens_per_sample(rounds: int) -> int:
    depth = nearest_measured_depth(rounds)
    return max(OBSERVED_MEAN_TOKENS_PER_SAMPLE[depth].values())


@dataclass(frozen=True)
class SweepEstimate:
    """Token arithmetic for one sweep, optionally repeated over several passes."""

    questions: int
    conditions: int
    models: tuple[str, ...]
    passes: int
    rounds: int
    token_limit: int
    max_tokens: int
    mean_tokens_by_model: dict[str, int]
    measured_depth_by_model: dict[str, int]

    @property
    def turns(self) -> int:
        """Assistant turns per sample at this depth: the answer, the rounds, the readout."""
        return turns_per_sample(self.rounds)

    def round_scale_for(self, model: str) -> float:
        """Factor applied to one model's measured mean to project this depth.

        1.0 where the depth was measured, which is the whole point of keying the table
        by depth. Otherwise the ratio of turn counts, which is arithmetic rather than a
        measurement: every turn re-sends the whole conversation so far, so real cost
        grows faster than the turn count while output per turn stays roughly flat.
        Measured between one round and three on the v0.2 slate the real factor was 3.4x
        to 3.9x where this ratio gives 1.67x, about 2.2x out.

        Which way it is wrong depends on the direction. Scaling UP from a shallower
        measurement understates, so the row is a floor. Scaling DOWN from a deeper one
        overstates, so the row is a ceiling. Measure a short pass at the depth and add it
        to the table instead of leaning on either.
        """
        return self.turns / turns_per_sample(self.measured_depth_by_model[model])

    @property
    def scaled_models(self) -> tuple[str, ...]:
        """Models whose projection is arithmetic off a depth they were not measured at."""
        return tuple(m for m in self.models if self.measured_depth_by_model[m] != self.rounds)

    @property
    def scaled_rounds(self) -> bool:
        """Whether any row in the projection is scaled rather than measured at this depth."""
        return bool(self.scaled_models)

    @property
    def samples_per_pass(self) -> int:
        return self.questions * self.conditions * len(self.models)

    @property
    def samples(self) -> int:
        return self.samples_per_pass * self.passes

    @property
    def generate_calls(self) -> int:
        return self.samples * self.turns

    @property
    def unmeasured_models(self) -> tuple[str, ...]:
        """Models with no recorded mean at any depth, whose share is a stand-in."""
        measured = {model for means in OBSERVED_MEAN_TOKENS_PER_SAMPLE.values() for model in means}
        return tuple(m for m in self.models if m not in measured)

    def observed_tokens_for(self, model: str) -> int:
        """Projected spend for one model across every question, condition and pass."""
        per_sample = self.mean_tokens_by_model[model] * self.round_scale_for(model)
        return round(self.questions * self.conditions * self.passes * per_sample)

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
        return self.turns * self.max_tokens

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
    rounds: int = DEFAULT_ROUNDS,
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
    # Bounds-checked by the solver rather than here, so the estimate cannot describe a
    # depth `inspect eval` would refuse to run.
    rounds = validate_rounds(rounds)
    if token_limit < 1:
        raise ValueError(f"token_limit must be at least 1, got {token_limit}")
    if max_tokens < 1:
        raise ValueError(f"max_tokens must be at least 1, got {max_tokens}")

    return SweepEstimate(
        questions=len(load_questions(dataset_path)),
        conditions=len(get_args(PressureCondition)),
        models=tuple(models),
        passes=passes,
        rounds=rounds,
        token_limit=token_limit,
        max_tokens=max_tokens,
        mean_tokens_by_model={model: _mean_for(model, rounds) for model in models},
        measured_depth_by_model={model: _depth_for(model, rounds) for model in models},
    )


def _depth_for(model: str, rounds: int) -> int:
    """The measured depth this model's projection is built from.

    An unmeasured model borrows the depth its stand-in rate came from, so its row is
    scaled by the same factor as a measured model would be.
    """
    measured = {m for means in OBSERVED_MEAN_TOKENS_PER_SAMPLE.values() for m in means}
    return nearest_measured_depth(rounds, model if model in measured else None)


def _mean_for(model: str, rounds: int) -> int:
    depth = _depth_for(model, rounds)
    return OBSERVED_MEAN_TOKENS_PER_SAMPLE[depth].get(
        model, unmeasured_mean_tokens_per_sample(rounds)
    )


def format_estimate(estimate: SweepEstimate) -> str:
    width = max(len(model) for model in estimate.models)
    lines = [
        f"questions          {estimate.questions}",
        f"conditions         {estimate.conditions}",
        f"models             {len(estimate.models)}",
        f"passes             {estimate.passes}",
        f"pushback rounds    {estimate.rounds}",
        f"samples            {estimate.samples:,}"
        + (f"  ({estimate.samples_per_pass:,} per pass)" if estimate.passes > 1 else ""),
        f"generate calls     {estimate.generate_calls:,}  ({estimate.turns} turns per sample)",
        "",
        "projected spend, per model, from measured means:",
    ]

    for model in estimate.models:
        mean = estimate.mean_tokens_by_model[model]
        depth = estimate.measured_depth_by_model[model]
        plural = "" if depth == 1 else "s"
        if model in estimate.unmeasured_models:
            marker = f"  <- no measurement, assumed (from {depth} round{plural})"
        elif model in estimate.scaled_models:
            marker = f"  <- SCALED from {depth} round{plural}"
        else:
            marker = f"  <- measured at {depth} round{plural}"
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
        rounds_plural = "" if estimate.rounds == 1 else "s"
        lines += [
            "",
            "Models marked `no measurement` have no recorded mean at any depth and are",
            f"projected at the highest rate measured at {estimate.rounds} "
            f"round{rounds_plural} "
            f"({unmeasured_mean_tokens_per_sample(estimate.rounds):,}/sample). Run a small",
            "pass and record the real figure before treating their share as tight.",
        ]

    if estimate.scaled_models:
        lines += [
            "",
            "Models marked SCALED were measured at a different depth and multiplied by the",
            "ratio of turn counts to reach this one. That ratio is arithmetic, not a",
            "measurement: every turn re-sends the conversation so far, so real cost grows",
            "faster than the turn count does. Measured between one round and three on the",
            "v0.2 slate the real factor was 3.4x to 3.9x where the ratio gives 1.67x, about",
            "2.2x out. Scaling UP therefore understates, so the row is a floor; scaling DOWN",
            "overstates, so the row is a ceiling. Measure a short pass at this depth before",
            "approving a full run.",
        ]
        lines += [
            f"  {model}: x{estimate.round_scale_for(model):.2f} from "
            f"{estimate.measured_depth_by_model[model]} round"
            f"{'' if estimate.measured_depth_by_model[model] == 1 else 's'}"
            f" ({'floor' if estimate.round_scale_for(model) > 1 else 'ceiling'})"
            for model in estimate.scaled_models
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
        "--rounds",
        type=int,
        default=DEFAULT_ROUNDS,
        help=(
            f"Pushback rounds per sample, matching `-T rounds=` (default {DEFAULT_ROUNDS}). "
            "Each round adds a turn to every sample."
        ),
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
                rounds=args.rounds,
                dataset_path=args.dataset,
                token_limit=args.token_limit,
                max_tokens=args.max_tokens,
            )
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
