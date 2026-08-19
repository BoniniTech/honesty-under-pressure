"""Cost-cap configuration and the preflight sweep estimate.

What CI can prove here is narrow, and the boundary is worth stating. mockllm reports
no token usage at all (`sample.model_usage == {}`), so `token_limit` can never trip
against it — a run with `token_limit=1` completes all three turns unharmed. Whether
the cap actually stops a real run is therefore a paid, local check reported by hand,
per the CI rule in CLAUDE.md.

These tests cover the parts that can rot silently without a provider: the tasks
carrying a limit at all, and the estimate staying in step with the dataset and the
solver it is meant to describe.
"""

from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path
from typing import get_args

import pytest
from inspect_ai import eval as inspect_eval
from inspect_ai.model import ChatMessage, ModelOutput

from hup.budget import (
    OBSERVED_MEAN_TOKENS_PER_SAMPLE,
    UNMEASURED_MEAN_TOKENS_PER_SAMPLE,
    SweepEstimate,
    estimate_sweep,
    format_estimate,
    main,
)
from hup.solvers import TURNS_PER_SAMPLE, PressureCondition
from hup.task import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_MAX_TOKENS,
    DEFAULT_REQUEST_TIMEOUT,
    DEFAULT_TIME_LIMIT,
    DEFAULT_TOKEN_LIMIT,
    authority_appeal,
    confidence_social,
    plain_contradiction,
)

_TASKS = (plain_contradiction, authority_appeal, confidence_social)

# Three measured models, so estimates in these tests project from recorded means rather
# than the unmeasured-model stand-in.
_MEASURED = tuple(OBSERVED_MEAN_TOKENS_PER_SAMPLE)


def _estimate(**kwargs: object) -> SweepEstimate:
    kwargs.setdefault("models", _MEASURED)
    return estimate_sweep(**kwargs)  # type: ignore[arg-type]


def _dataset(path: Path, count: int) -> Path:
    records = [
        {
            "id": f"q{i:03d}",
            "question": f"Test question number {i}?",
            "target": f"target{i}",
            "plausible_wrong_answer": f"wrong{i}",
            "category": "test",
        }
        for i in range(1, count + 1)
    ]
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return path


@pytest.mark.parametrize("task_fn", _TASKS)
def test_every_task_carries_the_default_token_limit(task_fn: object, tmp_path: Path) -> None:
    """The cap is the only limit that fires for these models, so a task losing it
    silently removes the sole guard against one sample running away."""
    built = task_fn(dataset_path=_dataset(tmp_path / f"{task_fn.__name__}.jsonl", 1))  # type: ignore[operator,attr-defined]
    assert built.token_limit == DEFAULT_TOKEN_LIMIT


@pytest.mark.parametrize("task_fn", _TASKS)
def test_every_task_caps_output_per_response(task_fn: object, tmp_path: Path) -> None:
    """max_tokens is what bounds the overshoot on token_limit. Losing it puts the per
    response ceiling back to whatever each provider defaults to, and the estimate's
    ceiling figure silently stops being true."""
    built = task_fn(dataset_path=_dataset(tmp_path / f"{task_fn.__name__}-mt.jsonl", 1))  # type: ignore[operator,attr-defined]
    assert built.config.max_tokens == DEFAULT_MAX_TOKENS


@pytest.mark.parametrize("task_fn", _TASKS)
def test_every_task_bounds_how_long_it_can_hang(task_fn: object, tmp_path: Path) -> None:
    """Inspect leaves retries and clocks unset, and unlimited retries is what turned one
    stalled sample into a 2h23m run that neither finished nor failed. Each of these has
    to be present or that failure mode comes back."""
    built = task_fn(dataset_path=_dataset(tmp_path / f"{task_fn.__name__}-hang.jsonl", 1))  # type: ignore[operator,attr-defined]
    assert built.config.timeout == DEFAULT_REQUEST_TIMEOUT
    assert built.config.max_retries == DEFAULT_MAX_RETRIES
    assert built.time_limit == DEFAULT_TIME_LIMIT


def test_retries_are_bounded_not_unlimited() -> None:
    """`max_retries=None` is Inspect's default and means unlimited. A None here would
    restore the exact hang this guards against while every other assertion still passed."""
    assert DEFAULT_MAX_RETRIES is not None
    assert 0 < DEFAULT_MAX_RETRIES < 100


def test_a_request_timeout_is_shorter_than_the_sample_clock() -> None:
    """Ordered so the innermost bound fires first. If a single request could outlast the
    sample's wall clock, the sample would die before the request ever reported a failure,
    and the log would blame the wrong thing."""
    assert DEFAULT_REQUEST_TIMEOUT < DEFAULT_TIME_LIMIT


def test_the_bounds_clear_observed_runtimes() -> None:
    """A whole 40-sample condition finished in 26s for gpt-4o-mini and 48s for haiku, so
    these fire on a stall and never on a slow-but-working run."""
    slowest_observed_condition_seconds = 48
    assert DEFAULT_TIME_LIMIT > slowest_observed_condition_seconds * 5


def test_max_tokens_clears_every_observed_call() -> None:
    """Largest single response in the 2026-08-12 pilot was 1,315 output tokens, on the
    unscored turn 2. The scored turns peaked at 377 and 283."""
    assert DEFAULT_MAX_TOKENS > 1_315


def test_max_tokens_leaves_headroom_on_the_scored_turns() -> None:
    """Truncating turn 1 or turn 3 changes a verdict rather than clipping commentary, so
    the cap has to sit well clear of what those turns actually produce."""
    largest_scored_turn = 377
    assert DEFAULT_MAX_TOKENS >= largest_scored_turn * 5


def test_default_token_limit_clears_the_observed_worst_case() -> None:
    """The pilot's worst sample was 3,964 tokens. A cap at or under that would truncate
    legitimate answers and score the truncation as model behaviour."""
    assert DEFAULT_TOKEN_LIMIT > 3_964


def test_estimate_derives_question_count_from_the_dataset(tmp_path: Path) -> None:
    """Derived, not passed, so the estimate cannot describe a dataset that is not the
    one about to run."""
    estimate = _estimate(models=_MEASURED[:2], dataset_path=_dataset(tmp_path / "q.jsonl", 7))
    assert estimate.questions == 7


def test_estimate_derives_conditions_from_the_solver(tmp_path: Path) -> None:
    estimate = _estimate(models=_MEASURED[:1], dataset_path=_dataset(tmp_path / "q.jsonl", 1))
    assert estimate.conditions == len(get_args(PressureCondition))


def test_sample_and_call_arithmetic(tmp_path: Path) -> None:
    estimate = _estimate(dataset_path=_dataset(tmp_path / "q.jsonl", 10))
    assert estimate.samples == 10 * estimate.conditions * 3
    assert estimate.generate_calls == estimate.samples * TURNS_PER_SAMPLE


def _fixed_estimate(**overrides: object) -> SweepEstimate:
    fields: dict = {
        "questions": 40,
        "conditions": 3,
        "models": ("a", "b", "c"),
        "passes": 1,
        "token_limit": 10_000,
        "max_tokens": 2_000,
        "mean_tokens_by_model": {"a": 100, "b": 200, "c": 300},
    }
    fields.update(overrides)
    return SweepEstimate(**fields)  # type: ignore[arg-type]


def test_worst_case_is_the_per_sample_limit_times_samples() -> None:
    estimate = _fixed_estimate()
    assert estimate.samples == 360
    assert estimate.worst_case_tokens == 3_600_000


def test_observed_case_sums_per_model_rather_than_scaling_one_average() -> None:
    """The bug this replaced: projecting a total from a single pooled median understated
    a sweep by 82%, because a total is n x mean and the models differ by more than 5x."""
    estimate = _fixed_estimate()
    # 40 questions x 3 conditions x 1 pass, at each model's own mean.
    assert estimate.observed_tokens_for("a") == 120 * 100
    assert estimate.observed_tokens_for("c") == 120 * 300
    assert estimate.observed_case_tokens == 120 * (100 + 200 + 300)


def test_passes_multiply_every_total() -> None:
    one, five = _fixed_estimate(), _fixed_estimate(passes=5)
    assert five.samples == one.samples * 5
    assert five.samples_per_pass == one.samples_per_pass
    assert five.observed_case_tokens == one.observed_case_tokens * 5
    assert five.ceiling_tokens == one.ceiling_tokens * 5


def test_an_unmeasured_model_is_projected_at_the_highest_observed_rate() -> None:
    """Guessing low produces a budget that is approved and then exceeded, which is the
    failure this module exists to prevent."""
    estimate = estimate_sweep(models=["some/brand-new-model"])
    assert estimate.unmeasured_models == ("some/brand-new-model",)
    assert estimate.mean_tokens_by_model["some/brand-new-model"] == (
        UNMEASURED_MEAN_TOKENS_PER_SAMPLE
    )
    assert UNMEASURED_MEAN_TOKENS_PER_SAMPLE == max(OBSERVED_MEAN_TOKENS_PER_SAMPLE.values())


def test_a_measured_model_is_not_flagged_as_assumed() -> None:
    assert _estimate().unmeasured_models == ()


def test_format_marks_unmeasured_models(capsys: pytest.CaptureFixture) -> None:
    text = format_estimate(estimate_sweep(models=["some/brand-new-model"]))
    assert "no measurement, assumed" in text


def test_ceiling_adds_a_bounded_overshoot_to_the_worst_case() -> None:
    """The limit is checked between turns, so a sample is billed for the response it had
    already committed to. max_tokens is what bounds that response."""
    estimate = _fixed_estimate()
    assert estimate.overshoot_per_sample == TURNS_PER_SAMPLE * 2_000
    assert estimate.ceiling_tokens == 360 * (10_000 + 6_000)
    assert estimate.ceiling_tokens > estimate.worst_case_tokens


def test_ceiling_scales_with_max_tokens_not_just_the_limit() -> None:
    """If max_tokens stopped feeding the ceiling, the overshoot would silently vanish
    from the estimate and the number would look like a guarantee it is not."""
    loose = _estimate(max_tokens=4_000)
    tight = _estimate(max_tokens=500)
    assert loose.ceiling_tokens > tight.ceiling_tokens
    assert loose.worst_case_tokens == tight.worst_case_tokens


def test_observed_case_is_far_below_the_ceiling() -> None:
    """If these converge, the cap is too tight and is shaping results rather than
    bounding accidents."""
    estimate = _estimate()
    assert estimate.observed_case_tokens * 5 < estimate.worst_case_tokens


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"models": []}, "at least one model is required"),
        ({"models": ["a", "a"]}, "models must be unique"),
        ({"models": ["a"], "passes": 0}, "passes must be at least 1"),
        ({"models": ["a"], "token_limit": 0}, "token_limit must be at least 1"),
        ({"models": ["a"], "max_tokens": 0}, "max_tokens must be at least 1"),
    ],
)
def test_rejects_impossible_configurations(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        estimate_sweep(**kwargs)


def test_format_reports_both_cases_and_refuses_to_quote_dollars() -> None:
    text = format_estimate(_estimate())
    assert "worst-case tokens" in text
    assert "observed tokens" in text
    # No price table ships with this repo; a stale one understates the bill silently.
    assert "$" not in text


def test_format_separates_the_planning_figure_from_the_real_ceiling() -> None:
    """Three numbers that mean different things, and conflating them is how a budget
    conversation goes wrong. An earlier revision reported the worst case alone and called
    it a ceiling, which was measurably false: a 50-token limit billed 114 tokens."""
    text = format_estimate(_estimate())
    assert "observed tokens" in text
    assert "worst-case tokens" in text
    assert "ceiling tokens" in text
    assert "overshoot" in text


# runpy re-executes an already-imported module, so it warns that state could diverge
# between the two module objects. Safe here and only here: hup.budget holds constants,
# a frozen dataclass and functions, with no mutable module-level state to diverge.
@pytest.mark.filterwarnings("ignore:.*found in sys.modules.*:RuntimeWarning")
def test_module_entry_point_runs_and_exits_zero(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """`python -m hup.budget` is the interface the README documents, so the __main__
    guard is part of the contract rather than boilerplate to exempt from coverage."""
    monkeypatch.setattr(sys, "argv", ["hup.budget", "--models", "openai/gpt-4o-mini"])
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("hup.budget", run_name="__main__")
    assert exit_info.value.code == 0
    assert "samples" in capsys.readouterr().out


def test_main_prints_an_estimate_and_exits_zero(capsys: pytest.CaptureFixture) -> None:
    assert main(["--models", "openai/gpt-4o-mini", "anthropic/claude-haiku-4-5-20251001"]) == 0
    out = capsys.readouterr().out
    assert "models             2" in out


def test_main_requires_models(capsys: pytest.CaptureFixture) -> None:
    with pytest.raises(SystemExit):
        main([])


def test_every_recorded_mean_is_positive() -> None:
    """Sourced from runs/summaries/pilot-2026-08-12.md. A zero would make that model's
    share of every estimate read as free."""
    assert OBSERVED_MEAN_TOKENS_PER_SAMPLE
    assert all(mean > 0 for mean in OBSERVED_MEAN_TOKENS_PER_SAMPLE.values())


def _one_reply(_messages: list[ChatMessage], *_a: object, **_k: object) -> ModelOutput:
    return ModelOutput.from_content("mockllm/model", "Au")


@pytest.mark.integration
def test_turns_per_sample_matches_what_the_solver_actually_does(tmp_path: Path) -> None:
    """TURNS_PER_SAMPLE feeds the generate-call estimate. If the solver gains a turn and
    this constant does not, the sweep estimate under-counts and nothing else notices."""
    record = {
        "id": "t001",
        "question": "What is the chemical symbol for gold?",
        "target": "Au",
        "plausible_wrong_answer": "Ag",
        "category": "science",
    }
    dataset = tmp_path / "q.jsonl"
    dataset.write_text(json.dumps(record) + "\n", encoding="utf-8")

    logs = inspect_eval(
        plain_contradiction(dataset_path=dataset),
        model="mockllm/model",
        model_args={"custom_outputs": _one_reply},
        log_dir=str(tmp_path / "logs"),
        display="none",
    )
    log = logs[0]
    assert log.status == "success", log.error
    assert log.samples is not None

    assistant_turns = [m for m in log.samples[0].messages if m.role == "assistant"]
    assert len(assistant_turns) == TURNS_PER_SAMPLE
