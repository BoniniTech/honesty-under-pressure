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
    OBSERVED_MEDIAN_TOKENS_PER_SAMPLE,
    SweepEstimate,
    estimate_sweep,
    format_estimate,
    main,
)
from hup.solvers import TURNS_PER_SAMPLE, PressureCondition
from hup.task import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TOKEN_LIMIT,
    authority_appeal,
    confidence_social,
    plain_contradiction,
)

_TASKS = (plain_contradiction, authority_appeal, confidence_social)


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
    estimate = estimate_sweep(models=2, dataset_path=_dataset(tmp_path / "q.jsonl", 7))
    assert estimate.questions == 7


def test_estimate_derives_conditions_from_the_solver(tmp_path: Path) -> None:
    estimate = estimate_sweep(models=1, dataset_path=_dataset(tmp_path / "q.jsonl", 1))
    assert estimate.conditions == len(get_args(PressureCondition))


def test_sample_and_call_arithmetic(tmp_path: Path) -> None:
    estimate = estimate_sweep(models=3, dataset_path=_dataset(tmp_path / "q.jsonl", 10))
    assert estimate.samples == 10 * estimate.conditions * 3
    assert estimate.generate_calls == estimate.samples * TURNS_PER_SAMPLE


def test_worst_case_is_the_per_sample_limit_times_samples() -> None:
    estimate = SweepEstimate(
        questions=40,
        conditions=3,
        models=3,
        token_limit=10_000,
        max_tokens=2_000,
        observed_median_tokens=536,
    )
    assert estimate.samples == 360
    assert estimate.worst_case_tokens == 3_600_000
    assert estimate.observed_case_tokens == 360 * 536


def test_ceiling_adds_a_bounded_overshoot_to_the_worst_case() -> None:
    """The limit is checked between turns, so a sample is billed for the response it had
    already committed to. max_tokens is what bounds that response."""
    estimate = SweepEstimate(
        questions=40,
        conditions=3,
        models=3,
        token_limit=10_000,
        max_tokens=2_000,
        observed_median_tokens=536,
    )
    assert estimate.overshoot_per_sample == TURNS_PER_SAMPLE * 2_000
    assert estimate.ceiling_tokens == 360 * (10_000 + 6_000)
    assert estimate.ceiling_tokens > estimate.worst_case_tokens


def test_ceiling_scales_with_max_tokens_not_just_the_limit() -> None:
    """If max_tokens stopped feeding the ceiling, the overshoot would silently vanish
    from the estimate and the number would look like a guarantee it is not."""
    loose = estimate_sweep(models=1, max_tokens=4_000)
    tight = estimate_sweep(models=1, max_tokens=500)
    assert loose.ceiling_tokens > tight.ceiling_tokens
    assert loose.worst_case_tokens == tight.worst_case_tokens


def test_observed_case_is_far_below_the_ceiling() -> None:
    """If these converge, the cap is too tight and is shaping results rather than
    bounding accidents."""
    estimate = estimate_sweep(models=3)
    assert estimate.observed_case_tokens * 5 < estimate.worst_case_tokens


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"models": 0}, "models must be at least 1"),
        ({"models": -1}, "models must be at least 1"),
        ({"models": 1, "token_limit": 0}, "token_limit must be at least 1"),
        ({"models": 1, "max_tokens": 0}, "max_tokens must be at least 1"),
    ],
)
def test_rejects_impossible_configurations(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        estimate_sweep(**kwargs)


def test_format_reports_both_cases_and_refuses_to_quote_dollars() -> None:
    text = format_estimate(estimate_sweep(models=3))
    assert "worst-case tokens" in text
    assert "observed tokens" in text
    # No price table ships with this repo; a stale one understates the bill silently.
    assert "$" not in text


def test_format_separates_the_planning_figure_from_the_real_ceiling() -> None:
    """Three numbers that mean different things, and conflating them is how a budget
    conversation goes wrong. An earlier revision reported the worst case alone and called
    it a ceiling, which was measurably false: a 50-token limit billed 114 tokens."""
    text = format_estimate(estimate_sweep(models=3))
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


def test_observed_median_is_recorded_not_zero() -> None:
    """Sourced from runs/summaries/pilot-2026-08-12.md. A zero here would make every
    observed estimate read as free."""
    assert OBSERVED_MEDIAN_TOKENS_PER_SAMPLE > 0


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
