"""Inspect task definitions for the honesty-under-pressure eval.

One task per pressure condition rather than a single task with a `condition`
parameter. `inspect eval` runs every `@task` in a file, so this makes the full
sweep the default:

    inspect eval src/hup/task.py --model <model>          # all three conditions
    inspect eval src/hup/task.py@authority_appeal ...     # one condition

The task name is the condition name, so it lands in the log filename and the
per-condition breakdown joins on it without a separate manifest.
"""

from __future__ import annotations

from pathlib import Path

from inspect_ai import Task, task

from hup.dataset import DEFAULT_DATA_PATH, load_dataset
from hup.scorers import flip_scorer
from hup.solvers import PressureCondition, pressure_solver

# Per-sample ceiling, not a run budget. Inspect has no run-level cap: every limit it
# exposes applies to a single sample, so total spend is bounded by arithmetic over the
# configuration (see hup.budget), and this only stops one sample running away.
#
# The 2026-08-12 pilot measured a per-sample worst case of 3,964 tokens across all three
# turns (gemini-flash-latest; haiku 754, gpt-4o-mini 618) — see runs/summaries/. This is
# ~2.5x that, high enough never to truncate a legitimate answer and low enough that a
# looping sample stops here instead of running unbounded.
#
# Deliberately not cost_limit. Inspect only records cost when the model has price data,
# and all three target models ship `cost=None` in its model database — _model.py guards
# the check with `if total_cost is not None`, so a cost limit would never fire and would
# read as protection that is not there. Supplying prices via --model-cost-config would
# mean a table that silently goes stale against provider pricing.
DEFAULT_TOKEN_LIMIT = 10_000


def _pressure_task(condition: PressureCondition, dataset_path: str | Path) -> Task:
    """Ask, push back under `condition`, then score the final answer.

    `dataset_path` is coerced to Path because Inspect hands `-T` task arguments
    through from the command line as strings.

    The token limit is a task default. `--token-limit` on the command line overrides
    it, so the cap is configurable without editing this file.
    """
    return Task(
        dataset=load_dataset(Path(dataset_path)),
        solver=pressure_solver(condition),
        scorer=flip_scorer(),
        token_limit=DEFAULT_TOKEN_LIMIT,
    )


@task
def plain_contradiction(dataset_path: str | Path = DEFAULT_DATA_PATH) -> Task:
    """Pushback that just asserts the wrong answer, with no source or confidence."""
    return _pressure_task("plain_contradiction", dataset_path)


@task
def authority_appeal(dataset_path: str | Path = DEFAULT_DATA_PATH) -> Task:
    """Pushback that attributes the wrong answer to a professor and a textbook."""
    return _pressure_task("authority_appeal", dataset_path)


@task
def confidence_social(dataset_path: str | Path = DEFAULT_DATA_PATH) -> Task:
    """Pushback that states certainty and claims everyone agrees."""
    return _pressure_task("confidence_social", dataset_path)
