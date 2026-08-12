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


def _pressure_task(condition: PressureCondition, dataset_path: str | Path) -> Task:
    """Ask, push back under `condition`, then score the final answer.

    `dataset_path` is coerced to Path because Inspect hands `-T` task arguments
    through from the command line as strings.
    """
    return Task(
        dataset=load_dataset(Path(dataset_path)),
        solver=pressure_solver(condition),
        scorer=flip_scorer(),
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
