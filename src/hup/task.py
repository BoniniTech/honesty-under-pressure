"""Inspect task definition for the honesty-under-pressure eval."""

from __future__ import annotations

from pathlib import Path

from inspect_ai import Task, task

from hup.dataset import DEFAULT_DATA_PATH, load_dataset
from hup.scorers import flip_scorer
from hup.solvers import PressureCondition, pressure_solver


@task
def honesty_under_pressure(
    condition: PressureCondition = "plain_contradiction",
    dataset_path: str | Path = DEFAULT_DATA_PATH,
) -> Task:
    """The full eval: ask, push back under `condition`, then score the final answer.

    `dataset_path` is coerced to Path because Inspect hands `-T` task arguments
    through from the command line as strings.
    """
    return Task(
        dataset=load_dataset(Path(dataset_path)),
        solver=pressure_solver(condition),
        scorer=flip_scorer(),
    )
