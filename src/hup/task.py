"""Inspect task definition for the honesty-under-pressure eval."""

from __future__ import annotations

from inspect_ai import Task, task

from hup.dataset import load_dataset
from hup.scorers import flip_scorer
from hup.solvers import PressureCondition, pressure_solver


@task
def honesty_under_pressure(condition: PressureCondition = "plain_contradiction") -> Task:
    return Task(
        dataset=load_dataset(),
        solver=pressure_solver(condition),
        scorer=flip_scorer(),
    )
