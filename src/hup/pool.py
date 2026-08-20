"""Pool metrics across repeated eval passes.

A single pass measures each model x condition x item cell once, and cells are not
stable: one cell re-run five times gave two flips, two ambiguous and one hold, with
nothing varying but sampling. See runs/summaries/verdict-instability-2026-08-19.md.
So a per-cell number from one sweep is one draw from a distribution, not a
measurement of it.

Inspect's `--epochs` cannot do the repeating. Its reducers keep `metadata` from the
first epoch only — `_reduced_score` in `inspect_ai/scorer/_reducer/reducer.py` assigns
`metadata=scores[0].metadata` with none of the equality checking it applies to `answer`
and `explanation` — and every metric in `hup.scorers` reads metadata. Epochs would
report pass-1 results at N times the spend. Passes are therefore separate evals, and
this module pools them.

    python -m hup.pool runs/d5/*.eval

Metrics are the same functions the scorer registers, applied to the union of samples in
a cell, so a pooled number and a single-pass number mean the same thing over different
amounts of evidence.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

from inspect_ai.log import read_eval_log
from inspect_ai.scorer import SampleScore, Score, Value

from hup.scorers import METRIC_FACTORIES


class PoolingError(RuntimeError):
    """Raised when the logs handed in cannot be pooled into an honest number."""


@dataclass(frozen=True)
class Cell:
    """One model against one pressure condition, the unit a flip rate is reported for."""

    model: str
    condition: str

    def __str__(self) -> str:
        return f"{self.model} / {self.condition}"


@dataclass
class PooledCell:
    cell: Cell
    scores: list[SampleScore]
    passes: int
    samples_per_pass: list[int]


def _sample_scores(log_samples: list) -> list[SampleScore]:
    scores: list[SampleScore] = []
    for sample in log_samples:
        if not sample.scores:
            raise PoolingError(f"sample {sample.id!r} carries no score")
        if len(sample.scores) != 1:
            raise PoolingError(
                f"sample {sample.id!r} has {len(sample.scores)} scorers "
                f"({sorted(sample.scores)}); pooling assumes flip_scorer alone"
            )
        evaluated = next(iter(sample.scores.values()))
        scores.append(
            SampleScore(
                score=Score(value=evaluated.value, metadata=evaluated.metadata),
                sample_id=sample.id,
            )
        )
    return scores


def load_cells(paths: list[Path]) -> dict[Cell, PooledCell]:
    """Group every sample in every log by (model, condition).

    Refuses a log that did not finish. A crashed pass contributes a partial cell, which
    would weight whichever items happened to run before the failure more heavily than
    the rest — a silent bias, and exactly the kind that survives into a published table.
    """
    if not paths:
        raise PoolingError("no log files given")

    pooled: dict[Cell, PooledCell] = {}
    for path in paths:
        log = read_eval_log(str(path))
        if log.status != "success":
            raise PoolingError(f"{path.name}: status is {log.status!r}, not 'success'")
        if not log.samples:
            raise PoolingError(f"{path.name}: no samples")

        cell = Cell(model=log.eval.model, condition=log.eval.task)
        scores = _sample_scores(log.samples)
        entry = pooled.setdefault(cell, PooledCell(cell, [], 0, []))
        entry.scores.extend(scores)
        entry.passes += 1
        entry.samples_per_pass.append(len(scores))

    return pooled


def uneven_cells(pooled: dict[Cell, PooledCell]) -> list[Cell]:
    """Cells whose passes did not all cover the same number of samples.

    Reported rather than raised. Unequal passes are usually a run that was cut short,
    and the pooled number is still computable — it just weights some items more than
    others, which a reader has to be told rather than left to assume.
    """
    return [cell for cell, entry in pooled.items() if len(set(entry.samples_per_pass)) > 1]


def pooled_metrics(scores: list[SampleScore]) -> dict[str, Value]:
    """Every metric the scorer registers, over the pooled samples."""
    return {name: factory()(scores) for name, factory in METRIC_FACTORIES.items()}


# Column headers, kept short so nine cells and eight metrics fit a terminal. The full
# names are the keys of METRIC_FACTORIES and are printed as a legend beneath the table.
_ABBREVIATIONS = {
    "flip_rate": "flip",
    "flip_rate_ci_lower": "ci_lo",
    "flip_rate_ci_upper": "ci_hi",
    "initial_accuracy": "init_acc",
    "ambiguous_rate": "ambig",
    "truncated_rate": "trunc",
    "eligible_rate": "elig",
    "excluded_wrong_final_rate": "exc_wrong",
}


def _cell_value(value: Value) -> str:
    number = float(value)  # type: ignore[arg-type]
    return "nan" if math.isnan(number) else f"{number:.4f}"


def format_pooled(pooled: dict[Cell, PooledCell]) -> str:
    names = list(METRIC_FACTORIES)
    missing = [name for name in names if name not in _ABBREVIATIONS]
    if missing:
        raise PoolingError(
            f"no column header for metric(s) {missing}; add them to _ABBREVIATIONS so a "
            "metric registered on the scorer cannot go missing from the pooled table"
        )

    label_width = max(len(str(cell)) for cell in pooled)
    columns = [_ABBREVIATIONS[name] for name in names]
    widths = [max(len(column), 9) for column in columns]

    header = f"{'cell':<{label_width}}  {'pass':>4} {'n':>5}  " + " ".join(
        f"{column:>{width}}" for column, width in zip(columns, widths, strict=True)
    )
    lines = [header, "-" * len(header)]

    for cell in sorted(pooled, key=str):
        entry = pooled[cell]
        values = pooled_metrics(entry.scores)
        rendered = " ".join(
            f"{_cell_value(values[name]):>{width}}"
            for name, width in zip(names, widths, strict=True)
        )
        lines.append(
            f"{str(cell):<{label_width}}  {entry.passes:>4} {len(entry.scores):>5}  {rendered}"
        )

    lines.append("")
    lines.append("  ".join(f"{_ABBREVIATIONS[name]}={name}" for name in names))

    uneven = uneven_cells(pooled)
    if uneven:
        lines.append("")
        lines.append("WARNING: passes covered different sample counts in these cells, so items")
        lines.append("are not weighted equally. Re-run the short passes before publishing.")
        for cell in sorted(uneven, key=str):
            lines.append(f"  {cell}: {pooled[cell].samples_per_pass}")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m hup.pool",
        description="Pool flip metrics across repeated eval passes.",
    )
    parser.add_argument("logs", nargs="+", type=Path, metavar="LOG", help=".eval log files.")
    args = parser.parse_args(argv)

    print(format_pooled(load_cells(args.logs)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
