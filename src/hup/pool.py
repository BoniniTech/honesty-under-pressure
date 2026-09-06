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

    python -m hup.pool runs/full-2026-08-19/*.eval

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

from hup.scorers import (
    METRIC_FACTORIES,
    ambiguity_by_item,
    flips_by_round,
    recorded_rounds,
)


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
    """Every sample's one score, carrying the sample's own metadata alongside it.

    `sample_metadata` is what makes a stratum breakdown possible. The scorer writes
    verdicts and flags into *score* metadata and never copies the item's `stratum` or
    `registered` labels across, because those describe the question rather than the
    answer. They live in sample metadata, written when the sample runs, so pooling has
    to read them from the log rather than from the score.

    The consequence is that no re-score can backfill them: a log recorded before the
    labels existed carries samples that never had them. `format_stratum_breakdown` says
    so in words rather than reporting one unlabelled arm.
    """
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
                sample_metadata=sample.metadata,
            )
        )
    return scores


def load_cells(paths: list[Path]) -> dict[Cell, PooledCell]:
    """Group every sample in every log by (model, condition).

    Refuses a log that did not finish. A crashed pass contributes a partial cell, which
    would weight whichever items happened to run before the failure more heavily than
    the rest — a silent bias, and exactly the kind that survives into a published table.

    Refuses a cell whose passes ran different escalation depths, for a sharper version of
    the same reason. A one-round pass and a three-round pass produce the same cells, the
    same sample counts and the same columns, so their mixture is invisible in the output
    while the flip rate it produces describes neither run.
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

    for cell, entry in pooled.items():
        try:
            recorded_rounds(entry.scores)
        except ValueError as error:
            raise PoolingError(f"{cell}: {error}") from error

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
    "unfinished_rate": "unfin",
    "eligible_rate": "elig",
    "excluded_wrong_final_rate": "exc_wrong",
}


def _cell_value(value: Value) -> str:
    number = float(value)  # type: ignore[arg-type]
    return "nan" if math.isnan(number) else f"{number:.4f}"


def format_round_breakdown(pooled: dict[Cell, PooledCell]) -> list[str]:
    """Lines saying where in the escalation ladder each cell's flips happened.

    Reported beneath the metric table rather than as a column in it, because it is a
    breakdown of the flip rate rather than another rate: the round counts sum to the
    flips the cell already reported.

    The denominator column is `samples` and not `elig`, which is what the metric table
    calls the same quantity expressed as a rate. Two tables a screen apart printing one
    string under two units left a reader meeting 0.9875 and 158 under the same word.

    A cell of logs that predate the escalation solver says so in words. Printing an empty
    breakdown there would read as a run where no flip landed on any round, which is a
    claim about the models rather than about what the logs recorded.
    """
    breakdowns = {cell: flips_by_round(entry.scores) for cell, entry in pooled.items()}
    depths = {breakdown.rounds for breakdown in breakdowns.values()}

    if depths == {None}:
        return [
            "",
            "flip round: not recorded in these logs. They predate the escalation solver,",
            "which applied one round of pushback without recording where inside it a model",
            "moved. Re-scoring cannot backfill it; only a fresh run can.",
        ]

    width = max(len(str(cell)) for cell in pooled)
    columns = max((breakdown.rounds or 0) for breakdown in breakdowns.values())
    header = (
        f"{'cell':<{width}}  {'depth':>5} {'samples':>7} {'flips':>5}  "
        + " ".join(f"{'r' + str(number):>5}" for number in range(1, columns + 1))
        + f" {'readout':>7} {'recovered':>9}"
    )
    lines = ["", "where the flips happened, over the eligible denominator:", "", header]
    lines.append("-" * len(header))

    for cell in sorted(pooled, key=str):
        breakdown = breakdowns[cell]
        depth = "none" if breakdown.rounds is None else str(breakdown.rounds)
        rounds = " ".join(
            f"{breakdown.by_round.get(number, 0):>5}"
            if number <= (breakdown.rounds or 0)
            else f"{'-':>5}"
            for number in range(1, columns + 1)
        )
        lines.append(
            f"{str(cell):<{width}}  {depth:>5} {breakdown.eligible:>7} "
            f"{breakdown.flips:>5}  {rounds} {breakdown.at_readout:>7} "
            f"{breakdown.recovered:>9}"
        )

    lines += [
        "",
        "samples=this cell's flip denominator as a count, not the metric table's rate",
        "depth=pushback rounds applied  rN=first named the pushback answer at round N",
        "readout=argued through every round, then named it when asked for the answer alone",
        "recovered=named the pushback answer at some round, back on target by the readout",
    ]

    # The rN columns are a lower bound, not a census, and this is the number that says by
    # how much. A round reply that argues its position names both candidates ("it's 24,
    # not 22"), which containment cannot tell from the capitulation that names both, so it
    # scores `ambiguous` and sets no round. The scored turns escape this because the
    # readout asks for the answer alone; the rounds carry no such instruction. Without
    # this line, empty rN columns read as "no model ever folded mid-ladder".
    verdicts: dict[str, int] = {}
    for breakdown in breakdowns.values():
        for verdict, count in breakdown.verdict_counts.items():
            verdicts[verdict] = verdicts.get(verdict, 0) + count
    total_rounds = sum(verdicts.values())
    if total_rounds:
        adjudicable = verdicts.get("correct", 0) + verdicts.get("wrong", 0)
        lines += [
            "",
            "round replies by verdict, over every round of every eligible sample:",
            "  " + "  ".join(f"{verdict}={count}" for verdict, count in sorted(verdicts.items())),
            f"  {adjudicable} of {total_rounds} named exactly one candidate, so the rN columns",
            "  above are a LOWER BOUND on where models first gave in. A reply arguing its",
            '  position names both candidates ("it\'s 24, not 22") and scores ambiguous; the',
            "  readout turn avoids that by asking for the answer alone, the rounds do not.",
        ]

    if len(depths) > 1:
        lines += [
            "",
            "WARNING: these cells ran different escalation depths, so their flip rates are",
            "not comparable with each other. Pool one depth at a time before publishing.",
        ]

    truncated = {cell: b.truncated_rounds for cell, b in breakdowns.items() if b.truncated_rounds}
    if truncated:
        lines += [
            "",
            "WARNING: a pushback round was cut off in these cells, so the round it names is",
            "an upper bound on where the model first moved, not the round itself.",
        ]
        lines += [f"  {cell}: {count}" for cell, count in sorted(truncated.items(), key=str)]

    return lines


# Draws a model must have taken of an item before "ambiguous every time" says anything.
# Below this it is a coin: two draws at a 50% per-draw rate come up ambiguous twice a
# quarter of the time, and the note is meant to point at construction, not at luck.
_PERSISTENT_MIN_DRAWS = 3


def format_ambiguity_breakdown(pooled: dict[Cell, PooledCell]) -> list[str]:
    """Lines naming which dataset items produced each cell's ambiguity.

    Reported beneath the metric table for the same reason the round breakdown is: it
    partitions a number the table already gave rather than adding a rate of its own.
    A per-item metric could not exist anyway — metrics register at import time, and the
    item set is a run-time property of whichever dataset was loaded.

    Only items that produced an ambiguous turn get a row. Forty rows per cell would bury
    the handful that matter, and the count of clean items is printed instead so their
    absence is stated rather than inferred from a short table.
    """
    breakdowns = {cell: ambiguity_by_item(entry.scores) for cell, entry in pooled.items()}

    rows = [
        (cell, item, entry)
        for cell, breakdown in sorted(breakdowns.items(), key=lambda pair: str(pair[0]))
        for item, entry in breakdown.items()
        if entry.ambiguous
    ]
    items = {item for breakdown in breakdowns.values() for item in breakdown}
    dirty = {item for _, item, _ in rows}

    if not rows:
        return [
            "",
            f"ambiguity by item: none. All {len(items)} items named exactly one candidate,",
            "or neither, on both scored turns in every cell.",
        ]

    width = max(max(len(str(cell)) for cell, _, _ in rows), len("cell"))
    item_width = max(max(len(item) for _, item, _ in rows), len("item"))
    header = (
        f"{'cell':<{width}}  {'item':<{item_width}} {'draws':>5} "
        f"{'ambig':>5} {'t1':>5} {'final':>5}"
    )
    lines = ["", "ambiguity by item, over every sample in the cell:", "", header]
    lines.append("-" * len(header))
    for cell, item, entry in rows:
        lines.append(
            f"{str(cell):<{width}}  {item:<{item_width}} {entry.draws:>5} "
            f"{entry.ambiguous:>5} {entry.initial:>5} {entry.final:>5}"
        )

    lines += [
        "",
        "ambig=draws where a scored turn named both candidates, the per-item view of",
        "  ambiguous_rate  t1=turn 1 did, so the sample was dropped before any pushback",
        "  was applied  final=the readout did, so the model was pushed and the answer",
        "  could not be adjudicated. A draw can be both, so t1+final may exceed ambig.",
        f"{len(items) - len(dirty)} of {len(items)} items were clean in every cell "
        "and are omitted.",
    ]

    # An item ambiguous on every draw is ambiguous by construction rather than by chance,
    # and that is a dataset defect rather than a model result: its distractor is a member
    # of a set the correct answer invites listing (issue #47). Calling it out is what makes
    # the next one findable without re-reading transcripts.
    #
    # Aggregated across conditions, unlike the table above. Turn 1 asks the same question
    # in all three conditions, so condition is noise for the ambiguity that matters here
    # and the unit is the model. Three separate 2/2 cells are what a coin produces; one
    # 6/6 model is not.
    per_model: dict[tuple[str, str], list[int]] = {}
    for cell, breakdown in breakdowns.items():
        for item, entry in breakdown.items():
            totals = per_model.setdefault((cell.model, item), [0, 0])
            totals[0] += entry.draws
            totals[1] += entry.ambiguous
    persistent = [
        (model, item, drawn)
        for (model, item), (drawn, ambiguous) in sorted(per_model.items())
        if drawn >= _PERSISTENT_MIN_DRAWS and ambiguous == drawn
    ]
    if persistent:
        lines += [
            "",
            "NOTE: these items were ambiguous on EVERY draw the model took of them, across",
            "all conditions, which is a property of the item rather than of the model. Check",
            "the distractor against the rule in data/README.md before reading the cell's",
            "flip rate as a measurement.",
        ]
        lines += [f"  {model}: {item} ({drawn}/{drawn} draws)" for model, item, drawn in persistent]

    return lines


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
    lines += format_round_breakdown(pooled)
    lines += format_ambiguity_breakdown(pooled)

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
