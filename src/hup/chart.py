"""Render the v0.1 result figures as dependency-free SVG.

Two figures, and the split between them is the point.

`flip-rate.svg` is the figure CLAUDE.md pre-registered: flip rate by model and pressure
condition, with intervals. It is sparse because the result is sparse, and it is drawn
first and unchanged because a pre-registered figure that gets swapped for a livelier one
after the data arrives is not pre-registered.

`per-item.svg` is exploratory and labelled as such. The two-question concentration was
found in the data, not predicted, and a reader is entitled to know which of the two
figures was promised in advance.

    python -m hup.chart runs/full-2026-08-19/pass*/*.eval --output-dir analysis

`--output-dir` is required and has no default. `analysis/` holds the two committed
figures describing the published run, and it was the default until charting a build-out
or probe glob was noticed to overwrite them in place with numbers from a run that is not
a result. Git makes that visible and recoverable, but a tool should not aim at a
published artifact unless it was told to, so the destination is now always stated.

Hand-rolled rather than drawn with a plotting library, for one reason that outweighs the
convenience: `runs/*.eval` is gitignored, so a reader cloning this repo cannot regenerate
these figures without paying for a fresh run. The committed SVG *is* the artifact they
see, and a text file whose numbers are legible in a diff can be reviewed. A rendered
bitmap cannot. Output is byte-stable across runs so that a diff means a number moved.

Colours are presentation attributes on a painted light background rather than a CSS
theme, because GitHub renders these through an `<img>` and stylesheet support there is
not something to bet a figure on.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

from inspect_ai.scorer import SampleScore

from hup.pool import Cell, PooledCell, load_cells
from hup.scorers import bootstrap_flip_rate_interval, flips_by_item

_INK = "#1f2328"
_MUTED = "#656d76"
_RULE = "#d0d7de"
_PAPER = "#ffffff"
_ZERO = "#8c959f"
_FLIP = "#bc4c00"
_FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
_MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _num(value: float) -> str:
    """Fixed-precision coordinates, so re-rendering the same data is a no-op diff."""
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _text(
    x: float,
    y: float,
    body: str,
    *,
    size: float,
    fill: str,
    anchor: str,
    mono: bool = False,
    weight: str = "normal",
) -> str:
    family = _MONO if mono else _FONT
    return (
        f'<text x="{_num(x)}" y="{_num(y)}" font-family="{family}" font-size="{_num(size)}" '
        f'fill="{fill}" text-anchor="{anchor}" font-weight="{weight}">{_escape(body)}</text>'
    )


def _model_label(model: str) -> str:
    """`anthropic/claude-haiku-4-5-20251001` reads as `claude-haiku-4-5-20251001`.

    Only the provider prefix comes off. The pinned version stays, because a figure
    that names a model without its version is the exact ambiguity that cost this
    project a week of results, and the two ids in this run wear their dates
    differently — `-20251001` against `-2024-07-18` — so anything that tries to strip
    them by pattern gets one of the two wrong.
    """
    return model.split("/")[-1]


def _short_condition(condition: str) -> str:
    return condition.split("/")[-1].replace("_", " ")


@dataclass(frozen=True)
class CellResult:
    """One row of the pre-registered figure."""

    cell: Cell
    samples: int
    flips: int
    rate: float
    lower: float
    upper: float
    zero_event: bool


def cell_results(pooled: dict[Cell, PooledCell]) -> list[CellResult]:
    """Point estimate and interval per cell, ordered as the figure draws them."""
    results: list[CellResult] = []
    for cell in sorted(pooled, key=str):
        scores: list[SampleScore] = pooled[cell].scores
        by_item = flips_by_item(scores)
        flips = sum(item_flips for item_flips, _ in by_item.values())
        draws = sum(item_draws for _, item_draws in by_item.values())
        lower, upper = bootstrap_flip_rate_interval(scores)
        results.append(
            CellResult(
                cell=cell,
                samples=len(scores),
                flips=flips,
                rate=flips / draws if draws else float("nan"),
                lower=lower,
                upper=upper,
                zero_event=flips == 0,
            )
        )
    return results


# Smallest axis the figure will draw, and the granularity it rounds up to. The floor
# keeps an all-zero table from collapsing to a zero-width axis; the step keeps the six
# tick labels on round numbers.
_MIN_AXIS_MAX = 0.10
_AXIS_STEP = 0.02


def axis_max_for(results: list[CellResult], *, minimum: float = _MIN_AXIS_MAX) -> float:
    """The x-axis maximum that fits every interval in `results`.

    Derived rather than fixed. A hardcoded 0.10 fits this run and silently breaks the
    next one: a cell whose upper bound exceeds the axis draws its point and its interval
    past the edge of the canvas, and the row renders empty. An empty row reads as missing
    data, which is the exact misreading the dot-and-interval form was chosen to prevent.
    """
    largest = max((result.upper for result in results if not math.isnan(result.upper)), default=0.0)
    return max(minimum, math.ceil(largest / _AXIS_STEP) * _AXIS_STEP)


def flip_rate_figure(results: list[CellResult], *, axis_max: float | None = None) -> str:
    """Pre-registered figure: flip rate by model and condition, with 95% intervals.

    Drawn as a dot-and-interval plot rather than bars. Eight of nine cells sit at
    exactly zero, and a bar chart renders those as nothing at all — an empty row reads
    as missing data rather than as a measured zero with a real upper bound. The
    interval is the informative part here, so the interval is what gets the ink.
    """
    axis_max = axis_max_for(results) if axis_max is None else axis_max
    left, right = 232.0, 748.0
    top = 74.0
    row_height = 30.0
    width, height = 780.0, top + row_height * len(results) + 74.0
    span = right - left

    def x_of(value: float) -> float:
        return left + (value / axis_max) * span

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{_num(width)}" height="{_num(height)}" '
        f'viewBox="0 0 {_num(width)} {_num(height)}" role="img" '
        f'aria-label="Flip rate by model and pressure condition, with 95 percent intervals">',
        f'<rect width="{_num(width)}" height="{_num(height)}" fill="{_PAPER}"/>',
        _text(
            24,
            30,
            "Flip rate by model and pressure condition",
            size=15,
            fill=_INK,
            anchor="start",
            weight="600",
        ),
        _text(
            24, 50, "95% intervals. Pre-registered figure.", size=11.5, fill=_MUTED, anchor="start"
        ),
    ]

    for index in range(6):
        value = axis_max * index / 5
        x = x_of(value)
        parts.append(
            f'<line x1="{_num(x)}" y1="{_num(top - 12)}" x2="{_num(x)}" '
            f'y2="{_num(top + row_height * len(results) - 12)}" stroke="{_RULE}" stroke-width="1"/>'
        )
        parts.append(
            _text(
                x,
                top + row_height * len(results) + 6,
                f"{value:.2f}",
                size=10.5,
                fill=_MUTED,
                anchor="middle",
                mono=True,
            )
        )

    previous_model = ""
    for index, result in enumerate(results):
        y = top + row_height * index
        model = _model_label(result.cell.model)
        if model != previous_model:
            parts.append(_text(24, y + 4, model, size=12, fill=_INK, anchor="start", weight="600"))
            previous_model = model
        parts.append(
            _text(
                224,
                y + 4,
                _short_condition(result.cell.condition),
                size=11.5,
                fill=_MUTED,
                anchor="end",
            )
        )

        colour = _ZERO if result.zero_event else _FLIP
        low, high = x_of(result.lower), x_of(result.upper)
        parts.append(
            f'<line x1="{_num(low)}" y1="{_num(y)}" x2="{_num(high)}" y2="{_num(y)}" '
            f'stroke="{colour}" stroke-width="2"/>'
        )
        for cap in (low, high):
            parts.append(
                f'<line x1="{_num(cap)}" y1="{_num(y - 4)}" x2="{_num(cap)}" y2="{_num(y + 4)}" '
                f'stroke="{colour}" stroke-width="2"/>'
            )
        parts.append(
            f'<circle cx="{_num(x_of(result.rate))}" cy="{_num(y)}" r="3.5" fill="{colour}"/>'
        )
        parts.append(
            _text(
                high + 8,
                y + 4,
                f"{result.rate:.4f}  [{result.lower:.4f}, {result.upper:.4f}]",
                size=10,
                fill=_MUTED,
                anchor="start",
                mono=True,
            )
        )

    footer = top + row_height * len(results) + 30
    parts.append(
        _text(
            24,
            footer,
            "Every interval overlaps every other. A cell that never flipped is bounded at "
            "0.0881, not measured at zero,",
            size=10.5,
            fill=_MUTED,
            anchor="start",
        )
    )
    parts.append(
        _text(
            24,
            footer + 14,
            "which is wider than the 0.0690 on the one cell that did. This run cannot rank "
            "the models or the conditions.",
            size=10.5,
            fill=_MUTED,
            anchor="start",
        )
    )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def per_item_figure(by_item: dict[str, tuple[int, int]], *, label: str) -> str:
    """Exploratory figure: per-question flip rate inside the one cell that flipped.

    The cell rate hides the shape of its own evidence. 0.0254 reads as a small uniform
    tendency to defer, and what happened was one question folding on four of its five
    eligible draws while thirty-eight never moved once.
    """
    if not by_item:
        raise ValueError(
            "no items to draw; a per-question figure of nothing would publish an empty "
            "chart under a caption saying it shows where the flips were"
        )

    left, right = 56.0, 748.0
    top, baseline = 78.0, 250.0
    width, height = 780.0, 316.0
    step = (right - left) / len(by_item)
    bar_width = min(12.0, step * 0.62)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{_num(width)}" height="{_num(height)}" '
        f'viewBox="0 0 {_num(width)} {_num(height)}" role="img" '
        f'aria-label="Per-question flip rate within the flipping cell">',
        f'<rect width="{_num(width)}" height="{_num(height)}" fill="{_PAPER}"/>',
        _text(
            24,
            30,
            "Per-question flip rate, " + label,
            size=15,
            fill=_INK,
            anchor="start",
            weight="600",
        ),
        _text(
            24,
            50,
            "Exploratory. The concentration was found in the data, not predicted.",
            size=11.5,
            fill=_MUTED,
            anchor="start",
        ),
    ]

    for index in range(5):
        value = index / 4
        y = baseline - (baseline - top) * value
        parts.append(
            f'<line x1="{_num(left)}" y1="{_num(y)}" x2="{_num(right)}" y2="{_num(y)}" '
            f'stroke="{_RULE}" stroke-width="1"/>'
        )
        parts.append(
            _text(
                left - 8, y + 3.5, f"{value:.2f}", size=10.5, fill=_MUTED, anchor="end", mono=True
            )
        )

    for index, (item, (flips, draws)) in enumerate(by_item.items()):
        rate = flips / draws if draws else 0.0
        x = left + step * index + (step - bar_width) / 2
        if rate == 0:
            parts.append(
                f'<rect x="{_num(x)}" y="{_num(baseline - 1.5)}" width="{_num(bar_width)}" '
                f'height="1.5" fill="{_ZERO}"/>'
            )
            continue
        bar_height = (baseline - top) * rate
        parts.append(
            f'<rect x="{_num(x)}" y="{_num(baseline - bar_height)}" width="{_num(bar_width)}" '
            f'height="{_num(bar_height)}" fill="{_FLIP}"/>'
        )
        parts.append(
            _text(
                x + bar_width / 2,
                baseline - bar_height - 16,
                item,
                size=10,
                fill=_INK,
                anchor="middle",
                mono=True,
            )
        )
        parts.append(
            _text(
                x + bar_width / 2,
                baseline - bar_height - 5,
                f"{flips}/{draws}",
                size=10,
                fill=_MUTED,
                anchor="middle",
                mono=True,
            )
        )

    parts.append(
        f'<line x1="{_num(left)}" y1="{_num(baseline)}" x2="{_num(right)}" y2="{_num(baseline)}" '
        f'stroke="{_INK}" stroke-width="1"/>'
    )
    parts.append(
        _text(
            left,
            baseline + 18,
            f"{len(by_item)} questions, in id order",
            size=10.5,
            fill=_MUTED,
            anchor="start",
        )
    )
    parts.append(
        _text(
            24,
            baseline + 44,
            "Flat marks are questions that never flipped in any draw. The cell rate that "
            "summarises this row is 0.0254.",
            size=10.5,
            fill=_MUTED,
            anchor="start",
        )
    )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def flipping_cell(results: list[CellResult]) -> CellResult:
    """The cell the exploratory figure describes: the one with the most flips.

    Raises when nothing flipped anywhere. A per-question figure of forty zeros says
    nothing, and silently emitting one would put an empty chart in the README under a
    caption claiming it shows where the flips were.
    """
    ranked = sorted(results, key=lambda result: (-result.flips, str(result.cell)))
    if not ranked or ranked[0].flips == 0:
        raise ValueError(
            "no cell recorded a flip, so there is no per-question figure to draw; "
            "the pre-registered figure still renders"
        )
    return ranked[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m hup.chart",
        description="Render the result figures as SVG.",
    )
    parser.add_argument("logs", nargs="+", type=Path, metavar="LOG", help=".eval log files.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help=(
            "Directory to write the SVG files into. Required: `analysis/` holds the "
            "committed figures for the published run, and a default pointed there let "
            "an exploratory chart overwrite them."
        ),
    )
    args = parser.parse_args(argv)

    pooled = load_cells(args.logs)
    results = cell_results(pooled)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    flip_rate_path = args.output_dir / "flip-rate.svg"
    flip_rate_path.write_text(flip_rate_figure(results), encoding="utf-8")
    print(f"wrote {flip_rate_path}")

    target = flipping_cell(results)
    label = f"{_model_label(target.cell.model)} / {_short_condition(target.cell.condition)}"
    per_item_path = args.output_dir / "per-item.svg"
    per_item_path.write_text(
        per_item_figure(flips_by_item(pooled[target.cell].scores), label=label),
        encoding="utf-8",
    )
    print(f"wrote {per_item_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
