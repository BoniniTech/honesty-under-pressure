"""Figure rendering.

The figures are committed artifacts. `runs/*.eval` is gitignored, so a reader cannot
regenerate them without a paid run, which makes the checked-in SVG the thing they
actually see. These tests are about the numbers reaching it intact and the markup being
well formed, because nothing downstream will notice if either fails.
"""

from __future__ import annotations

import runpy
import sys
import xml.etree.ElementTree as ElementTree
from pathlib import Path

import pytest
from inspect_ai.scorer import CORRECT, INCORRECT, NOANSWER, SampleScore, Score

from hup.chart import (
    CellResult,
    _escape,
    _model_label,
    _num,
    _short_condition,
    cell_results,
    flip_rate_figure,
    flipping_cell,
    main,
    per_item_figure,
)
from hup.pool import Cell, PooledCell


def _score(initial: str, final: str, item: str) -> SampleScore:
    flipped = initial == "correct" and final == "wrong"
    if flipped:
        value = INCORRECT
    elif initial == "correct" and final == "correct":
        value = CORRECT
    else:
        value = NOANSWER
    return SampleScore(
        score=Score(
            value=value,
            metadata={
                "initial_correct": initial == "correct",
                "final_correct": final == "correct",
                "flipped": flipped,
                "ambiguous": "ambiguous" in (initial, final),
                "truncated": False,
                "initial_verdict": initial,
                "final_verdict": final,
            },
        ),
        sample_id=item,
    )


def _cell(model: str, condition: str, scores: list[SampleScore]) -> dict[Cell, PooledCell]:
    cell = Cell(model=model, condition=condition)
    return {cell: PooledCell(cell, scores, 1, [len(scores)])}


def _d5_shaped() -> list[SampleScore]:
    """The real shape: q010 flips 4 of 5, q016 flips 2 of 4, 38 questions clean."""
    scores = [_score("correct", "wrong", "q010")] * 4 + [_score("correct", "correct", "q010")]
    scores += [_score("correct", "wrong", "q016")] * 2
    scores += [_score("correct", "correct", "q016")] * 2
    for index in range(38):
        scores += [_score("correct", "correct", f"c{index:03d}")] * 6
    return scores


class TestLabels:
    @pytest.mark.parametrize(
        "model, expected",
        [
            ("anthropic/claude-haiku-4-5-20251001", "claude-haiku-4-5-20251001"),
            ("openai/gpt-4o-mini-2024-07-18", "gpt-4o-mini-2024-07-18"),
            ("google/gemini-3.6-flash", "gemini-3.6-flash"),
            ("mockllm/model", "model"),
        ],
    )
    def test_only_the_provider_prefix_comes_off(self, model: str, expected: str) -> None:
        """The pinned version stays on the figure. The two dated ids in this run wear
        their dates differently, so stripping by pattern gets one of them wrong, and a
        figure naming a model without its version is the ambiguity this project already
        lost a week to."""
        assert _model_label(model) == expected

    def test_a_package_qualified_condition_is_stripped(self) -> None:
        assert _short_condition("hup/authority_appeal") == "authority appeal"

    def test_markup_in_a_label_is_escaped(self) -> None:
        """Item ids and model ids reach the figure as text. Nothing validates them
        upstream, and an unescaped angle bracket produces a silently broken SVG."""
        assert _escape("a<b>&c") == "a&lt;b&gt;&amp;c"

    def test_coordinates_are_trimmed_not_padded(self) -> None:
        assert _num(0.0) == "0"
        assert _num(12.5) == "12.5"
        assert _num(12.0) == "12"


class TestCellResults:
    def test_rate_and_interval_come_from_the_scores(self) -> None:
        results = cell_results(_cell("anthropic/x", "authority_appeal", _d5_shaped()))
        assert len(results) == 1
        assert results[0].flips == 6
        assert results[0].rate == pytest.approx(6 / 237)
        assert results[0].zero_event is False
        assert results[0].upper == pytest.approx(0.069, abs=0.005)

    def test_a_cell_that_never_flipped_is_marked_and_bounded(self) -> None:
        scores = [_score("correct", "correct", f"q{i:03d}") for i in range(40)]
        result = cell_results(_cell("openai/x", "plain_contradiction", scores))[0]
        assert result.flips == 0
        assert result.zero_event is True
        assert result.upper == pytest.approx(1 - 0.025 ** (1 / 40))

    def test_a_cell_with_nothing_eligible_reports_nan(self) -> None:
        scores = [_score("ambiguous", "wrong", f"q{i:03d}") for i in range(5)]
        result = cell_results(_cell("openai/x", "plain_contradiction", scores))[0]
        assert result.rate != result.rate  # nan

    def test_cells_are_ordered_the_way_the_figure_draws_them(self) -> None:
        pooled = _cell("b/model", "plain_contradiction", [_score("correct", "correct", "q1")])
        pooled.update(_cell("a/model", "authority_appeal", [_score("correct", "correct", "q1")]))
        assert [str(r.cell) for r in cell_results(pooled)] == sorted(
            str(r.cell) for r in cell_results(pooled)
        )


class TestFlipRateFigure:
    def test_every_cell_reaches_the_figure(self) -> None:
        results = cell_results(_cell("anthropic/x", "authority_appeal", _d5_shaped()))
        svg = flip_rate_figure(results)
        assert "0.0253" in svg or "0.0254" in svg
        assert "authority appeal" in svg

    def test_the_figure_is_well_formed_xml(self) -> None:
        """Hand-rolled markup with no parser in the loop. A malformed attribute renders
        as a blank box in the README and nothing else would catch it."""
        results = cell_results(_cell("anthropic/x", "authority_appeal", _d5_shaped()))
        root = ElementTree.fromstring(flip_rate_figure(results))
        assert root.tag.endswith("svg")

    def test_rendering_twice_gives_identical_bytes(self) -> None:
        """The committed SVG is reviewed as a diff, so a diff has to mean a number
        moved rather than a coordinate jittering."""
        results = cell_results(_cell("anthropic/x", "authority_appeal", _d5_shaped()))
        assert flip_rate_figure(results) == flip_rate_figure(results)

    def test_the_figure_says_the_zero_cells_are_bounded_not_measured(self) -> None:
        results = cell_results(_cell("anthropic/x", "authority_appeal", _d5_shaped()))
        assert "bounded at 0.0881, not measured at zero" in flip_rate_figure(results)

    def test_a_model_is_named_once_across_its_conditions(self) -> None:
        """Three rows per model, one label. A repeated name reads as three models."""
        pooled = _cell("anthropic/x", "authority_appeal", _d5_shaped())
        pooled.update(_cell("anthropic/x", "plain_contradiction", _d5_shaped()))
        svg = flip_rate_figure(cell_results(pooled))
        assert svg.count(">x<") == 1
        assert ">authority appeal<" in svg
        assert ">plain contradiction<" in svg


class TestPerItemFigure:
    def test_flipping_questions_are_labelled_with_their_fractions(self) -> None:
        svg = per_item_figure({"q010": (4, 5), "q016": (2, 4), "c000": (0, 6)}, label="x / y")
        assert ">q010<" in svg
        assert ">4/5<" in svg
        assert ">q016<" in svg
        assert ">2/4<" in svg

    def test_questions_that_never_flipped_are_not_labelled(self) -> None:
        svg = per_item_figure({"q010": (4, 5), "c000": (0, 6)}, label="x / y")
        assert ">c000<" not in svg

    def test_the_figure_is_well_formed_xml(self) -> None:
        root = ElementTree.fromstring(
            per_item_figure({"q010": (4, 5), "c000": (0, 6)}, label="x / y")
        )
        assert root.tag.endswith("svg")

    def test_it_is_labelled_exploratory(self) -> None:
        """The split between this and the pre-registered figure is the reason both
        exist. An unlabelled post-hoc figure beside a pre-registered one reads as if
        both were planned."""
        svg = per_item_figure({"q010": (4, 5)}, label="x / y")
        assert "Exploratory" in svg

    def test_an_item_with_no_eligible_draws_does_not_divide_by_zero(self) -> None:
        svg = per_item_figure({"q010": (0, 0), "q011": (1, 2)}, label="x / y")
        assert ElementTree.fromstring(svg).tag.endswith("svg")


class TestFlippingCell:
    def test_it_picks_the_cell_with_the_most_flips(self) -> None:
        many = cell_results(_cell("a/x", "authority_appeal", _d5_shaped()))[0]
        none = cell_results(
            _cell("b/x", "plain_contradiction", [_score("correct", "correct", "q1")])
        )[0]
        assert flipping_cell([none, many]) is many

    def test_no_flips_anywhere_raises(self) -> None:
        """A per-question figure of forty zeros says nothing, and emitting one anyway
        would put an empty chart under a caption claiming it shows the flips."""
        results = cell_results(
            _cell("b/x", "plain_contradiction", [_score("correct", "correct", "q1")])
        )
        with pytest.raises(ValueError, match="no cell recorded a flip"):
            flipping_cell(results)

    def test_ties_break_on_the_cell_name_so_the_choice_is_stable(self) -> None:
        first = CellResult(Cell("a/x", "c1"), 10, 3, 0.3, 0.0, 0.5, False)
        second = CellResult(Cell("b/x", "c1"), 10, 3, 0.3, 0.0, 0.5, False)
        assert flipping_cell([second, first]) is first


class TestMain:
    def test_it_writes_both_figures(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        pooled = _cell("anthropic/x", "authority_appeal", _d5_shaped())
        monkeypatch.setattr("hup.chart.load_cells", lambda _paths: pooled)

        assert main([str(tmp_path / "fake.eval"), "--output-dir", str(tmp_path / "out")]) == 0

        for name in ("flip-rate.svg", "per-item.svg"):
            written = (tmp_path / "out" / name).read_text(encoding="utf-8")
            assert ElementTree.fromstring(written).tag.endswith("svg")

    @pytest.mark.filterwarnings("ignore:.*found in sys.modules.*:RuntimeWarning")
    def test_module_entry_point_runs(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
    ) -> None:
        pooled = _cell("anthropic/x", "authority_appeal", _d5_shaped())
        # Patched on hup.pool, not hup.chart: runpy imports a fresh hup.chart module,
        # and its `from hup.pool import load_cells` binds whatever hup.pool holds then.
        monkeypatch.setattr("hup.pool.load_cells", lambda _paths: pooled)
        monkeypatch.setattr(
            sys,
            "argv",
            ["hup.chart", str(tmp_path / "fake.eval"), "--output-dir", str(tmp_path)],
        )
        with pytest.raises(SystemExit) as exit_info:
            runpy.run_module("hup.chart", run_name="__main__")
        assert exit_info.value.code == 0
        assert "flip-rate.svg" in capsys.readouterr().out
