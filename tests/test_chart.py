"""Figure rendering.

The figures are committed artifacts. `runs/*.eval` is gitignored, so a reader cannot
regenerate them without a paid run, which makes the checked-in SVG the thing they
actually see. These tests are about the numbers reaching it intact and the markup being
well formed, because nothing downstream will notice if either fails.
"""

from __future__ import annotations

import re
import runpy
import sys
import xml.etree.ElementTree as ElementTree
from pathlib import Path

import pytest
from inspect_ai.scorer import CORRECT, INCORRECT, NOANSWER, SampleScore, Score

from hup.chart import (
    ArmResult,
    CellResult,
    _escape,
    _model_label,
    _num,
    _short_condition,
    arm_results,
    axis_max_for,
    by_stratum_figure,
    cell_results,
    flip_rate_figure,
    flip_rate_footer,
    flipping_cell,
    main,
    per_item_figure,
)
from hup.pool import Arm, Cell, PooledCell


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


def _full_run_shaped() -> list[SampleScore]:
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
        results = cell_results(_cell("anthropic/x", "authority_appeal", _full_run_shaped()))
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
        results = cell_results(_cell("anthropic/x", "authority_appeal", _full_run_shaped()))
        svg = flip_rate_figure(results)
        assert "0.0253" in svg or "0.0254" in svg
        assert "authority appeal" in svg

    def test_the_figure_is_well_formed_xml(self) -> None:
        """Hand-rolled markup with no parser in the loop. A malformed attribute renders
        as a blank box in the README and nothing else would catch it."""
        results = cell_results(_cell("anthropic/x", "authority_appeal", _full_run_shaped()))
        root = ElementTree.fromstring(flip_rate_figure(results))
        assert root.tag.endswith("svg")

    def test_rendering_twice_gives_identical_bytes(self) -> None:
        """The committed SVG is reviewed as a diff, so a diff has to mean a number
        moved rather than a coordinate jittering."""
        results = cell_results(_cell("anthropic/x", "authority_appeal", _full_run_shaped()))
        assert flip_rate_figure(results) == flip_rate_figure(results)

    def test_the_figure_says_the_zero_cells_are_bounded_not_measured(self) -> None:
        """Needs a fixture that actually holds a zero cell. This test used to assert the
        sentence against `_full_run_shaped()`, which is one cell that flipped six times,
        so it pinned a claim the figure had no basis for."""
        pooled = _cell("anthropic/x", "authority_appeal", _full_run_shaped())
        pooled.update(_cell("anthropic/x", "confidence_social", _clean_cell_scores()))
        svg = flip_rate_figure(cell_results(pooled))
        assert "never flipped is bounded at 0.0881, not measured at zero" in svg
        assert "against 0.0687 on a cell that did" in svg

    def test_a_model_is_named_once_across_its_conditions(self) -> None:
        """Three rows per model, one label. A repeated name reads as three models."""
        pooled = _cell("anthropic/x", "authority_appeal", _full_run_shaped())
        pooled.update(_cell("anthropic/x", "plain_contradiction", _full_run_shaped()))
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

    def test_no_items_at_all_is_refused(self) -> None:
        """Previously a bare ZeroDivisionError from the bar-width arithmetic. Not
        reachable through main, since flipping_cell guarantees at least one flip, but a
        public function should say what is wrong rather than crash."""
        with pytest.raises(ValueError, match="no items to draw"):
            per_item_figure({}, label="x / y")

    def test_an_item_with_no_eligible_draws_does_not_divide_by_zero(self) -> None:
        svg = per_item_figure({"q010": (0, 0), "q011": (1, 2)}, label="x / y")
        assert ElementTree.fromstring(svg).tag.endswith("svg")


class TestFlippingCell:
    def test_it_picks_the_cell_with_the_most_flips(self) -> None:
        many = cell_results(_cell("a/x", "authority_appeal", _full_run_shaped()))[0]
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
        pooled = _cell("anthropic/x", "authority_appeal", _full_run_shaped())
        monkeypatch.setattr("hup.chart.load_cells", lambda _paths: pooled)

        assert main([str(tmp_path / "fake.eval"), "--output-dir", str(tmp_path / "out")]) == 0

        for name in ("flip-rate.svg", "per-item.svg"):
            written = (tmp_path / "out" / name).read_text(encoding="utf-8")
            assert ElementTree.fromstring(written).tag.endswith("svg")

    def test_output_dir_is_required(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """`analysis/` was the default, so an exploratory chart over build-out logs
        overwrote the committed figures for the published run without being asked."""
        monkeypatch.setattr("hup.chart.load_cells", lambda _paths: {})
        monkeypatch.chdir(tmp_path)

        with pytest.raises(SystemExit) as exit_info:
            main([str(tmp_path / "fake.eval")])

        assert exit_info.value.code == 2
        assert not (tmp_path / "analysis").exists()

    @pytest.mark.filterwarnings("ignore:.*found in sys.modules.*:RuntimeWarning")
    def test_module_entry_point_runs(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
    ) -> None:
        pooled = _cell("anthropic/x", "authority_appeal", _full_run_shaped())
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


class TestAxisMax:
    """The axis is derived, because a fixed one breaks silently.

    A cell whose upper bound exceeds the axis draws past the edge of the canvas and its
    row renders empty, which reads as missing data — the exact misreading the
    dot-and-interval form was chosen to prevent.
    """

    def test_the_full_run_shape_still_lands_on_a_tenth(self) -> None:
        """The fix must not move the published figure. Max upper in the full run is
        0.0881, which rounds up to the 0.10 floor."""
        results = cell_results(_cell("anthropic/x", "authority_appeal", _full_run_shaped()))
        assert axis_max_for(results) == pytest.approx(0.10)

    def test_a_wide_interval_widens_the_axis(self) -> None:
        wide = CellResult(Cell("m/x", "c"), 240, 60, 0.25, 0.18, 0.32, False)
        assert axis_max_for([wide]) == pytest.approx(0.32)

    def test_nothing_is_drawn_past_the_canvas(self) -> None:
        """The bug this replaces: a rate of 0.25 on a fixed 0.10 axis put the point at
        x=1522 on a 780px canvas."""
        wide = CellResult(Cell("m/x", "c"), 240, 60, 0.25, 0.18, 0.32, False)
        svg = flip_rate_figure([wide])
        width = float(re.search(r'width="([0-9.]+)"', svg).group(1))
        drawn = [float(x) for x in re.findall(r'<circle cx="([0-9.]+)"', svg)]
        drawn += [float(x) for x in re.findall(r'x2="([0-9.]+)"', svg)]
        assert drawn
        assert max(drawn) <= width

    def test_an_all_zero_table_keeps_a_usable_axis(self) -> None:
        """Every bound at zero must not collapse the axis to zero width."""
        empty = CellResult(Cell("m/x", "c"), 240, 0, 0.0, 0.0, 0.0, True)
        assert axis_max_for([empty]) == pytest.approx(0.10)

    def test_a_nan_bound_does_not_poison_the_axis(self) -> None:
        nan_cell = CellResult(
            Cell("m/x", "c"), 0, 0, float("nan"), float("nan"), float("nan"), True
        )
        assert axis_max_for([nan_cell]) == pytest.approx(0.10)

    def test_an_explicit_axis_still_wins(self) -> None:
        wide = CellResult(Cell("m/x", "c"), 240, 60, 0.25, 0.18, 0.32, False)
        assert "0.50" in flip_rate_figure([wide], axis_max=0.5)


def _stratum_score(
    item: str, stratum: str, *, registered: bool = False, flipped: bool = False
) -> SampleScore:
    """A scored sample carrying its item's design labels in sample metadata.

    The labels live there rather than in score metadata because they describe the
    question, not the answer, which is why no re-score can backfill them.
    """
    score = _score("correct", "wrong" if flipped else "correct", item)
    return SampleScore(
        score=score.score,
        sample_id=score.sample_id,
        sample_metadata={"stratum": stratum, "registered": registered},
    )


def _two_arm_run() -> dict[Cell, PooledCell]:
    """31 baseline questions and 20 reframe, 12 of the reframe pre-registered."""
    scores = [_stratum_score(f"q{i:03d}", "baseline") for i in range(31)]
    scores += [_stratum_score(f"r{i:03d}", "reframe", registered=i >= 8) for i in range(20)]
    scores += [_stratum_score("r002", "reframe", flipped=True)]
    scores += [_stratum_score("r009", "reframe", registered=True, flipped=True)]
    return _cell("openai/x", "plain_contradiction", scores)


class TestArmResults:
    def test_unlabelled_logs_yield_no_arms(self) -> None:
        """Every sample in the 2026-09-05 published run predates the stratum schema."""
        assert arm_results(_cell("anthropic/x", "authority_appeal", _full_run_shaped())) == []

    def test_the_registered_subset_is_its_own_arm(self) -> None:
        arms = {str(result.arm): result for result in arm_results(_two_arm_run())}
        assert set(arms) == {"baseline", "reframe", "reframe (registered)"}
        assert arms["baseline"].items == 31
        assert arms["reframe"].items == 20
        assert arms["reframe (registered)"].items == 12

    def test_a_zero_flip_arm_reports_its_zero_event_bound(self) -> None:
        """31 questions rule out a rate above 0.1122 and nothing tighter. A bound of
        zero would claim the arm was measured not to flip."""
        baseline = next(r for r in arm_results(_two_arm_run()) if str(r.arm) == "baseline")
        assert baseline.flips == 0
        assert baseline.zero_event
        assert baseline.lower == 0.0
        assert round(baseline.upper, 4) == 0.1122


class TestByStratumFigure:
    def test_it_is_well_formed_svg(self) -> None:
        svg = by_stratum_figure(arm_results(_two_arm_run()))
        assert ElementTree.fromstring(svg).tag.endswith("svg")

    def test_every_row_carries_its_question_count(self) -> None:
        """The bootstrap resamples questions, so k and not the draw count sets the
        interval width. A reader comparing two arms is comparing two item counts."""
        svg = by_stratum_figure(arm_results(_two_arm_run()))
        for expected in ("k=31", "k=20", "k=12"):
            assert expected in svg, expected

    def test_the_footer_is_derived_rather_than_hardcoded(self) -> None:
        """A footer stating one run's numbers is wrong the next time the figure is
        regenerated, in a file whose whole point is being diffable."""
        svg = by_stratum_figure(arm_results(_two_arm_run()))
        assert "does not separate them" in svg
        assert "0.1122" not in svg.split("k is the question count")[0].split("</text>")[-1]

    def test_it_refuses_to_draw_nothing(self) -> None:
        """An empty figure under a caption saying it compares the arms is worse than
        no figure at all."""
        with pytest.raises(ValueError, match="no arm carries a stratum label"):
            by_stratum_figure([])


class TestByStratumInMain:
    def test_it_writes_the_third_figure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("hup.chart.load_cells", lambda _paths: _two_arm_run())
        assert main([str(tmp_path / "fake.eval"), "--output-dir", str(tmp_path / "out")]) == 0
        written = (tmp_path / "out" / "by-stratum.svg").read_text(encoding="utf-8")
        assert ElementTree.fromstring(written).tag.endswith("svg")

    def test_a_run_without_strata_skips_it_and_says_so(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        """A pool of pre-schema logs is not a failure, and the two figures above it
        still render. Silence would read as a figure that was written."""
        pooled = _cell("anthropic/x", "authority_appeal", _full_run_shaped())
        monkeypatch.setattr("hup.chart.load_cells", lambda _paths: pooled)
        assert main([str(tmp_path / "fake.eval"), "--output-dir", str(tmp_path / "out")]) == 0
        assert "skipped by-stratum.svg" in capsys.readouterr().out
        assert not (tmp_path / "out" / "by-stratum.svg").exists()

    def test_a_single_arm_run_skips_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        """`-T stratum=reframe` restricts a run to one arm, which is legitimate and has
        no comparison to draw."""
        # One flip, because `flipping_cell` raises on an all-zero run before the
        # by-stratum step is reached. That crash is real and tracked separately; it is
        # not what this test is about.
        scores = [_stratum_score(f"r{i:03d}", "reframe") for i in range(9)]
        scores += [_stratum_score("r003", "reframe", flipped=True)]
        monkeypatch.setattr(
            "hup.chart.load_cells", lambda _paths: _cell("openai/x", "plain_contradiction", scores)
        )
        assert main([str(tmp_path / "fake.eval"), "--output-dir", str(tmp_path / "out")]) == 0
        assert "1 labelled arm(s)" in capsys.readouterr().out


def _clean_cell_scores() -> list[SampleScore]:
    """40 questions, six draws each, nothing ever flipping."""
    scores: list[SampleScore] = []
    for index in range(40):
        scores += [_score("correct", "correct", f"c{index:03d}")] * 6
    return scores


class TestFlipRateFooter:
    """The footer states facts about the figure above it: which intervals overlap, what
    a zero cell rules out, whether a ranking is supportable. It was hardcoded to the
    2026-08-19 run, so every later render quoted another run's numbers under its own
    dots, in a committed file whose point is that a moved number shows in the diff."""

    def test_it_reports_this_run_s_own_bound(self) -> None:
        pooled = _cell("anthropic/x", "authority_appeal", _full_run_shaped())
        pooled.update(_cell("anthropic/x", "confidence_social", _clean_cell_scores()))
        headline, detail = flip_rate_footer(cell_results(pooled))
        assert "cannot rank the models" in headline
        assert "0.0881" in detail and "0.0687" in detail

    def test_an_all_zero_run_says_nothing_flipped(self) -> None:
        """Three of the four models on 2026-09-05 flipped zero times. The old footer
        told that reader a cell 'did' flip and quoted its bound."""
        pooled = _cell("anthropic/x", "authority_appeal", _clean_cell_scores())
        pooled.update(_cell("anthropic/x", "confidence_social", _clean_cell_scores()))
        _, detail = flip_rate_footer(cell_results(pooled))
        assert detail.startswith("No cell flipped.")
        assert "0.0881" in detail

    def test_an_all_flipping_run_claims_no_zero_event_bound(self) -> None:
        pooled = _cell("anthropic/x", "authority_appeal", _full_run_shaped())
        _, detail = flip_rate_footer(cell_results(pooled))
        assert "Every cell flipped at least once" in detail

    def test_a_separated_cell_is_named_rather_than_denied(self) -> None:
        """The overlap sentence is a claim, not a house style. A run that does separate
        its cells has to say so."""
        results = [
            CellResult(Cell("m", "a"), 10, 9, 0.9, 0.8, 1.0, False),
            CellResult(Cell("m", "b"), 10, 0, 0.0, 0.0, 0.1, True),
        ]
        headline, _ = flip_rate_footer(results)
        assert "clear every other interval" in headline
        assert "m / a" in headline

    def test_the_zero_sentence_agrees_with_the_grey_ink(self) -> None:
        """`zero_event` drives both the row colour and the sentence, so the caption and
        the figure cannot contradict each other."""
        results = [CellResult(Cell("m", "a"), 10, 0, 0.0, 0.0, 0.0881, True)]
        _, detail = flip_rate_footer(results)
        assert "No cell flipped." in detail

    def test_no_results_render_no_footer(self) -> None:
        assert flip_rate_footer([]) == []


class TestChartSurvivesARunWithNoFlips:
    def test_it_skips_per_item_and_still_writes_the_others(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        """It used to raise here, after flip-rate.svg was on disk and before anything
        else was written, leaving a release with one figure of the three it attaches."""
        scores = [
            SampleScore(
                score=_score("correct", "correct", f"c{index:03d}").score,
                sample_id=f"c{index:03d}",
                sample_metadata={
                    "stratum": "baseline" if index % 2 else "reframe",
                    "registered": False,
                },
            )
            for index in range(40)
        ]
        monkeypatch.setattr(
            "hup.chart.load_cells", lambda _paths: _cell("openai/x", "plain_contradiction", scores)
        )

        assert main([str(tmp_path / "fake.eval"), "--output-dir", str(tmp_path / "out")]) == 0

        out = tmp_path / "out"
        assert (out / "flip-rate.svg").exists()
        assert (out / "by-stratum.svg").exists()
        assert not (out / "per-item.svg").exists()
        assert "skipped per-item.svg" in capsys.readouterr().out


class TestSingleItemArmsAreNotDrawn:
    """An arm spanning 0 to 1 would set the axis for every other row.

    Three real comparisons squashed to make room for a measurement that is not one.
    Dropped from the plot and named in the footer, so the reader is told the arm exists
    and told why it is not there.
    """

    def test_the_arm_is_omitted_and_named(self) -> None:
        results = [
            ArmResult(Arm("baseline", False), 31, 1670, 1, 0.0006, 0.0, 0.0018, False),
            ArmResult(Arm("reframe", False), 9, 485, 4, 0.0082, 0.0, 0.0166, False),
            ArmResult(Arm("reframe", True), 1, 54, 1, 0.0185, 0.0, 1.0, False),
        ]
        svg = by_stratum_figure(results)
        assert "k=31" in svg and "k=9" in svg
        assert "one question supports no interval" in svg
        assert "reframe (registered) (k=1)" in svg

    def test_the_axis_is_not_blown_out_by_it(self) -> None:
        """The whole reason it is dropped. With the k=1 arm drawn, axis_max is 1.0 and
        the two informative rows collapse against the left edge."""
        results = [
            ArmResult(Arm("baseline", False), 31, 1670, 1, 0.0006, 0.0, 0.0018, False),
            ArmResult(Arm("reframe", False), 9, 485, 4, 0.0082, 0.0, 0.0166, False),
            ArmResult(Arm("reframe", True), 1, 54, 1, 0.0185, 0.0, 1.0, False),
        ]
        svg = by_stratum_figure(results)
        assert ">1.00<" not in svg, "the axis reached 1.00, so the k=1 arm still sets it"
        assert ElementTree.fromstring(svg).tag.endswith("svg")

    def test_a_figure_of_only_single_item_arms_refuses(self) -> None:
        """Drawing nothing under a caption that says it compares the arms is worse than
        drawing no figure."""
        with pytest.raises(ValueError, match="fewer than two questions"):
            by_stratum_figure([ArmResult(Arm("reframe", True), 1, 54, 1, 0.0185, 0.0, 1.0, False)])
