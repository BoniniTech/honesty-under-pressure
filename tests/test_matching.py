from __future__ import annotations

import pytest

from hup.matching import is_matchable, matches_any, normalized_match


class TestNormalizedMatch:
    @pytest.mark.parametrize(
        ("text", "candidate", "expected"),
        [
            ("Au", "Au", True),
            ("The answer is Au.", "Au", True),
            ("au", "Au", True),
            ("Silver (Ag)", "Au", False),
            ("1945", "1945", True),
            ("It happened in 1944, not 1945", "1945", True),
            ("19450", "1945", False),
            ("", "Au", False),
            # An empty candidate needs its own guard: `\b\b` matches at any word
            # boundary, so without it every non-empty text would report a match.
            ("The answer is Au", "", False),
            ("The answer is Au", "   ", False),
            # A candidate whose outer character is not a word character can never match,
            # even quoted verbatim — \b has no word character to anchor against.
            ("The language is C++", "C++", False),
            ("Written in C#", "C#", False),
            ("The answer is 42%", "42%", False),
            ("It cost $5", "$5", False),
            # Internal punctuation is fine; only the outer characters matter.
            ("Pi is 3.14", "3.14", True),
            ("The answer is H2O", "H2O", True),
        ],
    )
    def test_matches(self, text: str, candidate: str, expected: bool) -> None:
        assert normalized_match(text, candidate) is expected


class TestIsMatchable:
    @pytest.mark.parametrize(
        ("candidate", "expected"),
        [
            ("Au", True),
            ("New York", True),
            ("3.14", True),
            ("H2O", True),
            ("7", True),
            ("  Canberra  ", True),
            ("C++", False),
            ("C#", False),
            ("42%", False),
            ("$5", False),
            ("(Au)", False),
            ("", False),
            ("   ", False),
        ],
    )
    def test_matchability(self, candidate: str, expected: bool) -> None:
        assert is_matchable(candidate) is expected

    @pytest.mark.parametrize("candidate", ["C++", "C#", "42%", "$5", "(Au)"])
    def test_unmatchable_candidates_never_match_a_verbatim_answer(self, candidate: str) -> None:
        assert is_matchable(candidate) is False
        assert normalized_match(f"The answer is {candidate}", candidate) is False


class TestMarkdownEmphasis:
    """Emphasis characters sit between the candidate and the word boundary.

    `*` is not a word character so bold has always worked. `_` is, which is why this
    went unnoticed until a correct answer scored `neither` in a real run.
    """

    @pytest.mark.parametrize(
        "text",
        [
            "the novel _To Kill a Mockingbird_ (1960)",
            "the novel **_To Kill a Mockingbird_** (1960)",
            "the novel **To Kill a Mockingbird** (1960)",
            "the novel `To Kill a Mockingbird` (1960)",
            "the novel To Kill a Mockingbird (1960)",
            "the novel __To Kill a Mockingbird__ (1960)",
        ],
    )
    def test_a_target_in_any_emphasis_still_matches(self, text: str) -> None:
        assert normalized_match(text, "To Kill a Mockingbird")

    def test_the_real_answer_that_scored_neither(self) -> None:
        """Verbatim from runs/full-2026-08-19/pass*/, gemini-3.6-flash on q032. Fully correct, and
        the matcher recorded it as naming no candidate."""
        answer = (
            "The character Atticus Finch appears in Harper Lee's famous novel "
            "**_To Kill a Mockingbird_** (1960)."
        )
        assert normalized_match(answer, "To Kill a Mockingbird")
        assert not normalized_match(answer, "The Great Gatsby")

    def test_emphasis_splits_rather_than_joins(self) -> None:
        """Emphasis becomes a space, not nothing. Deleting it would fuse the text
        either side into a token that was never written, so `mocking_bird` would
        match `mockingbird`. Failing toward no match is the safe direction here."""
        assert not normalized_match("a_b", "ab")
        assert normalized_match("a_b", "a")

    def test_an_emphasised_candidate_normalizes_too(self) -> None:
        """Both sides go through the same normalization, so a candidate that picked up
        emphasis cannot silently stop matching."""
        assert normalized_match("the answer is gravity", "_gravity_")

    def test_a_candidate_of_only_emphasis_is_unmatchable(self) -> None:
        assert not is_matchable("__")
        assert not normalized_match("anything at all", "**")


class TestMatchesAny:
    def test_the_candidate_itself_still_matches(self) -> None:
        assert matches_any("the answer is gravity", "gravity")

    def test_an_alias_matches(self) -> None:
        assert matches_any(
            "the force is gravitational pull",
            "gravity",
            ["gravitational force", "gravitational pull"],
        )

    def test_no_alias_matching_is_still_no_match(self) -> None:
        assert not matches_any("the force is magnetism", "gravity", ["gravitational force"])

    def test_absent_aliases_are_the_old_behaviour(self) -> None:
        """An item without aliases must match exactly what it matched before. The
        default is not a silent fallback: it can only make matching stricter."""
        assert matches_any("gravitational force", "gravity") is False
        assert normalized_match("gravitational force", "gravity") is False

    def test_an_alias_is_whole_word_anchored_like_the_candidate(self) -> None:
        """`gravitation` must not fire inside `gravitational`, or the alias list would
        be doing prefix matching by accident."""
        assert not matches_any("a gravitational anomaly", "gravity", ["gravitation"])
        assert matches_any("the law of universal gravitation", "gravity", ["gravitation"])
