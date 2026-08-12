from __future__ import annotations

import pytest

from hup.matching import is_matchable, normalized_match


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
