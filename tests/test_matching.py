from __future__ import annotations

import pytest

from hup.matching import normalized_match


class TestNormalizedMatch:
    @pytest.mark.parametrize(
        ("answer", "candidate", "expected"),
        [
            ("Au", "Au", True),
            ("The answer is Au.", "Au", True),
            ("au", "Au", True),
            ("Silver (Ag)", "Au", False),
            ("1945", "1945", True),
            ("It happened in 1944, not 1945", "1945", True),
            ("19450", "1945", False),
            ("", "Au", False),
        ],
    )
    def test_matches(self, answer: str, candidate: str, expected: bool) -> None:
        assert normalized_match(answer, candidate) is expected
