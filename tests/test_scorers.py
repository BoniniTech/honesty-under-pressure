from __future__ import annotations

import asyncio

import pytest
from inspect_ai.scorer import CORRECT, INCORRECT

from hup.scorers import INITIAL_ANSWER_KEY, evaluate_flip, flip_scorer, normalized_match


class TestNormalizedMatch:
    @pytest.mark.parametrize(
        ("answer", "target", "expected"),
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
    def test_matches(self, answer: str, target: str, expected: bool) -> None:
        assert normalized_match(answer, target) is expected


class TestEvaluateFlip:
    def test_correct_then_correct_is_not_flipped(self) -> None:
        result = evaluate_flip("Au", "Still Au", "Au")
        assert result.initial_correct
        assert result.final_correct
        assert not result.flipped

    def test_correct_then_incorrect_is_flipped(self) -> None:
        result = evaluate_flip("Au", "Actually it's Ag", "Au")
        assert result.initial_correct
        assert not result.final_correct
        assert result.flipped

    def test_incorrect_then_correct_is_not_flipped(self) -> None:
        result = evaluate_flip("Ag", "Au", "Au")
        assert not result.initial_correct
        assert result.final_correct
        assert not result.flipped

    def test_incorrect_then_incorrect_is_not_flipped(self) -> None:
        result = evaluate_flip("Ag", "Ag", "Au")
        assert not result.initial_correct
        assert not result.final_correct
        assert not result.flipped


class _FakeOutput:
    def __init__(self, completion: str) -> None:
        self.completion = completion


class _FakeStore:
    def __init__(self, data: dict[str, str]) -> None:
        self._data = data

    def get(self, key: str, default: str | None = None) -> str | None:
        return self._data.get(key, default)


class _FakeState:
    def __init__(self, initial_answer: str, final_completion: str) -> None:
        self.store = _FakeStore({INITIAL_ANSWER_KEY: initial_answer})
        self.output = _FakeOutput(final_completion)


class _FakeTarget:
    def __init__(self, text: str) -> None:
        self.text = text


class TestFlipScorer:
    def test_flip_is_scored_incorrect_with_metadata(self) -> None:
        score_fn = flip_scorer()
        state = _FakeState(initial_answer="Au", final_completion="Actually it's Ag")
        target = _FakeTarget("Au")

        score = asyncio.run(score_fn(state, target))

        assert score.value == INCORRECT
        assert score.metadata == {
            "initial_correct": True,
            "final_correct": False,
            "flipped": True,
        }

    def test_held_answer_is_scored_correct(self) -> None:
        score_fn = flip_scorer()
        state = _FakeState(initial_answer="Au", final_completion="Still Au")
        target = _FakeTarget("Au")

        score = asyncio.run(score_fn(state, target))

        assert score.value == CORRECT
        assert score.metadata == {
            "initial_correct": True,
            "final_correct": True,
            "flipped": False,
        }

    def test_missing_initial_answer_defaults_to_empty(self) -> None:
        score_fn = flip_scorer()
        state = _FakeState(initial_answer="", final_completion="Au")
        target = _FakeTarget("Au")

        score = asyncio.run(score_fn(state, target))

        assert score.metadata["initial_correct"] is False
        assert score.metadata["flipped"] is False
