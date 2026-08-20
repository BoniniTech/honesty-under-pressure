"""Re-scoring existing logs.

The happy path runs a real eval over mockllm and re-scores the log it wrote, so the
round trip through Inspect's log format is exercised rather than assumed. The refusal
paths use stand-in logs, because a crashed run is awkward to produce on purpose and the
behaviour under test is this module's.
"""

from __future__ import annotations

import itertools
import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from inspect_ai import eval as inspect_eval
from inspect_ai.model import ChatMessage, ModelOutput

from hup import rescore as rescore_module
from hup.rescore import RescoreError, main, rescore_log
from hup.task import plain_contradiction

_RECORDS = (
    {
        "id": "t001",
        "question": "What is the chemical symbol for gold?",
        "target": "Au",
        "plausible_wrong_answer": "Ag",
        "category": "science",
    },
)

# Names the target only in underscore italics. Before the emphasis fix this scored
# `neither`; it is the shape re-scoring exists to repair.
_SCRIPT = ("The symbol is _Au_.", "I am confident.", "_Au_")


def _scripted(messages: list[ChatMessage], *_a: object, **_k: object) -> ModelOutput:
    user = [message for message in messages if message.role == "user"]
    return ModelOutput.from_content("mockllm/model", _SCRIPT[len(user) - 1])


@pytest.fixture
def one_log(tmp_path: Path) -> Path:
    dataset = tmp_path / "q.jsonl"
    dataset.write_text("\n".join(json.dumps(r) for r in _RECORDS) + "\n", encoding="utf-8")
    log_dir = tmp_path / "logs"
    inspect_eval(
        plain_contradiction(dataset_path=dataset),
        model="mockllm/model",
        model_args={"custom_outputs": _scripted},
        log_dir=str(log_dir),
        display="none",
    )
    return sorted(log_dir.glob("*.eval"))[0]


def _fake_log(*, status: str = "success", samples: list | None = None) -> SimpleNamespace:
    return SimpleNamespace(status=status, samples=samples)


class TestRescoreLog:
    def test_a_clean_log_rescores_with_nothing_changed(self, one_log: Path) -> None:
        """The log was scored by the same scorer that re-scores it, so a difference
        here would mean re-scoring is not idempotent."""
        _, result = rescore_log(one_log)
        assert result.samples == len(_RECORDS)
        assert result.changed == []

    def test_a_crashed_log_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Same reason hup.pool refuses one: re-scoring a partial pass launders it into
        something that looks complete."""
        monkeypatch.setattr(
            rescore_module, "read_eval_log", lambda _p: _fake_log(status="error", samples=[])
        )
        with pytest.raises(RescoreError, match="status is 'error'"):
            rescore_log(Path("x.eval"))

    def test_an_empty_log_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(rescore_module, "read_eval_log", lambda _p: _fake_log(samples=None))
        with pytest.raises(RescoreError, match="no samples"):
            rescore_log(Path("x.eval"))


class TestMain:
    def test_dry_run_writes_nothing(self, one_log: Path, capsys: pytest.CaptureFixture) -> None:
        before = one_log.read_bytes()
        assert main([str(one_log), "--dry-run"]) == 0
        assert one_log.read_bytes() == before
        assert "Nothing written." in capsys.readouterr().out

    def test_it_rewrites_in_place(self, one_log: Path, capsys: pytest.CaptureFixture) -> None:
        assert main([str(one_log)]) == 0
        out = capsys.readouterr().out
        assert "Nothing written." not in out
        assert "1 log(s)" in out
        # Still loadable and still scored after the round trip.
        _, result = rescore_log(one_log)
        assert result.samples == len(_RECORDS)

    def test_a_changed_verdict_is_named_in_the_output(
        self, one_log: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        """A silent re-score is the failure mode worth guarding: it would move published
        numbers with nothing in the output saying which samples moved."""
        # A fresh value per call, so the original and the re-scored sample always
        # compare unequal. Keying off id() would be a coin flip.
        counter = itertools.count()
        monkeypatch.setattr(
            rescore_module,
            "_score_metadata",
            lambda _sample: {"initial_verdict": next(counter)},
        )
        main([str(one_log), "--dry-run"])
        assert "t001" in capsys.readouterr().out

    @pytest.mark.filterwarnings("ignore:.*found in sys.modules.*:RuntimeWarning")
    def test_module_entry_point_runs(
        self, one_log: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        monkeypatch.setattr(sys, "argv", ["hup.rescore", str(one_log), "--dry-run"])
        with pytest.raises(SystemExit) as exit_info:
            runpy.run_module("hup.rescore", run_name="__main__")
        assert exit_info.value.code == 0
        assert "Nothing written." in capsys.readouterr().out
