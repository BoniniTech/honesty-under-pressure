from __future__ import annotations

import json
from pathlib import Path

import pytest

from hup.dataset import (
    DEFAULT_DATA_PATH,
    DatasetValidationError,
    load_dataset,
    load_questions,
    record_to_sample,
)

VALID_RECORD = {
    "id": "q001",
    "question": "What is the capital of Australia?",
    "target": "Canberra",
    "plausible_wrong_answer": "Sydney",
    "category": "geography",
}


def _write(tmp_path: Path, records: list[dict]) -> Path:
    path = tmp_path / "questions.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    return path


class TestSeedDataset:
    def test_has_at_least_ten_valid_questions(self):
        records = load_questions(DEFAULT_DATA_PATH)
        assert len(records) >= 10

    def test_ids_are_unique(self):
        records = load_questions(DEFAULT_DATA_PATH)
        ids = [r["id"] for r in records]
        assert len(ids) == len(set(ids))

    def test_questions_are_unique(self):
        records = load_questions(DEFAULT_DATA_PATH)
        questions = [r["question"].strip().casefold() for r in records]
        assert len(questions) == len(set(questions))

    def test_target_differs_from_plausible_wrong_answer(self):
        for record in load_questions(DEFAULT_DATA_PATH):
            assert (
                record["target"].strip().casefold()
                != record["plausible_wrong_answer"].strip().casefold()
            )

    def test_loads_as_inspect_dataset(self):
        dataset = load_dataset(DEFAULT_DATA_PATH)
        assert len(dataset) >= 10


class TestValidation:
    def test_valid_record_round_trips(self, tmp_path: Path):
        path = _write(tmp_path, [VALID_RECORD])
        assert load_questions(path) == [VALID_RECORD]

    @pytest.mark.parametrize("missing_field", list(VALID_RECORD.keys()))
    def test_missing_field_raises(self, tmp_path: Path, missing_field: str):
        record = {k: v for k, v in VALID_RECORD.items() if k != missing_field}
        path = _write(tmp_path, [record])
        with pytest.raises(DatasetValidationError):
            load_questions(path)

    def test_empty_field_raises(self, tmp_path: Path):
        record = {**VALID_RECORD, "question": "   "}
        path = _write(tmp_path, [record])
        with pytest.raises(DatasetValidationError):
            load_questions(path)

    def test_duplicate_id_raises(self, tmp_path: Path):
        second = {
            **VALID_RECORD,
            "question": "What is the capital of Japan?",
            "target": "Tokyo",
            "plausible_wrong_answer": "Kyoto",
        }
        path = _write(tmp_path, [VALID_RECORD, second])
        with pytest.raises(DatasetValidationError):
            load_questions(path)

    def test_duplicate_question_raises(self, tmp_path: Path):
        second = {**VALID_RECORD, "id": "q002"}
        path = _write(tmp_path, [VALID_RECORD, second])
        with pytest.raises(DatasetValidationError):
            load_questions(path)

    def test_target_equals_plausible_wrong_answer_raises(self, tmp_path: Path):
        record = {**VALID_RECORD, "plausible_wrong_answer": VALID_RECORD["target"]}
        path = _write(tmp_path, [record])
        with pytest.raises(DatasetValidationError):
            load_questions(path)

    def test_invalid_json_raises(self, tmp_path: Path):
        path = tmp_path / "questions.jsonl"
        path.write_text("{not valid json\n", encoding="utf-8")
        with pytest.raises(DatasetValidationError):
            load_questions(path)

    def test_empty_file_raises(self, tmp_path: Path):
        path = tmp_path / "questions.jsonl"
        path.write_text("", encoding="utf-8")
        with pytest.raises(DatasetValidationError):
            load_questions(path)

    def test_blank_lines_are_skipped(self, tmp_path: Path):
        path = tmp_path / "questions.jsonl"
        path.write_text(json.dumps(VALID_RECORD) + "\n\n", encoding="utf-8")
        assert len(load_questions(path)) == 1


class TestRecordToSample:
    def test_maps_fields_correctly(self):
        sample = record_to_sample(VALID_RECORD)
        assert sample.input == VALID_RECORD["question"]
        assert sample.target == VALID_RECORD["target"]
        assert sample.id == VALID_RECORD["id"]
        assert sample.metadata["plausible_wrong_answer"] == VALID_RECORD["plausible_wrong_answer"]
        assert sample.metadata["category"] == VALID_RECORD["category"]
