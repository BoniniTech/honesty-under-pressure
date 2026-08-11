"""Loading and validation for the honesty-under-pressure question set."""

from __future__ import annotations

import json
from pathlib import Path

from inspect_ai.dataset import Dataset, Sample, json_dataset

from hup.matching import normalized_match

DEFAULT_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "questions.jsonl"

REQUIRED_FIELDS = ("id", "question", "target", "plausible_wrong_answer", "category")

# Sample-metadata key written by record_to_sample, read by both the solver (to build
# its pushback) and the scorer (to tell a held answer from a capitulation).
PLAUSIBLE_WRONG_ANSWER_KEY = "plausible_wrong_answer"


class DatasetValidationError(ValueError):
    """Raised when a questions.jsonl record fails schema validation."""


def _validate_record(record: dict, *, line_no: int) -> None:
    missing = [field for field in REQUIRED_FIELDS if field not in record]
    if missing:
        raise DatasetValidationError(f"line {line_no}: missing field(s) {missing}")

    for field in REQUIRED_FIELDS:
        value = record[field]
        if not isinstance(value, str) or not value.strip():
            raise DatasetValidationError(
                f"line {line_no}: field '{field}' must be a non-empty string"
            )

    target = record["target"]
    wrong_answer = record["plausible_wrong_answer"]

    if target.strip().casefold() == wrong_answer.strip().casefold():
        raise DatasetValidationError(
            f"line {line_no}: target and plausible_wrong_answer must differ (id={record['id']!r})"
        )

    # Whole-word containment either way leaves the item undecidable. The scorer classifies an
    # answer by which candidate it names, so an answer naming the longer candidate names both
    # and scores ambiguous — silently, and for every model, however the question is answered.
    if normalized_match(target, wrong_answer) or normalized_match(wrong_answer, target):
        raise DatasetValidationError(
            f"line {line_no}: one of target / plausible_wrong_answer contains the other "
            f"as a whole word (id={record['id']!r}); an answer naming the longer one names "
            f"both, which the scorer records as ambiguous and drops from the flip denominator"
        )


def load_questions(path: Path = DEFAULT_DATA_PATH) -> list[dict]:
    """Load and validate questions.jsonl, returning the raw records.

    Raises DatasetValidationError on malformed JSON, a missing/empty required
    field, a duplicate id, a duplicate question, or an empty file.
    """
    records: list[dict] = []
    seen_ids: set[str] = set()
    seen_questions: set[str] = set()

    with path.open(encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                raise DatasetValidationError(f"line {line_no}: invalid JSON ({e})") from e

            _validate_record(record, line_no=line_no)

            if record["id"] in seen_ids:
                raise DatasetValidationError(f"line {line_no}: duplicate id {record['id']!r}")
            seen_ids.add(record["id"])

            normalized_question = record["question"].strip().casefold()
            if normalized_question in seen_questions:
                raise DatasetValidationError(
                    f"line {line_no}: duplicate question (id={record['id']!r})"
                )
            seen_questions.add(normalized_question)

            records.append(record)

    if not records:
        raise DatasetValidationError(f"{path}: no records found")

    return records


def record_to_sample(record: dict) -> Sample:
    return Sample(
        input=record["question"],
        target=record["target"],
        id=record["id"],
        metadata={
            PLAUSIBLE_WRONG_ANSWER_KEY: record["plausible_wrong_answer"],
            "category": record["category"],
        },
    )


def load_dataset(path: Path = DEFAULT_DATA_PATH) -> Dataset:
    """Validate questions.jsonl and load it as an Inspect Dataset."""
    load_questions(path)
    return json_dataset(str(path), record_to_sample)
