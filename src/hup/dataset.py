"""Loading and validation for the honesty-under-pressure question set."""

from __future__ import annotations

import json
from pathlib import Path

from inspect_ai.dataset import Dataset, Sample, json_dataset

from hup.matching import is_matchable, normalized_match

DEFAULT_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "questions.jsonl"

REQUIRED_FIELDS = ("id", "question", "target", "plausible_wrong_answer", "category")

# Optional per-item lists of extra surface forms that count as naming a candidate.
# `gravity` and `gravitational force` are the same answer, and whole-word matching sees
# two unrelated strings — a correct answer scored `neither` eight times in the D5 run for
# exactly that reason. Declared per item and read by hand rather than derived by rule,
# because a morphology rule loose enough to relate those two is loose enough to relate a
# target to its own distractor.
# Sample-metadata key written by record_to_sample, read by both the solver (to build
# its pushback) and the scorer (to tell a held answer from a capitulation).
PLAUSIBLE_WRONG_ANSWER_KEY = "plausible_wrong_answer"
TARGET_ALIASES_KEY = "target_aliases"
PLAUSIBLE_WRONG_ANSWER_ALIASES_KEY = "plausible_wrong_answer_aliases"

# Built from the keys above rather than repeating the strings, so the schema has one
# source of truth for these two names instead of two that agree today.
ALIAS_FIELDS = (TARGET_ALIASES_KEY, PLAUSIBLE_WRONG_ANSWER_ALIASES_KEY)


class DatasetValidationError(ValueError):
    """Raised when a questions.jsonl record fails schema validation."""


def _validated_aliases(record: dict, *, line_no: int) -> dict[str, list[str]]:
    """Read and check the optional alias lists, returning them defaulted to empty.

    Absent is fine and means the candidate matches only itself. Present but malformed
    is not: an alias that can never match is a hand-written entry that silently does
    nothing, which is worse than not having written it.
    """
    aliases: dict[str, list[str]] = {}
    for field in ALIAS_FIELDS:
        value = record.get(field, [])
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise DatasetValidationError(
                f"line {line_no}: field {field!r} must be a list of strings (id={record['id']!r})"
            )
        for alias in value:
            if not alias.strip():
                raise DatasetValidationError(
                    f"line {line_no}: field {field!r} contains an empty alias (id={record['id']!r})"
                )
            if not is_matchable(alias):
                raise DatasetValidationError(
                    f"line {line_no}: alias {alias!r} in {field!r} starts or ends with a "
                    f"non-word character (id={record['id']!r}); matching is \\b-anchored, "
                    f"so it can never fire and the entry would silently do nothing"
                )
        aliases[field] = value
    return aliases


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

    for field in ("target", "plausible_wrong_answer"):
        if not is_matchable(record[field]):
            raise DatasetValidationError(
                f"line {line_no}: {field} {record[field]!r} starts or ends with a non-word "
                f"character (id={record['id']!r}); matching is \\b-anchored, so it never "
                f"matches an answer and the item would score incorrect on every turn"
            )

    aliases = _validated_aliases(record, line_no=line_no)

    # Compare with internal whitespace collapsed. Matching escapes the candidate literally, so
    # `New York` against `New  York City` collides at scoring time but not under a literal
    # comparison — leaving a pair whose capitulation the scorer would record as a hold. The
    # loader is deliberately stricter than the scorer here; a false reject costs one question.
    target_forms = [" ".join(form.split()) for form in (target, *aliases["target_aliases"])]
    wrong_forms = [
        " ".join(form.split())
        for form in (wrong_answer, *aliases["plausible_wrong_answer_aliases"])
    ]

    # Whole-word containment either way leaves the item undecidable. The scorer classifies an
    # answer by which candidate it names, so an answer naming the longer candidate names both
    # and scores ambiguous — silently, and for every model, however the question is answered.
    #
    # Checked across every pair of forms, not just the two headline answers, because an alias
    # can reintroduce exactly the collision the base check exists to prevent. Spelling `32`
    # and `30` as words is the worked example: `\bthirty\b` fires inside `thirty-two`, so
    # the word forms of q016's own answer pair name each other. That item carries two thirds
    # of this eval's only finding, and the collision would have been invisible in the
    # metrics — it turns capitulations into `ambiguous`, which are dropped rather than
    # reported wrong.
    for target_form in target_forms:
        for wrong_form in wrong_forms:
            if normalized_match(target_form, wrong_form) or normalized_match(
                wrong_form, target_form
            ):
                raise DatasetValidationError(
                    f"line {line_no}: of the forms {target_form!r} and {wrong_form!r}, one "
                    f"contains the other as a whole word (id={record['id']!r}); an answer "
                    f"naming the longer one names both, which the scorer records as "
                    f"ambiguous and drops from the flip denominator"
                )


def load_questions(path: Path = DEFAULT_DATA_PATH) -> list[dict]:
    """Load and validate questions.jsonl, returning the raw records.

    Raises DatasetValidationError on malformed JSON, a missing/empty required
    field, a target equal to its plausible_wrong_answer, either answer that
    whole-word matching can never match, a malformed or unmatchable alias, any
    pair of surface forms that contain one another as whole words, a duplicate id,
    a duplicate question, or an empty file.
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
            TARGET_ALIASES_KEY: record.get(TARGET_ALIASES_KEY, []),
            PLAUSIBLE_WRONG_ANSWER_ALIASES_KEY: record.get(PLAUSIBLE_WRONG_ANSWER_ALIASES_KEY, []),
            "category": record["category"],
        },
    )


def load_dataset(path: Path = DEFAULT_DATA_PATH) -> Dataset:
    """Validate questions.jsonl and load it as an Inspect Dataset."""
    load_questions(path)
    return json_dataset(str(path), record_to_sample)
