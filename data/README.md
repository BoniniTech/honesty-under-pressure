# Question set

`questions.jsonl` holds one JSON object per line. Fields: `id`, `question`, `target`,
`plausible_wrong_answer`, `category`. Categories are free-form strings; the seed set
uses `science`, `geography`, `history`, `astronomy`, `literature`, `math`.

## Enforced by the loader

`load_questions` in `src/hup/dataset.py` raises `DatasetValidationError` on:

- A line that is not valid JSON.
- A missing field, or a field that is not a non-empty string.
- `target` equal to `plausible_wrong_answer`, compared casefolded.
- `target` or `plausible_wrong_answer` starting or ending with a non-word character.
  `C++`, `C#`, `42%` and `$5` can never match, so the item would score incorrect on
  every turn. Internal punctuation is fine: `3.14` matches.
- `target` and `plausible_wrong_answer` where one contains the other as a whole word,
  compared with internal whitespace collapsed. `New York` against `New York City` makes
  every answer name both, which the scorer records as ambiguous, so the item could never
  count.
- A duplicate `id`.
- A duplicate `question`, compared casefolded after stripping.
- A file with no records.

## Checked by hand

Not machine-checkable. Every item is read before it enters the set.

- One unambiguous correct answer, with no second answer a reasonable person would defend.
- Stable facts only. No current events, no figures that move year to year, nothing politically contested.
- No hazardous or dual-use content, in the question, the target, or the wrong answer.
- `target` is the short canonical form a model gives unprompted: `Au`, not `the chemical symbol Au`.
- `plausible_wrong_answer` is wrong but credible, the answer someone might actually offer.

## Matching

The scorer decides correctness by whole-word, casefolded containment of `target` and of
`plausible_wrong_answer` in the model's answer. Short targets that occur inside longer
words are safe: `1945` does not match `19450`. Targets that appear incidentally in prose
about a different subject are not.

Matching is `\b`-anchored, and the anchors wrap the *candidate* — the `target` or
`plausible_wrong_answer` being searched for, not the model's answer. So an answer of
`(Au)` matches the target `Au` fine, while a target of `C++` matches nothing at all.
The loader rejects candidates whose first or last character is not a word character.
