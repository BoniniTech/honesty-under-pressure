"""Whole-word answer matching, shared by the dataset validator and the scorer.

Its own module because both `dataset` and `scorers` need it and `scorers` already
imports from `dataset`. The two must use one implementation, not two that agree
today: the loader's containment check exists to reject items the scorer would find
undecidable, so a divergence would let exactly those items through.
"""

from __future__ import annotations

import re


def normalized_match(text: str, candidate: str) -> bool:
    """Whether candidate appears in text as a whole word/phrase, case-insensitively."""
    text_norm = text.strip().casefold()
    candidate_norm = candidate.strip().casefold()
    if not candidate_norm:
        return False
    return re.search(rf"\b{re.escape(candidate_norm)}\b", text_norm) is not None


def is_matchable(candidate: str) -> bool:
    r"""Whether `candidate` can ever match inside a model's answer.

    `normalized_match` anchors with `\b`, which needs a word character on its inner
    side, so a candidate whose first or last character is not a word character never
    matches — not even an answer quoting it verbatim. `C++` and `$5` are unmatchable;
    `3.14` is fine, because only the outer characters matter.
    """
    normalized = candidate.strip()
    if not normalized:
        return False
    return re.match(r"\w", normalized) is not None and re.search(r"\w\Z", normalized) is not None
