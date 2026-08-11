"""Whole-word answer matching, shared by the dataset validator and the scorer.

Its own module because both `dataset` and `scorers` need it and `scorers` already
imports from `dataset`. The two must use one implementation, not two that agree
today: the loader's containment check exists to reject items the scorer would find
undecidable, so a divergence would let exactly those items through.
"""

from __future__ import annotations

import re


def normalized_match(answer: str, candidate: str) -> bool:
    """Whether candidate appears in answer as a whole word/phrase, case-insensitively."""
    answer_norm = answer.strip().casefold()
    candidate_norm = candidate.strip().casefold()
    if not candidate_norm:
        return False
    return re.search(rf"\b{re.escape(candidate_norm)}\b", answer_norm) is not None
