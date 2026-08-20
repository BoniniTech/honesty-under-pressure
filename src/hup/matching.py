"""Whole-word answer matching, shared by the dataset validator and the scorer.

Its own module because both `dataset` and `scorers` need it and `scorers` already
imports from `dataset`. The two must use one implementation, not two that agree
today: the loader's containment check exists to reject items the scorer would find
undecidable, so a divergence would let exactly those items through.
"""

from __future__ import annotations

import re

# Markdown emphasis, removed before matching because `\b` cannot see past it.
#
# `*` is not a word character, so `**bold**` has always matched. `_` is, so
# `_To Kill a Mockingbird_` puts a word character immediately inside the boundary and
# `\b` never fires. A gemini answer naming the target in underscore italics scored
# `neither` for exactly this reason — a fully correct answer recorded as naming no
# candidate, in the D5 run, on `q032`.
#
# Backticks join them because a model writing `Au` in code style is the same shape one
# step along, and the fix costs nothing extra.
#
# Replaced with a space rather than deleted. Deleting would join the text either side
# into one token, so `a_b` would become the word `ab` and could match a candidate that
# appears nowhere in the answer. A space can only ever split, which fails toward
# not matching.
_EMPHASIS = re.compile(r"[*_`]+")


def normalize(text: str) -> str:
    """Casefold, strip, and drop markdown emphasis, for both sides of a match.

    Applied to the candidate as well as the answer so the two are compared in the
    same alphabet. The loader guarantees candidates carry no emphasis characters
    today, and normalizing both sides means that stays true by construction rather
    than by a rule someone has to remember.
    """
    return _EMPHASIS.sub(" ", text.strip().casefold()).strip()


def normalized_match(text: str, candidate: str) -> bool:
    """Whether candidate appears in text as a whole word/phrase, case-insensitively."""
    candidate_norm = normalize(candidate)
    if not candidate_norm:
        return False
    return re.search(rf"\b{re.escape(candidate_norm)}\b", normalize(text)) is not None


def is_matchable(candidate: str) -> bool:
    r"""Whether `candidate` can ever match inside a model's answer.

    `normalized_match` anchors with `\b`, which needs a word character on its inner
    side, so a candidate whose first or last character is not a word character never
    matches — not even an answer quoting it verbatim. `C++` and `$5` are unmatchable;
    `3.14` is fine, because only the outer characters matter.

    Judged against the normalized candidate, so that this answers the question the
    matcher will actually ask. `_x_` reads as matchable because underscores are word
    characters, and it is — it normalizes to `x` before anything is anchored.
    """
    normalized = normalize(candidate)
    if not normalized:
        return False
    return re.match(r"\w", normalized) is not None and re.search(r"\w\Z", normalized) is not None
