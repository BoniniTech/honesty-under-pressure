"""Check every transcript quoted in the docs against the log it cites.

    python -m hup.transcripts

`tests/test_docs.py` checks that a quoted transcript carries a citation. Nothing checked
that the quote matched what the citation pointed at, and that gap shipped a wrong
transcript: `docs/transcripts.md` said every model reply was reproduced whole while
three of the `q014` replies were cut at their first mention of the pushback answer,
dropping the sentences where the model restated the target, and the `q038` replies had
lost five degree signs. Both were found by hand.

This cannot run in CI. `runs/` is gitignored, so the logs are not on GitHub and the job
would have nothing to compare against. It is a local step, run before a release beside
the figure regeneration that has the same constraint, and by `.githooks/pre-commit` when
a doc carrying a transcript is staged.

**What a quote may do, and how it says so.** A transcript is a blockquote whose turns
are labelled `**User:**` and `**Model:**`. Quoted whole, it must reproduce every turn of
the cited sample in order, exactly. Two abridgements are allowed, and both declare
themselves in the text a reader sees:

* Whole turns dropped, marked by an italic bracketed aside on its own line --
  `*[three rounds of pushback, each asserting 44, none offering any evidence]*`. The
  quoted turns then have to be an in-order subsequence of the sample's, and at least one
  turn has to actually be missing, so the aside cannot claim an elision that never
  happened.
* Text dropped inside a turn, marked by an ellipsis character. The fragments either side
  must appear in the reply in order; a fragment before the first ellipsis has to start
  the reply and one after the last has to end it, so an ellipsis at the edges is what
  says the quote does not run to the end. Three dots is not that character and is
  compared verbatim, which fails.

**What the comparison tolerates.** Whitespace only: runs of it collapse to one space on
both sides before anything is compared. Markdown soft-wraps a quoted reply to fit the
column while the log holds it unwrapped, so a check that read a newline as a difference
would fail on every reflow and get switched off. Everything else is compared character
for character, which is what catches the two defects above -- a dropped sentence, a
missing degree sign, a curly quote flattened to a straight one.

**What it does not cover.** Only labelled transcript blockquotes. A phrase from a reply
quoted inline in a sentence carries a citation and is not checked here; the `q038`
retirement note in the README is the live example.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from inspect_ai.log import read_eval_log_sample

REPO_ROOT = Path(__file__).resolve().parents[2]

# The historical record. Summaries are never edited in place (CLAUDE.md, "Writing up a
# run"), so a transcript quoted in one records what that run said at the time.
_EXCLUDED_DIRS = ("runs/summaries",)

_ROLE_LABELS = {"**User:**": "user", "**Model:**": "assistant"}
_ROLE_NAMES = {"user": "User", "assistant": "Model"}

# `<path>.eval`, then a sample id, then an epoch, in that order. CLAUDE.md, "Writing up a
# run", item 6. The path is long enough to wrap, so the separator is any run of
# whitespace.
_CITATION = re.compile(
    r"`(?P<path>[^`\n]*\.eval)`,?\s*sample `(?P<sample>[^`\n]+)`, epoch (?P<epoch>\d+)"
)

# A run of turns dropped rather than quoted, written as an italic bracketed aside on its
# own line so the gap is visible to a reader as well as to this check.
_OMISSION = re.compile(r"^\*\[.*\]\*$")

_ELLIPSIS = "…"


class TranscriptError(RuntimeError):
    """Raised when a quoted transcript cannot be checked against its log."""


@dataclass(frozen=True)
class Turn:
    role: str
    text: str


@dataclass(frozen=True)
class Quote:
    """One transcript blockquote and the citation standing above it."""

    doc: str
    line: int
    log: str
    sample: str
    epoch: int
    turns: tuple[Turn, ...]
    omits_turns: bool

    @property
    def where(self) -> str:
        return f"{self.doc}:{self.line}"


@dataclass(frozen=True)
class Result:
    quote: Quote
    differences: tuple[str, ...]
    unchecked: str | None = None


def live_docs(root: Path | None = None) -> list[Path]:
    """Every tracked Markdown file a reader is meant to read as current."""
    root = root or REPO_ROOT
    docs = [
        path
        for path in sorted(root.rglob("*.md"))
        if ".venv" not in path.parts
        and not any(excluded in path.relative_to(root).as_posix() for excluded in _EXCLUDED_DIRS)
    ]
    if not docs:
        raise TranscriptError(f"found no live Markdown under {root} -- the glob is wrong")
    return docs


def _normalise(text: str) -> str:
    """Collapse every run of whitespace to one space. See the module docstring."""
    return " ".join(text.split())


def _blockquotes(text: str) -> list[tuple[int, list[str]]]:
    """Each contiguous run of `>` lines, as (1-based start line, undecorated lines).

    A blank line ends a blockquote in Markdown, so a contiguous run is one block. Lines
    come back with one `>` and at most one following space removed, which is the prefix
    a quote is written with.
    """
    blocks: list[tuple[int, list[str]]] = []
    current: list[str] = []
    start = 0
    for number, line in enumerate(text.split("\n"), start=1):
        stripped = line.lstrip()
        if stripped.startswith(">"):
            body = stripped[1:]
            current.append(body[1:] if body.startswith(" ") else body)
            start = start or number
        elif current:
            blocks.append((start, current))
            current, start = [], 0
    if current:
        blocks.append((start, current))
    return blocks


def _parse_turns(lines: list[str]) -> tuple[tuple[Turn, ...], bool]:
    """Split a blockquote's lines into labelled turns, and say whether it drops any."""
    turns: list[Turn] = []
    role: str | None = None
    buffer: list[str] = []
    omits = False

    def flush() -> None:
        if role is not None:
            turns.append(Turn(role=role, text=_normalise("\n".join(buffer))))

    for line in lines:
        label = next((label for label in _ROLE_LABELS if line.startswith(label)), None)
        if label is not None:
            flush()
            role, buffer = _ROLE_LABELS[label], [line[len(label) :]]
        elif _OMISSION.match(line.strip()):
            omits = True
        elif role is not None:
            buffer.append(line)
    flush()
    return tuple(turns), omits


def parse_quotes(path: Path, root: Path | None = None) -> list[Quote]:
    """Every labelled transcript blockquote in one document, with its citation.

    The citation has to sit between the previous blockquote and this one. Searching all
    the preceding text instead would let a transcript that lost its citation silently
    borrow the one above it and get compared against the wrong sample.
    """
    root = root or REPO_ROOT
    relative = os.path.relpath(path, root).replace(os.sep, "/")
    text = path.read_text(encoding="utf-8")
    lines = text.split("\n")

    quotes: list[Quote] = []
    previous_end = 0
    for start, block in _blockquotes(text):
        end = start + len(block) - 1
        turns, omits = _parse_turns(block)
        roles = {turn.role for turn in turns}
        if roles != set(_ROLE_NAMES):
            previous_end = end
            continue
        preamble = "\n".join(lines[previous_end : start - 1])
        previous_end = end
        citations = list(_CITATION.finditer(preamble))
        if not citations:
            raise TranscriptError(
                f"{relative}:{start} quotes a transcript with no `.eval` path, sample id "
                f"and epoch between it and the block above. CLAUDE.md, 'Writing up a "
                f"run', item 6."
            )
        citation = citations[-1]
        quotes.append(
            Quote(
                doc=relative,
                line=start,
                log=citation.group("path"),
                sample=citation.group("sample"),
                epoch=int(citation.group("epoch")),
                turns=turns,
                omits_turns=omits,
            )
        )
    return quotes


def log_turns(path: Path, sample: str, epoch: int) -> tuple[Turn, ...]:
    """The user and assistant turns of one sample, in order.

    Attachments are resolved: the `.eval` format stores long message content out of
    line, and a reply worth quoting is exactly the kind that gets stored that way.
    """
    try:
        recorded = read_eval_log_sample(str(path), sample, epoch, resolve_attachments=True)
    except IndexError as error:
        raise TranscriptError(str(error)) from error
    return tuple(
        Turn(role=message.role, text=_normalise(message.text))
        for message in recorded.messages
        if message.role in _ROLE_NAMES
    )


def _reproduces(quoted: str, actual: str) -> bool:
    """Does a quoted turn reproduce `actual`, honouring any declared elision?"""
    if _ELLIPSIS not in quoted:
        return quoted == actual
    fragments = quoted.split(_ELLIPSIS)
    anchored_start = bool(fragments[0].strip())
    anchored_end = bool(fragments[-1].strip())
    pieces = [piece for fragment in fragments if (piece := fragment.strip())]
    if not pieces:
        return False
    position = 0
    for index, piece in enumerate(pieces):
        found = actual.find(piece, position)
        if found < 0 or (index == 0 and anchored_start and found != 0):
            return False
        position = found + len(piece)
    return not anchored_end or position == len(actual)


def _excerpt(quoted: str, actual: str, width: int = 60) -> str:
    """Both turns from where they first diverge, so the difference is the first thing read."""
    if _ELLIPSIS in quoted:
        return f"  quoted: {quoted[: width * 2]!r}\n  logged: {actual[: width * 2]!r}"
    at = _shared_prefix(quoted, actual)
    start = max(0, at - width // 2)
    return (
        f"  first differ at character {at}\n"
        f"  quoted: {quoted[start : at + width]!r}\n"
        f"  logged: {actual[start : at + width]!r}"
    )


def _shared_prefix(left: str, right: str) -> int:
    return next(
        (index for index, (a, b) in enumerate(zip(left, right)) if a != b),
        min(len(left), len(right)),
    )


def _nearest(quoted: Turn, remaining: tuple[Turn, ...]) -> str:
    """The turn a failed subsequence match was most likely aiming at.

    Without this the report says only that a quoted turn matched nothing, which leaves
    the reader to find the intended reply among however many are left. The candidate
    sharing the longest opening with the quote is the one to diff against, since the
    defects on record all cut or altered a reply partway through.
    """
    candidates = [turn.text for turn in remaining if turn.role == quoted.role]
    if not candidates:
        return ""
    return max(candidates, key=lambda text: _shared_prefix(quoted.text, text))


def _compare(quote: Quote, actual: tuple[Turn, ...]) -> list[str]:
    """Every way a quote departs from the sample it cites."""
    if not quote.omits_turns:
        if len(quote.turns) != len(actual):
            return [
                f"quotes {len(quote.turns)} turns of the sample's {len(actual)} without "
                f"marking an elision. A quote that drops turns says so with an italic "
                f"bracketed aside on its own line."
            ]
        return [
            f"turn {index} ({_ROLE_NAMES[quoted.role]}) does not match the log.\n"
            + _excerpt(quoted.text, recorded.text)
            for index, (quoted, recorded) in enumerate(zip(quote.turns, actual), start=1)
            if quoted.role != recorded.role or not _reproduces(quoted.text, recorded.text)
        ]

    differences: list[str] = []
    position = 0
    for index, quoted in enumerate(quote.turns, start=1):
        start = position
        while position < len(actual) and not (
            actual[position].role == quoted.role and _reproduces(quoted.text, actual[position].text)
        ):
            position += 1
        if position == len(actual):
            differences.append(
                f"turn {index} ({_ROLE_NAMES[quoted.role]}) matches no remaining turn of "
                f"the sample.\n" + _excerpt(quoted.text, _nearest(quoted, actual[start:]))
            )
            break
        position += 1
    if not differences and len(quote.turns) == len(actual):
        differences.append(
            "marks an elision but quotes every turn of the sample. The aside claims a gap "
            "that is not there."
        )
    return differences


def check(quote: Quote, root: Path | None = None) -> Result:
    """Compare one quote against the sample it cites.

    A log that is not on this machine is reported as unchecked rather than as a
    difference. "Could not look" and "looked, and it is wrong" are different findings,
    and a run that merged them would read as a pass with extra noise.
    """
    path = (root or REPO_ROOT) / quote.log
    if not path.exists():
        return Result(
            quote=quote,
            differences=(),
            unchecked=(
                f"{quote.log} is not on this machine. `runs/` is gitignored, so the logs "
                f"come from the run itself and not from a clone."
            ),
        )
    try:
        actual = log_turns(path, quote.sample, quote.epoch)
    except TranscriptError as error:
        return Result(quote=quote, differences=(), unchecked=str(error))
    return Result(quote=quote, differences=tuple(_compare(quote, actual)))


def check_docs(docs: list[Path], root: Path | None = None) -> list[Result]:
    root = root or REPO_ROOT
    return [check(quote, root) for doc in docs for quote in parse_quotes(doc, root)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m hup.transcripts",
        description="Check quoted transcripts against the logs they cite. Reads only.",
    )
    parser.add_argument(
        "docs",
        nargs="*",
        type=Path,
        metavar="DOC",
        help="Markdown files to check. Defaults to every tracked doc outside runs/summaries/.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=REPO_ROOT,
        help=(
            "Directory the cited `.eval` paths are relative to, and the default set of "
            "documents is found under. Defaults to the repo this module was installed from."
        ),
    )
    args = parser.parse_args(argv)

    root = args.root.resolve()
    docs = [path.resolve() for path in args.docs] or live_docs(root)
    results = check_docs(docs, root)
    if not results:
        print("no quoted transcripts found in the documents given", file=sys.stderr)
        return 1

    for result in results:
        head = f"{result.quote.where} {result.quote.sample} epoch {result.quote.epoch}"
        if result.unchecked:
            print(f"UNCHECKED  {head}\n  {result.unchecked}")
        elif result.differences:
            for difference in result.differences:
                print(f"MISMATCH   {head} {difference}")
        else:
            print(f"ok         {head} ({len(result.quote.turns)} turns)")

    mismatched = sum(1 for result in results if result.differences)
    unchecked = sum(1 for result in results if result.unchecked)
    print(
        f"\n{len(results)} transcript(s): {len(results) - mismatched - unchecked} verbatim, "
        f"{mismatched} mismatched, {unchecked} unchecked."
    )
    return 1 if mismatched or unchecked else 0


if __name__ == "__main__":
    raise SystemExit(main())
