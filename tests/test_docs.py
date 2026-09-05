"""The commands in the live docs are executed by something other than a reader.

`hup.pool`, `hup.chart` and `hup.rescore` take a glob of `.eval` logs, and the wide
form `runs/<run>/*.eval` does not error — it returns a number, computed over runs kept
on purpose to stay out of any pooled result. That exact glob survived in `CLAUDE.md`
for hours after the same bug was fixed in `README.md`, because nothing compared them.

Scope, stated so a green run is not read as more than it is. This checks the `.eval`
path arguments of documented `python -m hup.*` invocations, in tracked Markdown outside
`runs/summaries/`. It does not check `inspect eval` command lines, prose that describes
a glob without invoking one, or the summaries themselves — those are an append-only
historical record and a superseded command quoted inside a correction is supposed to
stay wrong.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# The historical record. Summaries are never edited in place (CLAUDE.md, "Writing up a
# run"), so a command quoted in one is evidence rather than instruction.
_EXCLUDED_DIRS = ("runs/summaries",)

# `python -m hup.<module> <args...>`, stopping at a closing backtick or end of line so a
# prose mention inside a sentence yields its module name and no arguments.
_INVOCATION = re.compile(r"python -m hup\.(?P<module>\w+)(?P<args>[^\n`]*)")

_EVAL_ARG = re.compile(r"\S*\.eval\b")


def _live_docs() -> list[Path]:
    docs = [
        path
        for path in sorted(REPO_ROOT.rglob("*.md"))
        if ".venv" not in path.parts
        and not any(
            excluded in path.relative_to(REPO_ROOT).as_posix() for excluded in _EXCLUDED_DIRS
        )
    ]
    assert docs, "found no live Markdown to check — the glob is wrong, not the docs"
    return docs


def _invocations() -> list[tuple[str, str, str]]:
    """Every documented `python -m hup.*` call, as (doc path, module, argument string)."""
    found = []
    for path in _live_docs():
        relative = path.relative_to(REPO_ROOT).as_posix()
        for match in _INVOCATION.finditer(path.read_text(encoding="utf-8")):
            found.append((relative, match.group("module"), match.group("args")))
    return found


def test_docs_invoke_at_least_one_hup_module() -> None:
    """Guards the checks below: a regex that matched nothing would pass silently."""
    modules = {module for _, module, _ in _invocations()}
    assert {"pool", "chart", "rescore"} <= modules


def test_documented_modules_exist() -> None:
    for doc, module, _ in _invocations():
        assert (REPO_ROOT / "src" / "hup" / f"{module}.py").is_file(), (
            f"{doc} documents `python -m hup.{module}`, which is not a module in src/hup/"
        )


@pytest.mark.parametrize("doc,module,args", _invocations())
def test_eval_globs_are_scoped_to_pass_directories(doc: str, module: str, args: str) -> None:
    """A documented `.eval` glob reaches pass directories only.

    `runs/<run>/` also holds `partial-hang/`, `failed-3.7-alias/`, `superseded/` and
    `pre-rescore/`. The narrow glob excludes them by construction; the wide one pools
    them into a published number without complaining.
    """
    for path in _EVAL_ARG.findall(args):
        parts = path.split("/")
        assert parts[0] == "runs", f"{doc}: `hup.{module}` glob {path!r} is not under runs/"
        assert len(parts) >= 4 and parts[2].startswith("pass"), (
            f"{doc}: `hup.{module}` glob {path!r} is not scoped to pass directories. "
            f"Use runs/<run>/pass*/*.eval — the wider glob silently pools the runs kept "
            f"under runs/<run>/ as a record of what went wrong."
        )


# A transcript is a paragraph quoting both sides of an exchange. Matching the *paragraph*
# rather than a line is deliberate: the defect this guards against welded the citation
# onto the first quoted line (`Source: > **User:** How many...`), which no line-anchored
# pattern sees as a quote at all, so it would have been skipped rather than failed.
_USER_TURN = "**User:**"
_MODEL_TURN = "**Model:**"

# `<path>.eval`, then a sample id, then an epoch, in that order. CLAUDE.md, "Writing up a
# run", item 6. The path may wrap, so the separator is any run of whitespace.
_CITATION = re.compile(r"\.eval`?,?\s*sample `\w+`, epoch \d+")

# Two transcripts in docs/transcripts.md and one abridged in the README. Asserted as a
# floor so that losing one, or a matcher that stops recognising one, fails loudly instead
# of quietly checking fewer things.
_EXPECTED_TRANSCRIPTS = 3


def _transcripts() -> list[tuple[str, int, str]]:
    """Every quoted transcript in live Markdown, as (doc path, line number, preceding text)."""
    found = []
    for path in _live_docs():
        relative = path.relative_to(REPO_ROOT).as_posix()
        text = path.read_text(encoding="utf-8")
        offset = 1
        for block in text.split("\n\n"):
            is_quote = any(line.lstrip().startswith(">") for line in block.split("\n"))
            if is_quote and _USER_TURN in block and _MODEL_TURN in block:
                preamble = "\n".join(text[: text.index(block)].split("\n")[-5:])
                found.append((relative, offset, preamble))
            offset += block.count("\n") + 2
    return found


def test_docs_contain_every_known_transcript() -> None:
    """Guards the check below: a matcher that found nothing would pass silently."""
    found = _transcripts()
    assert len(found) >= _EXPECTED_TRANSCRIPTS, (
        f"expected at least {_EXPECTED_TRANSCRIPTS} quoted transcripts, matched {len(found)}: "
        f"{[f'{doc}:{line}' for doc, line, _ in found]}. Either a transcript was lost, or the "
        f"matcher stopped recognising one — both mean this file is checking less than it says."
    )


@pytest.mark.parametrize("doc,line,preamble", _transcripts())
def test_quoted_transcripts_carry_a_citation(doc: str, line: int, preamble: str) -> None:
    """A quoted transcript names the log it came from, its sample and its epoch.

    The logs are gitignored, so a reader cannot go and look — the citation is the only
    thing tying a quote to the run that produced it. A docs move dropped one on
    2026-09-05 and green CI said nothing, because nothing compared them.
    """
    assert _CITATION.search(preamble), (
        f"{doc}:{line} quotes a transcript with no `.eval` path, sample id and epoch in "
        f"the lines above it. CLAUDE.md, 'Writing up a run', item 6."
    )
