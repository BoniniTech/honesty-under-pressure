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
