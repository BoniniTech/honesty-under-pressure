"""Re-score existing eval logs with the current scorer, without paying for a run.

    python -m hup.rescore --dry-run runs/d5/pass*/*.eval    # report, write nothing
    python -m hup.rescore runs/d5/pass*/*.eval              # rewrite in place

Makes no provider calls: `flip_scorer` is containment over text already in the log.
Inspect still constructs a client for the model named in the log, so a key has to be
present in the environment; any string works, and nothing is sent. That is deliberately
not faked here — a module that quietly invents credentials is a worse habit than an
error message.

**What this can and cannot backdate.** A change to the scorer's own logic applies to
every log you already have. A change to what the dataset records about a sample does
not: sample metadata is written when the sample runs, so a log recorded before a field
existed carries nothing for the scorer to read. Answer aliases are the worked example —
they live in the dataset, which is where a human can review them, and the cost of that
placement is that fixing an alias needs a fresh run rather than a re-score. See the
"A fix that could not be backdated" section in the README.

Rewrites in place, so back the logs up first if they are the only copy. `runs/` is
gitignored and a re-run costs money.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from inspect_ai import score
from inspect_ai.log import EvalLog, read_eval_log, write_eval_log

from hup.scorers import flip_scorer


class RescoreError(RuntimeError):
    """Raised when a log cannot be re-scored into an honest result."""


@dataclass(frozen=True)
class RescoreResult:
    path: Path
    samples: int
    changed: list[str]


def _score_metadata(sample: object) -> dict:
    scores = sample.scores  # type: ignore[attr-defined]
    return next(iter(scores.values())).metadata or {}


def rescore_log(path: Path) -> tuple[EvalLog, RescoreResult]:
    """Re-score one log, reporting which samples' verdicts moved.

    Refuses a log that did not finish, for the same reason `hup.pool` does: a partial
    pass weights whichever items ran before the failure, and re-scoring it would launder
    that into something that looks complete.
    """
    original = read_eval_log(str(path))
    if original.status != "success":
        raise RescoreError(f"{path.name}: status is {original.status!r}, not 'success'")
    if not original.samples:
        raise RescoreError(f"{path.name}: no samples")

    rescored = score(original, flip_scorer(), action="overwrite", display="none")

    changed = [
        str(old.id)
        for old, new in zip(original.samples, rescored.samples, strict=True)
        if _score_metadata(old) != _score_metadata(new)
    ]
    return rescored, RescoreResult(path=path, samples=len(original.samples), changed=changed)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m hup.rescore",
        description="Re-score eval logs with the current scorer. Rewrites in place.",
    )
    parser.add_argument("logs", nargs="+", type=Path, metavar="LOG", help=".eval log files.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would change without writing anything.",
    )
    args = parser.parse_args(argv)

    total_changed = 0
    for path in args.logs:
        rescored, result = rescore_log(path)
        total_changed += len(result.changed)
        if result.changed:
            print(
                f"{path}: {len(result.changed)} of {result.samples} changed "
                f"({', '.join(sorted(set(result.changed)))})"
            )
        if not args.dry_run:
            write_eval_log(rescored, str(path))

    verb = "would change" if args.dry_run else "changed"
    print(f"\n{len(args.logs)} log(s), {total_changed} sample verdict(s) {verb}.")
    if args.dry_run:
        print("Nothing written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
