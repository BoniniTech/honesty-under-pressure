# Summary template

Copy this to `runs/summaries/full-<date>.md` when writing up a results run. Copying the
previous run's summary instead carries its corrections and its history into a file that
has neither yet, and both are hard to notice once they are there.

The order below is fixed by `CLAUDE.md`, "Writing up a run", so that two runs can be
read against each other. Skip a section only when the run genuinely has nothing for it,
and say so in one line rather than dropping the heading — an absence a reader can see is
worth more than a heading nobody misses.

Delete this preamble and everything in angle brackets.

---

# Full run — <date>

<Two or three sentences, for someone who has never opened the repo: what the eval
measures, what this run was, and what makes it different from the last one. No jargon
and no metric names.>

**Latest published results.** These numbers are on the README.

<If corrections exist, one line each, newest first. The date, the one clause that
changes what the run supports, and a link to the full entry below. Not the text — that
lives in `## Corrections`, so this section stays short enough that the TL;DR is still
visible.>

- **<date>:** <what changed in one clause> — [see below](#corrections)

## TL;DR

<The conclusion as it stands after every correction in this file. What the run
establishes, and what it does not. If a correction changed the standing count, give the
standing count here and say which number it replaces.>

## Corrections

<Verbatim, in writing order, oldest first. Never edited in place, never folded into the
text they correct. Each one says what was written, what is now known, and what it
changes about the claims below. State plainly whether the tables are restated — they
normally are not, and a reader has to be told.>

### Correction, <date>: <what was wrong>

## What was run

<Passes, models with the id read back from the response, escalation depth, commit,
`inspect-ai` version, and the date. Then the glob every number below regenerates from —
`pass*/`, never the run directory.>

```
python -m hup.pool runs/<run>/pass*/*.eval
python -m hup.chart runs/<run>/pass*/*.eval --output-dir analysis
```

## The flips are <n> questions, not a tendency

<Per-item table first: a cell rate reads as a uniform tendency and a handful of flips
from a handful of items is not one. Legend every column abbreviation, one line each.>

## Pooled cells

<The per-cell table, exactly as `python -m hup.pool` prints it now. Never promoted from
an earlier addendum — one in `full-2026-08-19.md` predates a re-score and still reads a
superseded number. Legend every column abbreviation again; this table has different
ones.>

## Where in the ladder

<Only if the run used more than one round. Where each flip landed: a round, or the
readout turn after the model argued through every round. Say what the round columns
cannot see — rounds that named both candidates are undecidable, so the counts are a
floor.>

## Ambiguity by item

<Which items the scorer could not adjudicate, and for which models. A cell rate says how
much containment could not decide; it does not say which questions produced it, and the
difference decides whether the cause is the dataset or the model.>

## Two mechanisms, in the model's own words

<Transcripts. Every one carries its `.eval` path, sample id and epoch as text above the
quote, not as a link — the logs are gitignored, so a link is dead for exactly the reader
the citation is for. A transcript of a since-retired item carries the retirement notice
beside it.>

## What went wrong, and what it cost

<Every failed, quarantined or re-run pass, which directory it is kept in, and what it
cost the result — precision, a conclusion, or nothing. A run where nothing went wrong
says that in one line rather than dropping the heading.>

## Spend

<Measured tokens and billed cost per model, against what `python -m hup.budget`
projected. Where a figure is arithmetic rather than a measurement, say so and say which
way it is wrong.>

## What this does not establish

<The claims a reader might take from the tables and should not. Interval overlap, item
concentration, anything the design cannot separate, and the gap between the models
tested and the products they sit behind.>

## History

<details> <summary>Superseded tables</summary>

<Never deleted, never edited in place, marked as superseded here.>

</details>
