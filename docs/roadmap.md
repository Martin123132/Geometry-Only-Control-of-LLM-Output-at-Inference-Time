# ManifoldGuard Roadmap

This roadmap keeps development grounded in reference-bounded behavior. It is a
working plan, not a public guarantee.

## Current release baseline

- `manifold-guard==0.1.7` is published to TestPyPI and PyPI.
- Core regulation remains offline-first when `use_embeddings=False` or
  `--no-embeddings` is selected.
- EXP22 is closed at `18 / 18` for the checked seed cases.
- EXP23 is closed locally at `18 / 18` for the checked seed cases on the
  `0.1.4` development track.
- EXP24 is closed locally at `18 / 18` for the checked seed cases on the
  `0.1.5` development track.
- EXP25 is closed locally at `18 / 18` for the checked seed cases and should be
  treated as supporting development evidence, not a public benchmark claim.
- `0.1.6` shipped the built-in offline demo, `--why` alias, public demo corpus,
  CLI quickstart, and release verification record.
- `0.1.7` shipped candidate-pool diagnostics, duplicate grouping, summary
  counts, report-only filtering and ordering, and a public release verification
  record.

## Completed 0.1.7 scope

The bounded `0.1.7` candidate-pool ergonomics scope is complete. EXP24 and EXP25
remain closed milestone corpora, and the release improves how large candidate
pools are reviewed without changing public safety claims.

Completed scope:

- add deterministic candidate selection diagnostics to separate blocked
  candidates, the emitted candidate, and safe alternatives
- add normalized duplicate grouping so repeated candidate strings reuse the
  first evaluation while staying visible in audit output
- add compact per-report candidate-pool counts so large pools expose total,
  unique, duplicate, safe, and blocked volumes before row-level inspection
- add optional report sorting and filtering with explicit view metadata while
  preserving full-pool decisions, totals, and original candidate indices
- keep emit/block behavior unchanged unless a focused failing fixture justifies
  a safety change
- update JSON, Markdown, CSV, and docs contracts together when diagnostics
  become public report fields
- add a small roadmap/changelog note before broadening into EXP26-style probes
- leave public claims tied to release evidence, not exploratory pass rates

Release result: `v0.1.7` is tagged at `babe86d`, published through trusted
publishing, and verified from clean TestPyPI and PyPI installations.

## Next development track

Start `0.1.8` with a measured 257-candidate scale benchmark covering runtime,
memory, normalized duplicate compression, and serialized report size. Optimize
only where the measured baseline identifies a meaningful bottleneck.

## Product hardening

- Keep default installs lightweight and offline-first.
- Keep optional embeddings explicit through `.[embeddings]`.
- Add new guards only when they are explainable from supplied references.
- Prefer challenge-corpus seeds before changing regulator behavior.
- Keep public claims tied to reproducible release evidence.

## Release rhythm

- Open each cycle with an unreleased changelog section.
- Add a new exploratory seed before broad regulator changes.
- Close each promoted family with focused tests.
- Publish only after local validation, CI, TestPyPI smoke, and PyPI smoke.
