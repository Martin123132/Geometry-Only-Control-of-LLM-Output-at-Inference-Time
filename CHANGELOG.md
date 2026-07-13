# Changelog

## Unreleased

- Started the `0.1.8` development track with a reproducible offline
  257-candidate scale benchmark covering runtime, Python peak allocations,
  normalized duplicate compression, and serialized report size.
- Removed a redundant single-clause negation-extraction pass identified by the
  scale profile, preserving coordinated-clause handling and all candidate
  decisions while reducing isolated negation-parser median time by `13.636%`
  on the frozen workload.
- Removed duplicate common-helper passes from single-clause positive-relation
  extraction while preserving whole-text handling for coordinated clauses.
  Frozen-workload relation sets and all `257` candidate records remained exact,
  while isolated positive-relation parser median time fell by `27.025%`.
- Reused precomputed empty candidate-negation sets instead of reparsing the
  `172` non-negated unique candidates in the frozen pool. Candidate records
  remained exact while a 16-pair full-pool comparison measured a `12.750%`
  median regulation reduction.
- Added a permanent paired candidate-pool comparison mode with alternating
  execution order, exact report-parity enforcement, per-pair deltas, labelled
  callables, and a configurable performance-regression acceptance threshold.

## 0.1.7 - 2026-07-10

- Started the `0.1.7` development track with deterministic candidate-pool
  selection diagnostics in Python and CLI reports. Reports now distinguish
  candidates blocked before ranking, the emitted candidate, and safe
  alternatives retained for audit without changing emit/block safety behavior.
- Added normalized duplicate grouping for candidate pools. Repeated candidates
  reuse the first matching safety evaluation, stay visible as output rows, and
  expose `pool_group_key` plus `duplicate_of` in reports.
- Added per-report candidate-pool summaries for total, unique, duplicate, safe,
  and blocked candidates across JSON, Markdown, and CSV audit output.
- Added report-only candidate filtering and ordering controls for large pools,
  with explicit view metadata and full-pool safety totals preserved.
- Updated the report schema, README guidance, and roadmap to make candidate
  triage the next bounded hardening target after the `0.1.6` public release.

## 0.1.6 - 2026-07-06

- Added `manifold-check --demo` for a built-in offline demo that works from a
  PyPI install without requiring a local examples checkout.
- Added `--why` as a friendly alias for `--explain` across text, JSON, and
  Markdown regulation reports.
- Added `examples/demo_corpus.jsonl` and `docs/cli_quickstart.md` to give new
  users a small safe emit, blocked negation, and blocked numeric-drift
  walkthrough before reading the full regression corpus.
- Improved README demo guidance for PyPI-only users, repo-checkout users, and
  Markdown audit output.
- Added tests that keep the built-in demo aligned with the committed public
  demo corpus and verify demo text, JSON summary, Markdown audit, and
  `--fail-on-block` behavior.
- Kept the default demo and quickstart paths offline-first with no embedding
  model downloads.

## 0.1.5 - 2026-07-01

- Started the 0.1.5 development cycle after the public 0.1.4 PyPI release.
- Added the exploratory EXP24 challenge corpus seed for broader post-EXP23
  development probes across multi-antecedent conditionals, alias-bound
  permissions, unit/range paraphrases, chained exceptions, compact bindings,
  and all-bad token-reuse near misses.
- Added `docs/exp24_challenge.md` to document the EXP24 workflow and promotion
  boundary.
- Added offline multi-antecedent conditional and role-permission bindings for
  the first EXP24 improvement pass.
- Added scoped unit/range paraphrase bindings for the EXP24 tank, cable, and
  service-window probes.
- Added chained-exception, alias-permission, compact-sensor, and dose-range
  hardening to close the EXP24 seed locally at `18 / 18`.
- Added the exploratory EXP25 challenge corpus seed for the next development
  surface across temporal role binding, nested conditionals, quantifier
  thresholds, compact dimension binding, scoped exceptions, and all-bad
  token-reuse near misses.
- Added compact dimension-binding support for EXP25 weight/volume,
  range/payload, and power/duration probes, raising the local EXP25 baseline
  from `1 / 18` to `4 / 18`.
- Added quantifier-threshold support for EXP25 minimum, maximum, and exact-count
  probes, raising the local EXP25 baseline from `4 / 18` to `7 / 18`.
- Added nested-conditional support for EXP25 badge/room, sensor-alarm, and
  verified-owner export probes, raising the local EXP25 baseline from `7 / 18`
  to `10 / 18`.
- Added scoped-exception support for EXP25 exporter and staff/auditor probes,
  raising the local EXP25 baseline from `10 / 18` to `12 / 18`.
- Added temporal role-binding support for EXP25 repair/inspect,
  approve/reject, and label/seal paraphrase probes, raising the local EXP25
  baseline from `12 / 18` to `15 / 18`.
- Added token-reuse binding support for EXP25 route/dock,
  service-permission, and freezer-threshold all-bad probes, raising the local
  EXP25 baseline from `15 / 18` to `18 / 18`.
- Clarified EXP25 corpus commands, roadmap status, and release-candidate claim
  boundaries after closing the checked EXP25 seed at `18 / 18`.
- Added a Dependabot guard to keep optional `sentence-transformers` major
  updates from widening the supported embedding extra beyond the checked `<3`
  range.
- Clarified README and getting-started guidance around what ManifoldGuard does,
  what it does not do, and how to run a first offline relation-binding demo.

## 0.1.4 - 2026-06-21

- Started the 0.1.4 development cycle after the public 0.1.3 PyPI release.
- Added the exploratory EXP23 challenge corpus seed for broader post-EXP22
  development probes across range bounds, conditional scope, nested exceptions,
  ordinal binding, aggregate binding, and all-bad near misses.
- Added `docs/roadmap.md` to capture the next development track and release
  hardening rhythm.
- Added offline ordinal-binding extraction for EXP23 runner rank, priority
  label, and pipeline stage-slot probes.
- Added offline nested-exception support for EXP23 lab access, animated GIF,
  and emergency walk-in exclusion probes.
- Added offline conditional-scope and permission bindings for EXP23 alarm lock,
  switch/motor, and Team Red/Blue ledger probes.
- Added offline aggregate count-to-group binding for the EXP23 trial
  adult/children enrollment swap probe.
- Added offline applicant age-range paraphrase support for the EXP23
  `at least 18 and under 65` to `18 through 64` bound conversion.

## 0.1.3 - 2026-06-20

- Started the 0.1.3 development cycle after the public 0.1.2 PyPI release.
- Published `manifold-guard==0.1.3` to TestPyPI and PyPI, then verified fresh
  public installs with offline EXP22-style smoke checks.
- Added the exploratory EXP22 challenge corpus seed for broader post-EXP21
  development probes across coreference, comparison, exception scope, unit
  binding, temporal order, and all-bad near misses.
- Added offline identity-binding relation extraction for EXP22 coreference,
  valve/tank, supervision-chain, and numbered pump/boiler swap probes.
- Added offline numeric-comparison binding for EXP22 cost, latency, and
  capacity paraphrases so directed swaps are clamped while supported
  comparative summaries can emit.
- Added scoped exception/exclusion bindings for EXP22 weekday, file-format,
  and form-requirement negation probes.
- Added temporal order and active signing-role bindings for the remaining
  EXP22 before/after and alias-role near-miss probes.

## 0.1.2 - 2026-06-19

- Started the 0.1.2 development cycle after the public 0.1.1 PyPI release.
- Added relation-composition extraction for shared-subject verb chains and
  additional relation verbs so EXP21 recombination swaps are clamped offline.
- Added negation-scope binding for compact contrast objects and ellipsis forms
  so lettered subjects and delayed-opening cases remain aligned offline.
- Added alias-aware acquisition/location bindings plus sea-level and office-date
  guards to close the remaining EXP21 challenge corpus gaps offline.
- Added `docs/exp21_challenge.md` to document the challenge corpus workflow,
  replay packs, report comparison, and promotion boundary.
- Added the EXP21 challenge guide to the product readiness manifest and
  docs-quality required document set.
- Added `docs/benchmark.md` to document evidence tiers, benchmark commands, and
  public claim boundaries.
- Added `docs/getting_started.md` to improve first-run usability,
  onboarding, and personal extension guidance.
- Added `examples/personal_corpus_template.jsonl` as a starter template for
  users building private reference-bounded corpora.
- Added `docs/extending.md` to explain safe data-first and code-level extension
  paths for personal use.
- Added opt-in `manifold-check --explain` decision explanations for text, JSON,
  and Markdown regulation reports.
- Documented `manifold-check --explain` in the README CLI examples.
- Added `examples/explain_report.md` and included it in docs-quality coverage.
- Linked the explanation example from the report schema documentation.
- Added `scripts/compare_eval_reports.py` to compare saved offline evaluator
  JSON reports and highlight summary, family, and case-level changes.
- Added `scripts/build_eval_replay_pack.py` to turn evaluator JSON plus corpus
  rows into compact Markdown or JSON replay packs for failed/selected cases.
- Added an exploratory EXP21 challenge corpus seed for harder non-claim
  development probes across negation scope, aliases, temporal binding,
  relation composition, qualifier overclaims, and all-bad near misses.
- Added evaluator taxonomy prefixes for EXP21 challenge families so challenge
  reports summarize by intended family instead of per-case fragments.
- Added reference-side negative relation support so candidates can preserve
  supplied negative evidence while positive counterclaims are clamped.
- Added temporal binding relation extraction for year/value, version/change,
  quarter/value, opened-date, and start/end-date swaps.

## 0.1.1 - 2026-06-18

- Published `manifold-guard==0.1.1` to PyPI and verified a clean public install
  with `manifold-check --version` plus `manifold-eval` passing 229 / 229 cases.
- Added `manifold-eval` filtering ergonomics with `--family`, `--case-id`,
  `--failures-only`, and `--list-families` for faster offline corpus triage.
- Tightened the optional embeddings extra to `sentence-transformers>=2.6.0,<3`
  and aligned docs-quality enforcement with the release-hardening policy.

- Rebrand the public package and CLI surface to ManifoldGuard (`manifold-guard`, `manifold-check`, `manifold-eval`) while keeping `mbt_ai_tools`, `mbt-check`, and `mbt-eval` as compatibility paths.
- Added a safe manual package build/publish workflow for TestPyPI and PyPI Trusted Publishing.
- Added Markdown evaluation report generation from regulator evaluation JSON artifacts.
- Added package publishing setup documentation for GitHub Actions Trusted Publishing.
- Added package-index installation and smoke-test documentation for TestPyPI and PyPI.
- Added a built-wheel install smoke check to package CI so distribution
  artifacts prove public API and CLI entry points before upload.
- Added per-candidate evaluator diagnostics for clamp summaries, literal drift,
  extracted relations, and safety decisions in offline regression reports.
- Expanded the offline regression corpus to 229 cases with boundary coverage for
  unsupported negation, relation direction swaps, and overclaim clamps.

## 0.1.0 Release Candidate 3 - 2026-06-14

- Added CSV audit export with `--format csv` for spreadsheet-friendly candidate review.
- Added `manifold-check --version` and package `__version__` for reproducible audit logs.
- Added a machine-readable product readiness manifest covering supported install modes, CLI formats, examples, CI gates, and support boundaries.
- Added contributor, security, support, and conduct documentation for product-grade project governance.
- Added structured GitHub issue forms and a pull request template for reproducible bug reports and claim-safe review.
- Added release checklist and quality gates docs for repeatable product-standard release decisions.
- Added Dependabot configuration for weekly GitHub Actions and Python packaging maintenance.
- Added docs-quality GitHub workflow to validate manifest JSON and referenced support/guide files.
- Added JSON schema + sample fixtures for report payloads, with CI validation in docs-quality workflow.
- Added local report fixture validator script to run JSON schema checks in CI and by developers.
- Added unified docs-quality checker script and switched CI workflow to the shared entrypoint.
- Added `scripts/preflight.py` as a one-command release preflight for docs checks and tests.
- Replaced placeholder package URLs with the public GitHub repository URL.
- Added project package URL metadata and propagated canonical project links into `docs/product_readiness_manifest.json`.
- Hardened `docs-quality` by enforcing parity between `pyproject.toml` package URLs and manifest `project.package_urls`.
- Added docs-quality regression coverage for package URL parity mismatches and drift.
- Added CI installability smoke coverage for `pip install -e .` and `pip install -e .[embeddings]` in `.github/workflows/tests.yml` with public entrypoint import checks.
- Hardened manifest validation with install-mode/CI-policy contract checks to prevent missing/duplicate dependency mode definitions.
- Extended `.github/workflows/docs-quality.yml` to run installability matrix checks for both `pip install -e . --no-deps` and `pip install -e .[embeddings]` with import and optional dependency availability assertions.
- Made docs-quality core installability smoke explicitly install runtime deps (`numpy`, `scipy`) before package entrypoint import checks.
- Tightened manifest CI-policy checks to enforce expected `core`/`embeddings` install command contracts and added matching regression tests.
- Enforced exact mode-name parity between `install_modes` and `ci_policy` in the manifest, with regression coverage for mismatches.
- Added docs-quality version contract validation so manifest, `pyproject.toml`, and `mbt_ai_tools/__init__.py` stay synchronized.
- Added changelog alignment enforcement so release notes must contain a section matching the declared package version in `pyproject.toml` / manifest.
- Added distribution metadata parity validation (`importlib.metadata.version`) to docs-quality with regression coverage for installed version match/mismatch behavior.
- Stabilized CLI version regression coverage by reading `mbt_ai_tools.__version__` from package metadata instead of hardcoding a fixed string.
- Added docs-quality workflow manual controls (`workflow_dispatch` inputs), scope-summary consistency checks for both release CI workflows, and a documented release evidence snapshot checklist for operator handoffs.
- Rejected `--token-shock` with `--no-embeddings` at CLI parse time and documented token-shock reporting as embedding-only.
- Added `scripts/release_evidence.py` to generate machine-readable release gate evidence in dry-run or execution mode.
- Added `scripts/validate_examples.py` and refreshed committed CLI JSONL, Markdown, and CSV examples to current offline output.
- Added `scripts/release_readiness.py` for an operator-facing readiness summary with explicit release blockers.
- Added `RELEASE_PROCESS.md` as the canonical maintainer release sequence from install modes through evidence, readiness, CI parity, and tagging.
- Added `scripts/release_check.py` as a one-command release evidence and readiness runner.
- Added `scripts/evaluate_regulator.py` for frozen offline corpus metrics and expected-behavior checks.
- Added a golden offline behavior regression test for key emit/block paths.
- Added CI upload of `release-evidence.json` as a workflow artifact in the core preflight path.
- Added taxonomy summaries and JSON artifact output for frozen offline regulator evaluation.
- Added `manifold-eval` as an installed console command for the offline frozen corpus evaluator.
- Added packaged regression corpus data so `manifold-eval` has a default corpus after installation.

## 0.1.0 Release Candidate 2 - 2026-06-11

- Added CLI JSON regulation reports with emitted candidate index, per-candidate scores, clamp summaries, relations, and negated relations.
- Added optional CLI token-shock reporting controls for regulation reports.
- Added `--output` for writing CLI output to files.
- Added `--input-jsonl` for batch regulation reports.
- Added `--summary` and `--fail-on-block` for CI-friendly batch guard usage.
- Added Markdown regulation and batch audit reports with `--format markdown`.
- Documented the JSON report schema under `docs/`.
- Added a CLI JSON report demo under `examples/`.
- Added a Markdown audit report demo under `examples/`.
- Updated GitHub Actions dependencies to Node 24-compatible major versions.

## 0.1.0 Release Candidate 1 - 2026-06-10

- Kept the default package install offline-first by moving `sentence-transformers` behind the optional `.[embeddings]` extra.
- Added CI coverage for both dependency modes: full offline core tests and deterministic embeddings-mode smoke tests.
- Added clearer missing-dependency guidance for embedding-backed paths.
- Hardened unsupported negation handling across contractions, auxiliary verbs, future modals, and `has no` forms.
- Added configurable `token_shock_map` controls for `max_samples`, `top_k`, and output ordering.
- Documented install modes consistently across README, replication notes, and claims scope.
- Verified the local focused and full suites with `25 passed` and `26 passed`.
