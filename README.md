# ManifoldGuard

![ManifoldGuard banner](https://github.com/user-attachments/assets/57aa4d12-7d14-4726-a8f1-bd921692d94e)

> Rebrand note: ManifoldGuard is the public package/product name for the project
> previously published as `mbt-ai-tools`. The `mbt_ai_tools` import path and
> `mbt-check` / `mbt-eval` CLI aliases remain available for compatibility.
>
> Naming map: product `ManifoldGuard`; install package `manifold-guard`;
> legacy Python import `mbt_ai_tools`; preferred CLIs `manifold-check` and
> `manifold-eval`; legacy CLI aliases `mbt-check` and `mbt-eval`.

[![Offline regression tests](https://github.com/Martin123132/ManifoldGuard/actions/workflows/tests.yml/badge.svg)](https://github.com/Martin123132/ManifoldGuard/actions/workflows/tests.yml)
[![Docs and manifest quality](https://github.com/Martin123132/ManifoldGuard/actions/workflows/docs-quality.yml/badge.svg)](https://github.com/Martin123132/ManifoldGuard/actions/workflows/docs-quality.yml)
[![Package build](https://github.com/Martin123132/ManifoldGuard/actions/workflows/package-publish.yml/badge.svg)](https://github.com/Martin123132/ManifoldGuard/actions/workflows/package-publish.yml)
[![PyPI version](https://img.shields.io/pypi/v/manifold-guard.svg)](https://pypi.org/project/manifold-guard/)

ManifoldGuard tests whether AI candidate outputs remain inside a supplied
semantic and relational reference manifold.

It runs at inference time:

- no training
- no fine-tuning
- no model-weight inspection
- no hidden classifier

The regulator checks candidate outputs and either emits the safest supported
candidate or blocks when every candidate is unsafe.

## 30-Second Quickstart

Install ManifoldGuard from PyPI:

```bash
python -m pip install manifold-guard
```

Run an offline reference-bounded check:

```bash
manifold-check \
  --reference "The capital of France is Paris." \
  --candidate "The capital of France is London." \
  --candidate "The capital of France is Paris." \
  --no-embeddings
```

Expected result:

```text
EMIT | The capital of France is Paris. | score=0.0000
```

Python API:

```python
from mbt_ai_tools import regulate_candidates

result = regulate_candidates(
    [
        "The capital of France is London.",
        "The capital of France is Paris.",
    ],
    ["The capital of France is Paris."],
    use_embeddings=False,
)

print(result.action)
print(result.emitted_text)
```

ManifoldGuard is useful when you already have trusted reference statements and
want to reject outputs that drift away from them.
For a fuller onboarding path, see `docs/getting_started.md`.
For a short CLI walkthrough with safe, blocked, and numeric-drift examples, see `docs/cli_quickstart.md`.
For a starter personal corpus, see `examples/personal_corpus_template.jsonl`.
For extension guidance, see `docs/extending.md`.

## What It Does and Does Not Do

ManifoldGuard is a reference-bounded output regulator. It is strongest when you
give it compact trusted statements and ask whether candidate outputs preserve
the same entities, numbers, units, roles, relations, negations, and scoped
conditions.

It does:

- block unsupported candidate drift from supplied references
- emit a supported candidate when at least one candidate stays inside scope
- abstain when every candidate is unsafe
- explain whether each candidate was blocked before ranking, emitted, or kept
  as a safe alternative
- run offline by default with `--no-embeddings` / `use_embeddings=False`
- expose optional embedding-backed diagnostics when explicitly installed

It does not:

- fact-check the world without references
- prove that a reference is true
- inspect or modify model weights
- replace human review for legal, medical, financial, or safety-critical use
- turn exploratory EXP pass rates into public benchmark claims

## Core Claim

ManifoldGuard treats hallucination as semantic or relational drift from supplied
reference structure. It is not a fact oracle and does not claim direct access
to external truth.

```text
Universe does facts.
Humans describe the universe.
AI describes human descriptions.
ManifoldGuard regulates AI descriptions against supplied human reference structure.
```

Supported public claim:

```text
ManifoldGuard v11 blocked hallucinated AI outputs against supplied reference manifolds
and relation constraints. In the frozen EXP20 ledger, it achieved confusion
[[97, 0], [0, 160]] over 257 labelled candidates across 53 cases.
```

## Claims and Scope

See [`CLAIMS.md`](CLAIMS.md) for the claim register. The short version:
ManifoldGuard regulates outputs against supplied references and the public
offline corpus; it is not claimed to be a universal fact checker.

Continuous integration for the offline corpus lives in [`.github/workflows/tests.yml`](.github/workflows/tests.yml).

## Current Locked Result

Frozen ledger:

```text
MBT5_EXP20_combined_guarded_master_ledger_v11

Candidates: 257
Cases:      53

Candidate-level confusion:
[[TN, FP], [FN, TP]]
[[97, 0], [0, 160]]

Accuracy:   1.0000
Precision:  1.0000
Recall:     1.0000
F1:         1.0000

Case-level:
Correct: 53 / 53
Emitted: 28
Blocked: 25
```

The current public claim is limited to the supplied test suites and reference
manifolds included in the project.

## Known Limitations

- ManifoldGuard is not a universal fact checker.
- It only regulates against the references you provide.
- If the references are wrong, incomplete, or ambiguous, outputs are judged
  against that flawed reference structure.
- Offline mode emphasizes literal, relation, negation, numeric, and unit drift;
  embedding-backed semantic geometry is optional.
- Public performance claims are limited to the frozen supplied corpus and
  documented experiment lineage.
- Token-shock diagnostics require embedding dependencies and are not available
  in strict `--no-embeddings` mode.

## What ManifoldGuard Checks

ManifoldGuard combines:

- semantic geometry
- internal consistency scoring
- token-level shock analysis
- literal drift guards
- entity, number, and unit protection
- overclaim detection
- copular relation checks
- non-copular relation checks
- relation polarity checks
- unsupported negation clamps
- abstention when every candidate is unsafe

Examples of blocked drift:

```text
The capital of France is London.
The Sun is a planet.
Earth is flat.
Water boils at 90 degrees Celsius at sea level.
Gravity is fully solved by modern physics.
Scientific descriptions do not use measurements.
DNA contains the nucleus.
The Sun orbits Earth.
```

## Core Mechanisms

### Semantic Shock

Candidate outputs are embedded into semantic space. ManifoldGuard measures
distance from the reference manifold:

```text
shock = Gamma * ||candidate_embedding - reference_center||^2
```

Higher shock means stronger semantic drift.

### Literal Drift Guards

Geometry alone can miss small but important substitutions. ManifoldGuard
protects numbers, units, named entities, and key content tokens.

### Relation Clamps

ManifoldGuard checks relation structure, not just semantic similarity.

Copular examples:

```text
The Sun is a planet.
A dog is a bird.
Rome is the capital city of France.
```

Non-copular examples:

```text
Earth orbits the Moon.
The Sun orbits Earth.
DNA contains the nucleus.
Heat produces friction.
Photosynthesis converts oxygen into carbon dioxide and water.
```

### Negation Clamp

ManifoldGuard blocks unsupported negations of positive reference support.

```text
Water is not liquid at room temperature.
Sound does not need a material medium to travel.
Scientific descriptions do not use measurements or predictions.
General relativity proves gravity has no connection to mass or energy.
```

### Abstention

When every candidate is unsafe, ManifoldGuard blocks instead of emitting the
least-bad candidate.

## Installation

Install modes:

Public PyPI install:

- Offline/core mode: `python -m pip install manifold-guard`
- Optional semantic mode: `python -m pip install "manifold-guard[embeddings]"`

Source checkout install:

- Offline baseline (default): `python -m pip install -e . --no-deps`
- Optional semantic mode: `python -m pip install -e .[embeddings]`

If you need a plain editable install for local experimentation, use
`pip install -e .` or `python -m pip install -e .`; both remain supported for
compatibility.

The optional extra currently installs `sentence-transformers>=2.6.0,<3` for
model-backed operation.

<!-- markdownlint-disable-next-line MD013 -->
If `sentence-transformers` is unavailable, use offline literal/relation-only regulation with `--no-embeddings` / `use_embeddings=False`.

```python
from mbt_ai_tools import evaluate_candidate

evaluate_candidate(
    "Paris is the capital city of France.",
    ["The capital of France is Paris."],
    use_embeddings=False,
)
```

When embedding-backed operation is requested without `sentence-transformers`,
you'll now get a direct error directing to install the dependency or use
offline mode.

## Python Usage

```python
from mbt_ai_tools import regulate_candidates

references = [
    "The capital of France is Paris.",
    "Paris is the capital city of France.",
]
candidates = [
    "The capital of France is London.",
    "The capital of France is Paris.",
]

result = regulate_candidates(candidates, references, use_embeddings=False)
print(result.action)        # emit
print(result.emitted_text)  # The capital of France is Paris.
```

## CLI Usage

Check the installed CLI version:

```bash
manifold-check --version
```

```bash
manifold-check \
  --reference "The capital of France is Paris." \
  --candidate "The capital of France is London." \
  --candidate "The capital of France is Paris." \
  --no-embeddings
```

Expected output:

```text
EMIT | The capital of France is Paris. | score=0.0000
[0] blocked | ...
[1] safe | ...
```

JSON report output:

```bash
manifold-check \
  --reference "The capital of France is Paris." \
  --candidate "The capital of France is London." \
  --candidate "The capital of France is Paris." \
  --no-embeddings \
  --format json
```

See `examples/cli_json_report.md` for a complete offline JSON report demo.

Decision explanations:

```bash
manifold-check \
  --reference "The capital of France is Paris." \
  --candidate "The capital of France is London." \
  --candidate "The capital of France is Paris." \
  --no-embeddings \
  --why
```

`--why` is a friendly alias for `--explain`. It adds per-candidate summaries
and guard reasons to text, JSON, and Markdown regulation reports without
changing the default report shape.
See `examples/explain_report.md` for a complete offline explanation example.
For a short walkthrough with safe, blocked, and numeric-drift cases, see
`docs/cli_quickstart.md` and `examples/demo_corpus.jsonl`.

Optional token-level shock details can be included in regulation reports when
embedding dependencies are installed:

```bash
manifold-check \
  --reference "The capital of France is Paris." \
  --candidate "The capital of France is London." \
  --format json \
  --token-shock \
  --token-shock-top-k 5
```

Token-level shock is embedding-backed, so keep `--no-embeddings` off when using
`--token-shock`.

Batch JSONL evaluation:

```bash
manifold-check \
  --input-jsonl examples/batch_input.jsonl \
  --no-embeddings \
  --output batch-report.jsonl
```

CI guard mode:

```bash
manifold-check \
  --input-jsonl examples/batch_input.jsonl \
  --no-embeddings \
  --summary \
  --fail-on-block
```

`--fail-on-block` exits with status `2` when a single regulation run blocks or
any batch row blocks. `--summary` appends a final batch summary JSON object.

Markdown audit report:

```bash
manifold-check \
  --input-jsonl examples/batch_input.jsonl \
  --no-embeddings \
  --format markdown \
  --output audit.md
```

See `examples/markdown_audit_report.md` for a complete Markdown audit demo.
For a smaller front-door demo, run:

PyPI-only users can run the built-in offline demo directly:

```bash
manifold-check --demo --why
```

For a full Markdown audit:

```bash
manifold-check --demo --format markdown --why
```

If you cloned the repo, you can also use the committed demo corpus:

```bash
manifold-check \
  --input-jsonl examples/demo_corpus.jsonl \
  --no-embeddings \
  --format markdown \
  --why
```

If you installed from PyPI without cloning the repo, create the same tiny demo
file manually when you want to edit the cases yourself:

```bash
cat > demo_corpus.jsonl <<'JSONL'
{"id":"safe_emit_reference_match","references":["The capital of France is Paris."],"candidates":["The capital of France is Paris."]}
{"id":"blocked_negation","references":["Water is liquid at room temperature."],"candidates":["Water is not liquid at room temperature."]}
{"id":"blocked_numeric_drift","references":["The medication dose is 10 mg."],"candidates":["The medication dose is 20 mg."]}
JSONL

manifold-check \
  --input-jsonl demo_corpus.jsonl \
  --no-embeddings \
  --format markdown \
  --why
```

That demo shows one supported emit, one blocked negation, and one blocked
numeric drift.

JSON, Markdown, and CSV reports also include candidate-pool selection metadata:
`pool_group_key`, `duplicate_of`, `selection_status`, `selection_rank`, and
`selection_reason`. These fields make large candidate pools easier to triage by
separating normalized duplicates, candidates blocked before ranking, the safe
candidate that was emitted, and any safe alternatives retained for audit.

Expected case-level actions:

```text
safe_emit_reference_match -> emit
blocked_negation -> block
blocked_numeric_drift -> block
```

CSV audit export:

```bash
manifold-check \
  --input-jsonl examples/batch_input.jsonl \
  --no-embeddings \
  --format csv \
  --output audit.csv
```

See `examples/csv_audit_report.csv` for a spreadsheet-friendly batch audit demo.

The JSON/Markdown/CSV report schema is documented in `docs/report_schema.md`.
The release support contract is captured in `docs/product_readiness_manifest.json`.
Release quality expectations are captured in `docs/quality_gates.md` and `docs/release_checklist.md`.
The maintainer release flow is documented in `RELEASE_PROCESS.md`.

## Local Validation

For schema-driven sanity checks before publishing docs or changing report contracts:

```bash
python -m pip install jsonschema
python scripts/validate_reports.py \
  --schema docs/report_schema.json \
  examples/single_report_example.json \
  examples/batch_report_example.jsonl
```

Run the full local docs-quality check (includes manifest and example consistency):

```bash
python scripts/docs_quality.py
```

Run the full preflight (docs-quality + pytest) before review:

```bash
python scripts/preflight.py
```

Run only docs checks:

```bash
python scripts/preflight.py --docs-only
```

Run only tests:

```bash
python scripts/preflight.py --tests-only
```

Run the frozen offline regression corpus evaluation:

```bash
manifold-eval
manifold-eval --output regulator-evaluation.json
manifold-eval --list-families
manifold-eval --family unsupported_negation
manifold-eval --case-id unsupported_negation_water
manifold-eval --failures-only --format json --output failing-cases.json
python scripts/evaluate_regulator.py
python scripts/evaluate_regulator.py --output regulator-evaluation.json
python scripts/build_eval_report.py --input regulator-evaluation.json --output docs/evaluation_report.md
```

`manifold-eval` uses the packaged regression corpus by default and accepts
`--corpus path/to/corpus.jsonl` for custom offline checks. Use `--family` or
`--case-id` to narrow a regression pass, `--failures-only` to export only
failing case details, and `--list-families` to inspect available taxonomy
groups.
The generated benchmark report lives at `docs/evaluation_report.md`.
Benchmark/evidence tiers and claim boundaries are documented in
`docs/benchmark.md`.

Compare two saved evaluator JSON reports:

```bash
python scripts/compare_eval_reports.py \
  --before baseline-evaluation.json \
  --after candidate-evaluation.json \
  --output evaluation-diff.json
```

Build a compact replay pack for failed or selected evaluator cases:

```bash
python scripts/build_eval_replay_pack.py \
  --evaluation regulator-evaluation.json \
  --corpus examples/regression_corpus.jsonl \
  --output eval-replay.md
```

## Package Build and Publish

Package distribution artifacts are built by
`.github/workflows/package-publish.yml` on version tags and manual runs.
The build job also installs the built wheel in a clean virtual environment and
runs `scripts/install_smoke.py` before artifacts are uploaded or published.
Publishing is manual-only: run the workflow with `publish=true` and choose
`testpypi` or `pypi` after configuring PyPI Trusted Publishing environments
named `testpypi` and `pypi`.

The exact Trusted Publishing setup values are documented in `docs/package_publishing.md`.
Package-index install commands and smoke checks are documented in `docs/package_installation.md`.
The `v0.1.3` post-release verification note is recorded in
`docs/v0.1.3_release_verification.md`.

Recommended release order:

```bash
python -m build
python -m twine check dist/*
```

Use TestPyPI first for release-candidate publishing. Do not publish to PyPI
until tag CI, release evidence, and the regulator evaluation report are green.

Run the canonical release check sequence:

```bash
python scripts/release_check.py --output release-evidence.json
```

## Release Evidence Snapshot

Use this command block before release candidates or public claims:

```bash
python -m pytest -q
python scripts/docs_quality.py
python scripts/evaluate_regulator.py
python scripts/preflight.py
python scripts/preflight.py --docs-only
python scripts/release_check.py --output release-evidence.json
python -m pytest -q tests/test_docs_quality.py
```

Expected evidence in a healthy release path:

- A full `pytest` success summary (all tests passed, no failures).
- docs-quality validator exits successfully.
- regulator evaluation reports all frozen corpus cases passing, with taxonomy
  metrics in JSON output.
- preflight prints `Preflight completed successfully.` for both full and
  docs-only modes.
- no schema or URL drift errors.

## Regression Corpus

The lightweight public regression corpus lives in
`examples/regression_corpus.jsonl`. It currently contains 229 offline cases
covering entity swaps, multi-word capital handling, all-bad abstention, numeric
drift, unit drift, role swaps, shared-subject relation repair, unsupported
negation, historical-date drift, supported paraphrase, relation-direction
boundaries, negation boundaries, and overclaim blocking.

Regenerate the corpus:

```bash
uv run python examples/build_regression_corpus.py
```

An exploratory EXP21 challenge seed lives in `examples/challenge_corpus.jsonl`.
It is separate from the frozen release/regression evidence and is intended for
harder development probes, not current public pass-rate claims.
The EXP21 workflow is documented in `docs/exp21_challenge.md`.

The next post-release challenge seed lives in
`examples/exp22_challenge_corpus.jsonl`. EXP22 is broader and intentionally
rougher than EXP21, with guidance in `docs/exp22_challenge.md`.

The EXP23 development seed lives in `examples/exp23_challenge_corpus.jsonl`.
EXP23 closed the `0.1.4` track at `18 / 18` for the checked seed cases.

The EXP24 development seed lives in `examples/exp24_challenge_corpus.jsonl`.
EXP24 closed the `0.1.5` track at `18 / 18` for the checked seed cases. Its
workflow is documented in `docs/exp24_challenge.md`; EXP24 remains development
evidence, not a benchmark claim.

The current exploratory seed lives in `examples/exp25_challenge_corpus.jsonl`.
EXP25 covers temporal role binding, nested conditionals, quantifier thresholds,
compact dimension binding, scoped exceptions, and all-bad token-reuse near
misses. Current local EXP25 seed status is `18 / 18` after compact
dimension-binding, quantifier-threshold, nested-conditional, scoped-exception,
temporal role-binding, and all-bad token-reuse passes. Its workflow is
documented in `docs/exp25_challenge.md`; EXP25 is development evidence, not a
benchmark claim. The broader development direction is recorded in
`docs/roadmap.md`.

```bash
uv run python examples/build_exp25_challenge_corpus.py
manifold-eval --corpus examples/exp25_challenge_corpus.jsonl --output exp25-evaluation.json
python scripts/build_eval_replay_pack.py \
  --evaluation exp25-evaluation.json \
  --corpus examples/exp25_challenge_corpus.jsonl \
  --output exp25-replay.md
```

For release candidates, treat EXP25 as supporting development evidence only.
Release claims should still come from the frozen regression suite, release
evidence artifact, and `CLAIMS.md`.

Run the tests:

```bash
uv run --with pytest python -m pytest -q
```

## Experiment Lineage

The full EXP01-EXP20 record is in `MBT5_EXP01_EXP20_TECHNICAL_LEDGER.md`.
The expanded CSV experiment exports live in `data/csv_exports/`.

Key frozen output artifacts:

```text
data/csv_exports/mbt5_exp20_master_candidate_ledger.csv
data/csv_exports/mbt5_exp20_master_case_ledger.csv
data/csv_exports/mbt5_exp20_summary_metrics.csv
data/csv_exports/mbt5_exp20_case_summary.csv
data/csv_exports/mbt5_exp20_clamp_counts.csv
data/csv_exports/mbt5_exp20_failure_table.csv
data/csv_exports/mbt5_exp20_patch_lineage.csv
```

A `docs-quality` workflow also validates manifest JSON and referenced
docs/examples so support artifacts stay consistent.

## Project Layout

```text
mbt_ai_tools/
  mbt/
    embeddings.py      SentenceTransformer loader
    geometry.py        geometric median, shock, distance
    stability.py       self-consistency / entropy scoring
    tokens.py          leave-one-out token shock
    consensus.py       multi-agent / council logic
    regulator.py       v11 candidate regulator
  data/
    regression_corpus.jsonl
  cli.py               manifold-check command
  eval.py              manifold-eval offline frozen corpus evaluator
.github/workflows/
  tests.yml            GitHub Actions offline regression test workflow
  docs-quality.yml     docs and manifest quality workflow
  package-publish.yml  package build and manual publish workflow
CHANGELOG.md          release notes
CLAIMS.md             scoped public claims register
RELEASE_PROCESS.md    maintainer release flow
data/csv_exports/     expanded EXP01-EXP20 CSV exports
scripts/
  docs_quality.py
  build_eval_report.py
  evaluate_regulator.py
  release_evidence.py
  release_readiness.py
  release_check.py
  install_smoke.py
  preflight.py
  validate_reports.py
docs/
  product_readiness_manifest.json
  report_schema.json
  quality_gates.md
  release_checklist.md
  report_schema.md
examples/
  batch_input.jsonl
  build_regression_corpus.py
  cli_json_report.md
  csv_audit_report.csv
  markdown_audit_report.md
  batch_report_example.jsonl
  single_report_example.json
  regression_corpus.jsonl
tests/
  test_regulator.py
  test_regression_corpus.py
REPLICATION.md        local/GitHub/Colab replication instructions
pyproject.toml
README.md
LICENSE
```

## License

See `LICENSE`.
