<img width="1920" height="640" alt="MBT-5 banner" src="https://github.com/user-attachments/assets/57aa4d12-7d14-4726-a8f1-bd921692d94e" />

# MBT-5 Geometry-Only Output Regulator

MBT-5 tests whether AI candidate outputs remain inside a supplied semantic and relational reference manifold.

It runs at inference time:

- no training
- no fine-tuning
- no model-weight inspection
- no hidden classifier

The regulator checks candidate outputs and either emits the safest supported candidate or blocks when every candidate is unsafe.

## Core Claim

MBT-5 treats hallucination as semantic or relational drift from supplied reference structure. It is not a fact oracle and does not claim direct access to external truth.

```text
Universe does facts.
Humans describe the universe.
AI describes human descriptions.
MBT-5 regulates AI descriptions against supplied human reference structure.
```

Supported public claim:

```text
MBT-5 v11 blocked hallucinated AI outputs against supplied reference manifolds
and relation constraints. In the frozen EXP20 ledger, it achieved confusion
[[97, 0], [0, 160]] over 257 labelled candidates across 53 cases.
```

## Claims and Scope

See [`CLAIMS.md`](CLAIMS.md) for the claim register. The short version: MBT-5 regulates outputs against supplied references and the public offline corpus; it is not claimed to be a universal fact checker.

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

The current public claim is limited to the supplied test suites and reference manifolds included in the project.

## What MBT-5 Checks

MBT-5 combines:

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

Candidate outputs are embedded into semantic space. MBT-5 measures distance from the reference manifold:

```text
shock = Gamma * ||candidate_embedding - reference_center||^2
```

Higher shock means stronger semantic drift.

### Literal Drift Guards

Geometry alone can miss small but important substitutions. MBT-5 protects numbers, units, named entities, and key content tokens.

### Relation Clamps

MBT-5 checks relation structure, not just semantic similarity.

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

MBT-5 blocks unsupported negations of positive reference support.

```text
Water is not liquid at room temperature.
Sound does not need a material medium to travel.
Scientific descriptions do not use measurements or predictions.
General relativity proves gravity has no connection to mass or energy.
```

### Abstention

When every candidate is unsafe, MBT-5 blocks instead of emitting the least-bad candidate.

## Installation

```bash
pip install -e .
```

For semantic-embedding features, install the package dependencies from `pyproject.toml`. Offline literal/relation-only regulation works with `use_embeddings=False`.

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

```bash
mbt-check \
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

## Regression Corpus

The lightweight public regression corpus lives in `examples/regression_corpus.jsonl`. It currently contains 220 offline cases covering entity swaps, multi-word capital handling, all-bad abstention, numeric drift, unit drift, role swaps, shared-subject relation repair, unsupported negation, historical-date drift, supported paraphrase, and overclaim blocking.

Regenerate the corpus:

```bash
uv run python examples/build_regression_corpus.py
```

Run the tests:

```bash
uv run --with pytest python -m pytest -q
```

## Experiment Lineage

The full EXP01-EXP20 record is in `MBT5_EXP01_EXP20_TECHNICAL_LEDGER.md`. The expanded CSV experiment exports live in `data/csv_exports/`.

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
  cli.py               mbt-check command
.github/workflows/
  tests.yml            GitHub Actions offline regression test workflow
CLAIMS.md             scoped public claims register
data/csv_exports/     expanded EXP01-EXP20 CSV exports
examples/
  build_regression_corpus.py
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
