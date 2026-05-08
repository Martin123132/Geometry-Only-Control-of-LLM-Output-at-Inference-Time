<img width="1920" height="640" alt="a653b64f-4fd4-46ec-b425-dcab2cb3c7b4" src="https://github.com/user-attachments/assets/57aa4d12-7d14-4726-a8f1-bd921692d94e" />

MBT-5 tests whether an AI output stays inside a supplied semantic and relational reference manifold.
It runs at inference time.
No training.  
No fine-tuning.  
No model-weight inspection.  
No hidden classifier.
The regulator checks candidate outputs and either:
1. emits the safest supported candidate, or
2. blocks / abstains when every candidate is unsafe.
## Core Claim
MBT-5 treats AI hallucination as **semantic or relational drift**.
A hallucination is not defined here as “lack of access to universal fact”.  
It is defined as the model leaving the intended answer manifold and producing unsupported content.
```text
Universe does facts.
Humans describe the universe.
AI describes human descriptions.
MBT-5 regulates AI descriptions against supplied human reference structure.

MBT-5 is therefore not a fact oracle.
It is an output regulator.

It tests whether an AI output remains consistent with the reference descriptions, candidate set, semantic manifold, and relation constraints it was given.

Current Locked Result

Frozen ledger:

MBT5_EXP20_combined_guarded_master_ledger_v11

Result:

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

In the frozen EXP20 ledger, MBT-5 produced no locked false positives and no locked false negatives.

Supported claim:

MBT-5 v11 blocked hallucinated AI outputs against supplied reference manifolds and relation constraints. In the frozen EXP20 ledger, it achieved confusion [[97, 0], [0, 160]] over 257 labelled candidates across 53 cases.

<img width="1448" height="1086" alt="5fc1387a-ba8a-49a9-aa92-fae7671591b1" src="https://github.com/user-attachments/assets/1f61db20-f95b-4d31-b325-9dfd5f37cdfc" />

What MBT-5 Checks

MBT-5 combines:

* semantic geometry
* internal consistency scoring
* token-level shock analysis
* literal drift guards
* entity / number / unit protection
* overclaim detection
* copular relation checks
* non-copular relation checks
* relation polarity checks
* unsupported negation clamps
* abstention when every candidate is unsafe

Examples of Blocked Drift

MBT-5 is designed to catch outputs such as:

The capital of France is London.
The Sun is a planet.
Earth is flat.
Water boils at 90 degrees Celsius at sea level.
Gravity is fully solved by modern physics.
Scientific descriptions do not use measurements.
DNA contains the nucleus.
The Sun orbits Earth.

These are treated as output-regulation failures: the candidate has left the supplied semantic or relational support structure.

Core Mechanisms

1. Semantic Shock

Candidate outputs are embedded into semantic space.

MBT-5 measures distance from the reference manifold:

shock = Γ · ||candidate_embedding - reference_center||²

Higher shock means stronger semantic drift.

2. Internal Entropy

Multiple candidate answers can be sampled.

MBT-5 measures disagreement between candidates in embedding space.

High internal entropy means the output set is unstable.

3. Token Shock

MBT-5 removes tokens one at a time and measures how much each token contributes to drift.

Example:

The capital of France is London.

Token shock localises to:

London

4. Literal Drift Guards

Geometry alone can miss small but important substitutions.

Example:

Water boils at 90 degrees Celsius at sea level.

This is semantically close to the supported statement, but the number is wrong.

MBT-5 therefore protects:

* numbers
* units
* named entities
* key content tokens

5. Relation Clamps

MBT-5 checks relation structure, not just semantic similarity.

Copular examples:

The Sun is a planet.
A dog is a bird.
Rome is the capital city of France.

Non-copular examples:

Earth orbits the Moon.
The Sun orbits Earth.
DNA contains the nucleus.
Heat produces friction.
Photosynthesis converts oxygen into carbon dioxide and water.

6. Negation Clamp

MBT-5 blocks unsupported negations of positive reference support.

Examples:

Water is not liquid at room temperature.
Sound does not need a material medium to travel.
Scientific descriptions do not use measurements or predictions.
General relativity proves gravity has no connection to mass or energy.

<img width="1448" height="1086" alt="a2c6cb64-aff1-4f53-b7ea-ca7beb634eef" src="https://github.com/user-attachments/assets/8e598225-bda0-4ff2-a402-3ea6a5d7a5ae" />

7. Abstention

When every candidate is unsafe, MBT-5 blocks instead of emitting the least-bad candidate.

Experiment Lineage

EXP01  Basic geometry tested against hard negatives.
EXP02  Added literal guards.
EXP03  Tested regulated candidate selection.
EXP04  Tested abstention when no valid candidate exists.
EXP05  Added unsupported content-token clamp.
EXP06  Tested naive paraphrase tolerance; it was too permissive.
EXP06B Added guarded paraphrase tolerance.
EXP07  Added adversarial generalisation suite.
EXP08  Stress-tested known false negatives in all-bad pools.
EXP09  Added relation polarity clamp.
EXP10  Applied relation clamp back to EXP07.
EXP10B Fixed relation false positive.
EXP11  Confirmed EXP10B against EXP08.
EXP12  Built first master regression ledger.
EXP13  Tested non-copular verb relations.
EXP13B Added non-copular relation clamp.
EXP13C Fixed coordinated verb subject handling.
EXP14  Stress-tested valid paraphrases.
EXP14B Added reference-member override.
EXP15  Tested large all-bad abstention pools.
EXP15B Added negated positive support clamp.
EXP16B Built locked master regression ledger.
EXP17  Unseen holdout exposed relation recall misses.
EXP18  Added relation recall patch.
EXP19  Found unguarded EXP18 patch regression.
EXP19B Guarded EXP18 patch and fixed regression.
EXP20  Frozen combined guarded v11 ledger.

EXP20 Output Files

mbt5_exp20_master_candidate_ledger.csv
mbt5_exp20_master_case_ledger.csv
mbt5_exp20_summary_metrics.csv
mbt5_exp20_case_summary.csv
mbt5_exp20_clamp_counts.csv
mbt5_exp20_failure_table.csv
mbt5_exp20_patch_lineage.csv

Installation

pip install mbt-ai-tools

Basic Usage

from mbt_ai_tools import confidence_score, hallucination_risk, token_shock_map
text = "The capital of France is Paris."
score = confidence_score(text)
risk = hallucination_risk(text)
shocks = token_shock_map(text)
print(score)
print(risk)
print(shocks)

Candidate Regulation

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

result = regulate_candidates(candidates, references)
print(result.action)        # emit
print(result.emitted_text)  # The capital of France is Paris.
```

For offline literal/relation-only checks, pass `use_embeddings=False`. The regulator still applies literal drift, relation, negation, overclaim, reference-member, and abstention clamps.

CLI

mbt-check "The capital of France is Paris."

The CLI reports a confidence / risk label and the numeric internal entropy score.

Project Layout

mbt_ai_tools/
├── mbt/
│   ├── embeddings.py      # SentenceTransformer loader
│   ├── geometry.py        # geometric median, shock, distance
│   ├── stability.py       # self-consistency / entropy scoring
│   ├── tokens.py          # leave-one-out token shock
│   ├── consensus.py       # multi-agent / council logic
│   └── utils.py
├── cli.py                 # mbt-check command
├── pyproject.toml
├── README.md
└── LICENSE


## Research Status

MBT-5 v11 is the current frozen public ledger.

The current public result is:

```text
COMBINED_LOCKED_V11
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
This repository currently reports the v11 inference-time regulator and its frozen regression ledger.

The public claim is limited to the supplied test suites and reference manifolds included in the project:

MBT-5 v11 blocked labelled hallucinated candidates and preserved labelled valid candidates across the frozen EXP20 ledger.
The project is released for inspection, reproduction, and independent testing.

Output Artifacts

The frozen v11 ledger is represented by:

mbt5_exp20_master_candidate_ledger.csv
mbt5_exp20_master_case_ledger.csv
mbt5_exp20_summary_metrics.csv
mbt5_exp20_case_summary.csv
mbt5_exp20_clamp_counts.csv
mbt5_exp20_failure_table.csv
mbt5_exp20_patch_lineage.csv
License

See LICENSE.


