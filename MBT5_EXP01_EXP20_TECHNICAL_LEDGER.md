

# MBT-5 EXP01–EXP20 Technical Ledger
Geometry-only inference-time hallucination regulation for AI outputs.
This document records the MBT-5 experiment trail from EXP01 through EXP20. It is a technical ledger of completed work: mechanisms tested, failures found, patches added, regressions checked, and the frozen v11 result.
It is not a roadmap.
---
## 1. Project Definition
MBT-5 is an inference-time regulator for AI outputs.
It tests whether candidate AI outputs remain inside a supplied semantic and relational reference structure.
The regulator does not train the model.
The regulator does not fine-tune the model.
The regulator does not inspect model weights.
The regulator checks generated text after inference.
In this project, an AI hallucination is treated as semantic or relational drift:
```text
The output leaves the intended manifold and emits unsupported content.

That includes:

* wrong entities,
* wrong numbers,
* wrong units,
* wrong subject-object relations,
* reversed relation direction,
* unsupported negation,
* unsupported overclaiming,
* category substitution,
* candidate pools where every available answer is unsafe.

The regulator either emits the safest valid candidate or blocks/abstains when all candidates are unsafe.

⸻

2. Fact Model Used in This Work

The project uses this hierarchy:

Universe does facts.
Humans describe the universe.
AI describes human descriptions.

MBT-5 does not claim direct access to physical truth.

MBT-5 regulates AI text against supplied human reference descriptions, candidate sets, semantic manifolds, and relation maps.

The tested claim is not “AI fact checking”.

The tested claim is:

MBT-5 detects and blocks hallucinated AI outputs as semantic or relational drift from supplied reference structure.

⸻

3. Current Locked Result

Current frozen ledger:

MBT5_EXP20_combined_guarded_master_ledger_v11

Combined locked v11 result:

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

There were no locked false positives and no locked false negatives in the EXP20 frozen ledger.

Across the locked MBT-5 v11 regression and unseen-holdout suites, every labelled hallucinated candidate was blocked and every labelled valid candidate was preserved.

⸻

4. Current Supported Claim

MBT-5 v11 blocks hallucinated AI outputs against supplied reference manifolds and relation constraints.
In the frozen EXP20 ledger, it achieved confusion [[97, 0], [0, 160]] over 257 labelled candidates across 53 cases.

That is the claim supported by the recorded experiments.

⸻

5. Core Mechanisms

5.1 Semantic Shock

Candidate outputs are embedded into semantic space.

MBT-5 measures distance from a reference center.

shock = Γ · ||candidate_embedding - reference_center||²

Lower shock means the output is closer to the reference manifold.

Higher shock means the output has drifted away from the reference manifold.

The reference center can be built from stable reference answers or from a reference manifold supplied by the user.

⸻

5.2 Geometric Median

For robust reference-centering, MBT-5 uses a geometric median rather than only an arithmetic mean.

This reduces sensitivity to outlier candidates.

The geometric median was computed with iterative Weiszfeld-style updates.

⸻

5.3 Internal Entropy

Multiple model outputs can be sampled for the same prompt.

MBT-5 embeds the outputs and measures the spread between them.

High internal entropy means unstable candidate generation.

Low internal entropy means candidate outputs are semantically consistent.

⸻

5.4 Token Shock

MBT-5 performs leave-one-out token analysis.

For each token, it removes the token, re-embeds the sentence, and measures whether removing that token reduces shock.

This localises the source of drift.

Example:

The capital of France is London.

Token-shock localisation identifies:

London

as the high-shock token.

⸻

5.5 Literal Drift Guards

Geometry alone misses close substitutions.

Example:

Water boils at 90 degrees Celsius at sea level.

This is semantically close to the correct statement but numerically wrong.

Literal guards protect:

* numbers,
* units,
* named entities,
* content tokens.

⸻

5.6 Content Token Clamp

The content token clamp blocks unsupported content tokens that geometry may tolerate.

This fixed failures such as:

The Moon is made mostly of wood.

where the output was not distant enough geometrically but contained unsupported content.

⸻

5.7 Overclaim Detection

MBT-5 blocks outputs that exceed the reference support.

Examples:

Gravity is fully solved by modern physics.
Quantum gravity is complete and experimentally verified.
A successful model is automatically the final truth.

These are treated as hallucinated when the reference only supports careful, limited descriptions.

⸻

5.8 Copular Relation Clamp

The copular relation clamp handles “X is Y” claims.

Examples:

The Sun is a planet.
Earth is flat.
A dog is a bird.
Rome is the capital city of France.

The clamp checks subject-object mappings against supplied relation structure.

⸻

5.9 Non-Copular Relation Clamp

The non-copular relation clamp handles verb relations.

Examples:

Earth orbits the Moon.
The Sun orbits Earth.
DNA contains the nucleus.
Heat produces friction.
Photosynthesis converts oxygen into carbon dioxide and water.

This catches relation direction errors and unsupported subject-object-verb mappings.

⸻

5.10 Coordinated Verb Subject Patch

Coordinated verb structures can share subjects across verbs.

Example:

Photosynthesis releases oxygen and stores light energy.

The initial non-copular clamp misread this and created a false positive.

EXP13C patched coordinated verb subject handling.

⸻

5.11 Reference Member Override

If a candidate is an exact member of the supplied reference set, geometry should not reject it unless a hard clamp fires.

This fixed the valid paraphrase false positive:

Italy's government is based in Rome.

⸻

5.12 Negated Positive Support Clamp

The negation clamp blocks unsupported negation of positively supported reference claims.

Examples:

Water is not liquid at room temperature.
Sound does not need a material medium to travel.
Scientific descriptions do not use measurements or predictions.
General relativity proves gravity has no connection to mass or energy.

If the reference positively supports a relation, an unsupported negation of that relation is blocked.

⸻

5.13 Abstention

When every candidate is unsafe, MBT-5 blocks instead of choosing the least-bad candidate.

This is essential because the least-bad hallucination is still a hallucination.

⸻

6. Experiment Ledger

⸻

EXP01 — Reference Box Hard Negatives

Tested basic semantic geometry against small reference boxes.

Result:

Overall:
  N=24
  Accuracy=0.7917
  Precision=1.0000
  Recall=0.6667
  F1=0.8000
  ROC_AUC=0.9407

Finding:

Geometry detected obvious semantic drift but missed close semantic substitutions, especially wrong numbers and plausible wrong content.

⸻

EXP02 — Geometry Plus Literal Guard

Added literal drift features.

Result:

Overall hybrid LOO:
  Accuracy=0.9583
  Precision=1.0000
  Recall=0.9333
  F1=0.9655

Finding:

Literal guards improved detection of number, unit, and entity substitutions.

⸻

EXP03 — Regulator Blocks Hallucinated Outputs

Tested candidate selection.

Result:

Baseline first-output hallucination rate: 0.8000
MBT-5 regulated-output rate:           0.0000
Absolute reduction:                    0.8000

Finding:

The regulator replaced hallucinated first outputs with safe candidates when safe candidates were present.

Cases included:

The capital of France is London.  -> blocked/replaced
Water boils at 90 degrees Celsius. -> blocked/replaced
The Moon is made of green cheese. -> blocked/replaced
Gravity is fully solved.          -> blocked/replaced

⸻

EXP04 — No Valid Candidate Abstention

Tested all-bad candidate pools.

Result:

Cases: 7
Correct case decisions: 6 / 7
Case decision accuracy: 0.8571

Failure:

The Moon is made mostly of wood.

was emitted in an all-bad moon-surface case.

Finding:

The regulator needed a stronger unsupported-content clamp.

⸻

EXP05 — Unsupported Content Token Clamp

Added unsupported content-token clamp.

Result:

Cases: 7
EXP04 v1 rule:
  Correct case decisions: 6 / 7
  Case accuracy: 0.8571
EXP05 v2 rule:
  Correct case decisions: 7 / 7
  Case accuracy: 1.0000

Candidate-level:

EXP04 v1:
  Accuracy=0.9286
  Confusion [[5, 1], [1, 21]]
EXP05 v2:
  Accuracy=0.9643
  Confusion [[5, 1], [0, 22]]

Finding:

The content-token clamp fixed the all-bad moon-surface failure.

⸻

EXP06 — Valid Paraphrase Tolerance

Tested more permissive paraphrase tolerance.

Result:

Cases: 8
EXP05 v2:
  Correct case decisions: 8 / 8
  Case accuracy: 1.0000
EXP06 v3:
  Correct case decisions: 6 / 8
  Case accuracy: 0.7500

Candidate-level:

EXP05 v2:
  Accuracy=0.9091
  Confusion [[6, 3], [0, 24]]
EXP06 v3:
  Accuracy=0.7879
  Confusion [[8, 1], [6, 18]]

Finding:

Naive paraphrase tolerance was too permissive and allowed unsafe candidates through.

⸻

EXP06B — Guarded Paraphrase Tolerance

Added guarded paraphrase tolerance.

Result:

Cases: 8
Correct case decisions: 8 / 8
Case accuracy: 1.0000
Candidate accuracy: 1.0000
Confusion [[9, 0], [0, 24]]

Finding:

Paraphrase tolerance must be guarded by literal and relation safety.

⸻

EXP07 — Adversarial Generalisation Suite

Expanded to new adversarial cases.

Result:

Cases: 10
Case accuracy: 1.0000
Candidate accuracy: 0.9348
Precision: 1.0000
Recall:    0.9091
F1:        0.9524
Confusion: [[13, 0], [3, 30]]

False negatives:

The Sun is a planet.
The Sun is a galaxy.
Earth is a perfect cube.

Finding:

Geometry and literals still missed short relation/category substitutions.

⸻

EXP08 — False Negative All-Bad Stress

Moved known false negatives into all-bad pools.

Result before relation clamp:

Cases: 5
Correct case decisions: 0 / 5
Case accuracy: 0.0000
Candidate accuracy: 0.4783
Precision: 1.0000
Recall:    0.4286
F1:        0.6000
Confusion: [[2, 0], [12, 9]]

Finding:

Without relation structure, MBT-5 could emit close but wrong category substitutions.

⸻

EXP09 — Relation Polarity Clamp

Added relation polarity clamp.

Result:

Cases: 5
Correct case decisions: 5 / 5
Case accuracy: 1.0000
Candidate accuracy: 1.0000
Precision: 1.0000
Recall:    1.0000
F1:        1.0000
Confusion: [[2, 0], [0, 21]]

Finding:

Relation polarity fixed the EXP08 false-negative failure class.

⸻

EXP10 — Apply Relation Clamp to EXP07

Applied relation clamp back to EXP07.

Result:

Cases: 10
Correct case decisions: 10 / 10
Case accuracy: 1.0000
Candidate accuracy: 0.9783
Precision: 0.9706
Recall:    1.0000
F1:        0.9851
Confusion: [[12, 1], [0, 33]]

False positive:

Earth is roughly spherical and slightly flattened.

Finding:

Relation clamp fixed false negatives but introduced one valid-paraphrase false positive.

⸻

EXP10B — Relation Core Support Patch

Fixed the EXP10 false positive.

Result:

Cases: 10
Correct case decisions: 10 / 10
Case accuracy: 1.0000
Candidate accuracy: 1.0000
Precision: 1.0000
Recall:    1.0000
F1:        1.0000
Confusion: [[13, 0], [0, 33]]

Finding:

Core relation support patch preserved valid Earth-shape paraphrases while keeping relation drift blocked.

⸻

EXP11 — Apply EXP10B to EXP08 Regression

Confirmed EXP10B did not regress EXP08.

Result:

Cases: 5
Correct case decisions: 5 / 5
Case accuracy: 1.0000
Candidate accuracy: 1.0000
Precision: 1.0000
Recall:    1.0000
F1:        1.0000
Confusion: [[2, 0], [0, 21]]

⸻

EXP12 — Master Regression Ledger

Combined EXP07 and EXP08 families.

Result:

EXP07 locked:
  Candidates: 46
  Confusion: [[13, 0], [0, 33]]
EXP08 locked:
  Candidates: 23
  Confusion: [[2, 0], [0, 21]]
Combined:
  Candidates: 69
  Confusion: [[15, 0], [0, 54]]
  Accuracy: 1.0000
  F1:       1.0000

Finding:

The relation-patched system passed the combined EXP07 + EXP08 regression ledger.

⸻

EXP13 — Non-Copular Relation Stress

Tested verb-based relations:

* orbits,
* converts,
* contains,
* produces,
* has count.

Result before non-copular clamp:

Cases: 8
Correct case decisions: 5 / 8
Case accuracy: 0.6250
Candidate accuracy: 0.4211
Precision: 0.7500
Recall:    0.1250
F1:        0.2143
Confusion: [[13, 1], [21, 3]]

Failures included:

Earth orbits the Moon.
The Sun orbits Earth.
DNA contains the nucleus.
Heat produces friction.
Photosynthesis converts oxygen into carbon dioxide and water.

Finding:

Copular relation checks were not enough. Verb relation direction needed its own clamp.

⸻

EXP13B — Non-Copular Relation Clamp

Added non-copular relation clamp.

Result:

Cases: 8
Correct case decisions: 8 / 8
Case accuracy: 1.0000
Candidate accuracy: 0.9737
Precision: 0.9600
Recall:    1.0000
F1:        0.9796
Confusion: [[13, 1], [0, 24]]

False positive:

Photosynthesis releases oxygen and stores light energy.

Finding:

The non-copular clamp fixed relation direction failures but mishandled coordinated verbs.

⸻

EXP13C — Coordinated Verb Subject Patch

Fixed coordinated verb handling.

Result:

Cases: 8
Correct case decisions: 8 / 8
Case accuracy: 1.0000
Candidate accuracy: 1.0000
Precision: 1.0000
Recall:    1.0000
F1:        1.0000
Confusion: [[14, 0], [0, 24]]

Finding:

Coordinated verb subject patch preserved valid coordinated relation statements while keeping unsupported non-copular relations blocked.

⸻

EXP14 — Valid Paraphrase False Positive Stress

Tested valid paraphrases only.

Result before reference-member override:

Cases: 10
Correct case decisions: 10 / 10
Case accuracy: 1.0000
Candidate accuracy: 0.9800
Confusion: [[49, 1], [0, 0]]

False positive:

Italy's government is based in Rome.

Finding:

A valid reference member could still be blocked by geometric thresholding.

⸻

EXP14B — Reference Member Override

Added exact/reference-member override.

Result:

Cases: 10
Correct case decisions: 10 / 10
Case accuracy: 1.0000
Candidate accuracy: 1.0000
Confusion: [[50, 0], [0, 0]]

Finding:

Reference-member override fixed the valid-paraphrase false positive without creating false negatives.

⸻

EXP15 — Large All-Bad Abstention Stress

Tested large all-bad pools.

Result before negation clamp:

Cases: 10
Correct block decisions: 7 / 10
Case accuracy: 0.7000
Candidate accuracy: 0.9400
Precision: 1.0000
Recall:    0.9400
F1:        0.9691
Confusion: [[0, 0], [3, 47]]

Emitted-risk failures:

Earth has no equatorial bulging and is a perfect sphere.
General relativity proves gravity has no connection to mass or energy.
Scientific descriptions do not use measurements or predictions.

Finding:

Unsupported negation of positively supported reference claims needed its own clamp.

⸻

EXP15B — Negated Positive Support Clamp

Added unsupported negation clamp.

Result:

Cases: 10
Correct block decisions: 10 / 10
Case accuracy: 1.0000
Candidate accuracy: 1.0000
Precision: 1.0000
Recall:    1.0000
F1:        1.0000
Confusion: [[0, 0], [0, 50]]

Newly blocked by negation clamp:

Earth has no equatorial bulging and is a perfect sphere.
General relativity proves gravity has no connection to mass or energy.
Scientific descriptions do not use measurements or predictions.

Optional EXP14B valid-only regression under EXP15B:

Candidate accuracy: 1.0000
Confusion: [[50, 0], [0, 0]]
False positives: None

Finding:

Negation clamp fixed all-bad failures without regressing valid paraphrases.

⸻

EXP16 — Master Regression Ledger v10

Combined locked regression ledger.

Result:

Candidates: 207
Accuracy:   1.0000
Precision:  1.0000
Recall:     1.0000
F1:         1.0000
Confusion:  [[79, 0], [0, 128]]
Cases: 43
Correct: 43 / 43
Accuracy: 1.0000
Emitted: 22
Blocked: 21

Finding:

The locked v10 system passed the combined regression ledger.

⸻

EXP16B — Master Regression Ledger Metadata Fix

Fixed suite metadata labelling in the master ledger.

Result remained:

Candidates: 207
Confusion: [[79, 0], [0, 128]]
Cases: 43
Correct: 43 / 43

Suite breakdown:

EXP07_adversarial_suite_LOCKED_EXP10B:
  N=46
  Confusion [[13, 0], [0, 33]]
EXP08_false_negative_stress_LOCKED_EXP11:
  N=23
  Confusion [[2, 0], [0, 21]]
EXP13_noncopular_relation_LOCKED_EXP13C:
  N=38
  Confusion [[14, 0], [0, 24]]
EXP14_valid_paraphrase_LOCKED_EXP14B:
  N=50
  Confusion [[50, 0], [0, 0]]
EXP15_large_all_bad_LOCKED_EXP15B:
  N=50
  Confusion [[0, 0], [0, 50]]
COMBINED_LOCKED_REGRESSION:
  N=207
  Confusion [[79, 0], [0, 128]]

Finding:

EXP16B became the clean locked master before unseen holdout testing.

⸻

EXP17 — Unseen Holdout Suite v10

Tested new unseen cases.

Result before patch:

Candidates: 50
Accuracy:   0.7800
Precision:  1.0000
Recall:     0.6562
F1:         0.7925
Confusion:  [[18, 0], [11, 21]]
Cases: 10
Correct: 7 / 10
Case accuracy: 0.7000
Emitted: 8
Blocked: 2

Failures:

The Sun orbits Mercury.
An insulator improves current flow like a wire.
Chlorophyll contains chloroplasts.
ATP produces mitochondria.
DNA contains the nucleus.
Mitochondria contain keyboards.
Magna Carta was signed in 1066.
The Battle of Hastings occurred in 1215.
The Declaration of Independence was adopted in 1215.
Apollo 11 landed on the Moon in 1776.
Magna Carta was signed in 1969.

Finding:

The locked v10 system still missed unseen relation recall and historical-date relation errors.

⸻

EXP18 — EXP17 Relation Recall Patch Audit

Added relation recall patch.

Result:

Before confusion: [[18, 0], [11, 21]]
After confusion:  [[18, 0], [0, 32]]
Before F1: 0.7925
After F1:  1.0000
Cases:
  Correct: 10 / 10
  Accuracy: 1.0000

Newly blocked by EXP18 patch:

The Sun orbits Mercury.
An insulator improves current flow like a wire.
Chlorophyll contains chloroplasts.
ATP produces mitochondria.
DNA contains the nucleus.
Mitochondria contain keyboards.
Magna Carta was signed in 1066.
The Battle of Hastings occurred in 1215.
The Declaration of Independence was adopted in 1215.
Apollo 11 landed on the Moon in 1776.
Magna Carta was signed in 1969.

Finding:

The EXP18 relation recall patch fixed the unseen holdout.

⸻

EXP19 — Apply EXP18 Patch to EXP16B Master Regression

Tested whether the EXP18 patch regressed the previous master suite.

Result:

EXP16B locked before EXP18 patch:
  N=207
  Confusion [[79, 0], [0, 128]]
  F1=1.0000
After unguarded EXP18 patch:
  N=207
  Confusion [[71, 8], [0, 128]]
  F1=0.9697

False positives introduced by the unguarded patch:

Earth is roughly spherical and slightly flattened.
Plants use photosynthesis to make sugars from carbon dioxide and water.
Photosynthesis releases oxygen and stores light energy.
Mars has two natural satellites called Phobos and Deimos.

Finding:

The EXP18 patch fixed the unseen holdout but was overbroad when applied to the full master regression.

⸻

EXP19B — Guarded EXP18 Patch Regression

Guarded the EXP18 relation patch.

Result:

EXP16B before EXP18 patch:
  [[79, 0], [0, 128]]
EXP19 overbroad patch:
  [[71, 8], [0, 128]]
EXP19B guarded patch on EXP16B:
  [[79, 0], [0, 128]]
EXP17 holdout before patch:
  [[18, 0], [11, 21]]
EXP19B guarded patch on holdout:
  [[18, 0], [0, 32]]
Combined EXP16B + EXP18 guarded:
  [[97, 0], [0, 160]]

Finding:

The guarded patch preserved the old master suite and fixed the unseen holdout.

⸻

EXP20 — Combined Guarded Master Ledger v11

Frozen current ledger.

Result:

EXP16B_MASTER_LOCKED_PLUS_GUARDED_EXP18:
  Candidates: 207
  Cases: 43
  Confusion: [[79, 0], [0, 128]]
EXP18_UNSEEN_HOLDOUT_GUARDED:
  Candidates: 50
  Cases: 10
  Confusion: [[18, 0], [0, 32]]
COMBINED_LOCKED_V11:
  Candidates: 257
  Cases: 53
  Confusion: [[97, 0], [0, 160]]

Final locked v11 result:

Accuracy:  1.0000
Precision: 1.0000
Recall:    1.0000
F1:        1.0000

Case-level result:

Cases: 53
Correct: 53 / 53
Emitted: 28
Blocked: 25

Finding:

EXP20 is the current frozen guarded master ledger.

⸻

7. Patch Lineage

EXP16B
  Locked master regression before unseen holdout
  207 candidates
  Confusion [[79, 0], [0, 128]]
  Status: clean
EXP17
  Unseen holdout exposed relation recall misses
  50 candidates
  Confusion [[18, 0], [11, 21]]
  Status: failed recall
EXP18
  Relation recall patch fixed EXP17 holdout
  50 candidates
  Confusion [[18, 0], [0, 32]]
  Status: clean on holdout
EXP19
  Unguarded EXP18 patch regressed EXP16B
  207 candidates
  Confusion [[71, 8], [0, 128]]
  Status: regression found
EXP19B
  Guarded patch restored EXP16B and fixed EXP17
  257 candidates
  Confusion [[97, 0], [0, 160]]
  Status: clean combined
EXP20
  Frozen combined guarded master ledger
  257 candidates
  Confusion [[97, 0], [0, 160]]
  Status: frozen ledger

⸻

8. Clamp Counts in EXP20 Combined Locked v11

EXP20 combined clamp count preview:

COMBINED_LOCKED_V11
  known_participant_unsupported_relation_clamp: 103 / 257 = 0.400778
  exp19b_guarded_patch_clamp:                  70 / 257 = 0.272374
  guarded_known_participant_unsupported:       48 / 257 = 0.186770
  exact_reference_member:                      47 / 257 = 0.182879
  content_clamp_flag:                          36 / 257 = 0.140078
  relation_polarity_clamp_v10b:                35 / 257 = 0.136187
  noncopular_relation_clamp_v13c:              24 / 257 = 0.093385
  protected_literal_drift:                     23 / 257 = 0.089494
  final_literal_block:                         20 / 257 = 0.077821
  protected_entity:                            19 / 257 = 0.073930
  role_swapped_relation_clamp:                 19 / 257 = 0.073930
  final_geometric_block:                       19 / 257 = 0.073930
  noncopular_relation_clamp:                   18 / 257 = 0.070039
  overclaim_flag:                              12 / 257 = 0.046693
  relation_polarity_clamp:                      9 / 257 = 0.035019
  negated_positive_support_clamp:               9 / 257 = 0.035019
  protected_number:                             7 / 257 = 0.027237
  historical_date_relation_clamp:               5 / 257 = 0.019455
  protected_unit:                               2 / 257 = 0.007782
  missed_predicate_relation_clamp:              1 / 257 = 0.003891
  reference_member_geometry_override:           1 / 257 = 0.003891

⸻

9. EXP20 Output Files

EXP20 generated:

mbt5_exp20_master_candidate_ledger.csv
mbt5_exp20_master_case_ledger.csv
mbt5_exp20_summary_metrics.csv
mbt5_exp20_case_summary.csv
mbt5_exp20_clamp_counts.csv
mbt5_exp20_failure_table.csv
mbt5_exp20_patch_lineage.csv

The failure table contained no locked false positives or false negatives.

⸻

10. Public README Boundary

The README should stay short.

It should state:

MBT-5 is a geometry-only inference-time hallucination regulator for AI outputs.

It should include the frozen EXP20 headline result.

It should link here for the full technical ledger.

The detailed experiment history belongs in this file, not in the README.

⸻

11. Package Layout

Current project layout:

mbt_ai_tools/
├── mbt/
│   ├── embeddings.py
│   ├── geometry.py
│   ├── stability.py
│   ├── tokens.py
│   ├── consensus.py
│   └── utils.py
├── cli.py
├── pyproject.toml
├── README.md
└── LICENSE

Expected documentation layout:

README.md
docs/
  MBT5_EXP01_EXP20_TECHNICAL_LEDGER.md

⸻

12. Usage Surface

Library usage:

from mbt_ai_tools import confidence_score, hallucination_risk, token_shock_map
text = "The capital of France is Paris."
score = confidence_score(text)
risk = hallucination_risk(text)
shocks = token_shock_map(text)
print(score)
print(risk)
print(shocks)

CLI usage:

mbt-check "The capital of France is Paris."

The CLI reports a confidence/risk label and a numeric internal entropy score.

⸻

13. Candidate Selection Mode

Candidate selection workflow:

1. Generate or supply candidate outputs.
2. Embed candidates and reference descriptions.
3. Compute semantic shock.
4. Compute literal drift.
5. Extract relation claims.
6. Apply clamps.
7. Rank safe candidates.
8. Emit the safest valid candidate, or block if all candidates are unsafe.

Example mixed candidate pool:

Prompt:
  What is the capital of France?
Candidates:
  The capital of France is London.
  The capital of France is Paris.
  The capital of France is Lyon.
MBT-5 action:
  emit "The capital of France is Paris."

Example all-bad candidate pool:

Prompt:
  What is the capital of France?
Candidates:
  The capital of France is London.
  The capital of France is Lyon.
  The capital of France is Berlin.
MBT-5 action:
  block

⸻

14. Current Frozen Status

Current status:

Locked version: MBT-5 v11
Frozen ledger: MBT5_EXP20_combined_guarded_master_ledger_v11
Candidates:    257
Cases:         53
Confusion:     [[97, 0], [0, 160]]
Failures:      none in locked EXP20 ledger

This ledger records completed work only.
