# Markdown Audit Report Demo

This example turns a batch JSONL evaluation into a human-readable audit report.

```bash
manifold-check \
  --input-jsonl examples/batch_input.jsonl \
  --no-embeddings \
  --format markdown \
  --output audit.md
```

Guard-mode audit command:

```bash
manifold-check \
  --input-jsonl examples/batch_input.jsonl \
  --no-embeddings \
  --format markdown \
  --fail-on-block
```

The guard command exits with status `2` when any input row blocks.

Expected output:

```markdown
# ManifoldGuard Audit Report

## Summary

- Total cases: 2
- Emitted: 1
- Blocked: 1
- Safe candidate evaluations: 1
- Blocked candidate evaluations: 2

## Case: france-capital

- Action: emit
- Emitted index: 1
- Emitted text: The capital of France is Paris.
- Input line: 1

### Candidate Evaluations

#### Candidate 0 - blocked

- Text: The capital of France is London.
- Score: 0.2250
- Shock: 0.0000
- Literal score: 1.5000
- Clamps: protected_entity, protected_literal_drift, final_literal_block, content_clamp_flag, known_participant_unsupported_relation_clamp, guarded_known_participant_unsupported_relation_clamp, exp19b_guarded_patch_clamp
- Selection: blocked_before_ranking
- Selection rank: `null`
- Selection reason: Candidate was dropped from emission ranking because it failed a hard guard or geometry check.
- Relations: capital of france / is / london; france / capital / london
- Negated relations: none

#### Candidate 1 - safe

- Text: The capital of France is Paris.
- Score: 0.0000
- Shock: 0.0000
- Literal score: 0.0000
- Clamps: exact_reference_member
- Selection: emitted
- Selection rank: 1
- Selection reason: Candidate was selected as the lowest-score safe candidate in the pool.
- Relations: capital of france / is / paris; france / capital / paris
- Negated relations: none


## Case: unsupported-negation

- Action: block
- Emitted index: `null`
- Emitted text: `null`
- Input line: 2

### Candidate Evaluations

#### Candidate 0 - blocked

- Text: Water is not liquid at room temperature.
- Score: 0.0000
- Shock: 0.0000
- Literal score: 0.0000
- Clamps: negated_positive_support_clamp
- Selection: blocked_before_ranking
- Selection rank: `null`
- Selection reason: Candidate was dropped from emission ranking because it failed a hard guard or geometry check.
- Relations: none
- Negated relations: water / is / liquid at room temperature
```
