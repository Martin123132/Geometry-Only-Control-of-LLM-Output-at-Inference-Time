# CLI JSON Report Demo

This example runs the offline regulation path and emits a structured JSON report.

```bash
manifold-check \
  --reference "The capital of France is Paris." \
  --candidate "The capital of France is London." \
  --candidate "The capital of France is Paris." \
  --no-embeddings \
  --format json
```

Expected output:

```json
{
  "action": "emit",
  "emitted_index": 1,
  "emitted_score": 0.0,
  "emitted_text": "The capital of France is Paris.",
  "evaluations": [
    {
      "clamps": [
        "protected_entity",
        "protected_literal_drift",
        "final_literal_block",
        "content_clamp_flag",
        "known_participant_unsupported_relation_clamp",
        "guarded_known_participant_unsupported_relation_clamp",
        "exp19b_guarded_patch_clamp"
      ],
      "exact_reference_member": false,
      "index": 0,
      "literal_score": 1.5,
      "mbt5_shock": 0.0,
      "negated_relations": [],
      "pred_hallucinated": true,
      "regulator_score": 0.22499999999999998,
      "relations": [
        [
          "capital of france",
          "is",
          "london"
        ],
        [
          "france",
          "capital",
          "london"
        ]
      ],
      "safe_to_emit": false,
      "selection_rank": null,
      "selection_reason": "Candidate was dropped from emission ranking because it failed a hard guard or geometry check.",
      "selection_status": "blocked_before_ranking",
      "status": "blocked",
      "text": "The capital of France is London.",
      "threshold": 0.02
    },
    {
      "clamps": [
        "exact_reference_member"
      ],
      "exact_reference_member": true,
      "index": 1,
      "literal_score": 0.0,
      "mbt5_shock": 0.0,
      "negated_relations": [],
      "pred_hallucinated": false,
      "regulator_score": 0.0,
      "relations": [
        [
          "capital of france",
          "is",
          "paris"
        ],
        [
          "france",
          "capital",
          "paris"
        ]
      ],
      "safe_to_emit": true,
      "selection_rank": 1,
      "selection_reason": "Candidate was selected as the lowest-score safe candidate in the pool.",
      "selection_status": "emitted",
      "status": "safe",
      "text": "The capital of France is Paris.",
      "threshold": 0.02
    }
  ]
}
```
