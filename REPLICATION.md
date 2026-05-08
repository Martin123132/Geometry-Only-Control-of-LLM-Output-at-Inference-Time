# MBT-5 Replication Log

This log records independent or fresh-environment checks of the public MBT-5 package and regression corpus.

## 2026-05-08 Colab Offline Corpus Replication

Environment:

```text
Google Colab fresh runtime
Input artifact: Geometry-Only-Control-of-LLM-Output-at-Inference-Time-main.zip
Mode tested: offline literal / relation / negation clamp path
Embeddings: disabled with --no-embeddings
```

Procedure:

```text
1. Uploaded the original repository zip into a fresh Colab runtime.
2. Installed test/runtime dependencies.
3. Installed the package from the extracted repository.
4. Applied or confirmed the relation-parser patch for capital-city paraphrases.
5. Generated the deterministic public regression corpus.
6. Ran pytest against the expanded offline corpus.
7. Ran a CLI smoke test for an Earth/Sun role-swapped relation.
```

Observed result:

```text
pytest:
5 passed
return code: 0

CLI smoke test:
EMIT | Earth orbits the Sun. | score=0.0000
[0] blocked | score=0.0000 | clamps=role_swapped_relation_clamp|known_participant_unsupported_relation_clamp|guarded_known_participant_unsupported_relation_clamp|exp19b_guarded_patch_clamp
[1] safe | score=0.0000 | clamps=exact_reference_member
cli return code: 0
```

Status:

```text
PASS
```

Scope note:

This replication validates the offline MBT-5 clamp path over the public regression corpus. It does not validate the optional semantic embedding path.

## 2026-05-08 Local Offline Corpus Expansion

Environment:

```text
Windows local workspace
Python runtime: uv-managed CPython
Mode tested: offline literal / relation / negation clamp path
Embeddings: disabled with use_embeddings=False
```

Procedure:

```text
1. Expanded the deterministic corpus generator from 124 to 220 cases.
2. Added direct regression coverage for multi-word capital extraction.
3. Regenerated examples/regression_corpus.jsonl.
4. Ran the package test suite.
```

Observed result:

```text
uv run python examples/build_regression_corpus.py
wrote 220 cases

uv run --with pytest python -m pytest -q
6 passed
```

Status:

```text
PASS
```
