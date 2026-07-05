# ManifoldGuard CLI Quickstart

This short walkthrough uses the offline path only. It does not require
`sentence-transformers` or any embedding model downloads.

## Install

```bash
python -m pip install manifold-guard
```

For local development from a checkout:

```bash
python -m pip install -e . --no-deps
```

## Single check with an explanation

Use `--why` when you want ManifoldGuard to explain the decision.

```bash
manifold-check \
  --reference "Water is liquid at room temperature." \
  --candidate "Water is not liquid at room temperature." \
  --no-embeddings \
  --why
```

Expected shape:

```text
BLOCK | no safe candidate
[0] blocked | score=... | clamps=...
    explain | Blocked because these guards fired: ...
    reason | negated_positive_support_clamp | Candidate negates a relation that is positively supported by the references.
```

## Tiny demo corpus

The public demo corpus is intentionally small:

- `safe_emit_reference_match`: exact supported reference match.
- `blocked_negation`: unsupported negation drift.
- `blocked_numeric_drift`: unsupported numeric drift.

Run it as a Markdown audit:

```bash
manifold-check \
  --input-jsonl examples/demo_corpus.jsonl \
  --no-embeddings \
  --format markdown \
  --why
```

Expected case-level actions:

```text
safe_emit_reference_match -> emit
blocked_negation -> block
blocked_numeric_drift -> block
```

Use `--fail-on-block` when this command is part of CI and any blocked case
should produce exit status `2`.

## When to use batch mode

Use `--input-jsonl` when you want a repeatable audit over many references and
candidates. Each non-empty line should be one JSON object with:

- `references`: string or list of strings.
- `candidates`: string or list of strings.
- `id`: optional human-readable case identifier.

For larger project checks, use `examples/regression_corpus.jsonl` with
`manifold-eval`.
