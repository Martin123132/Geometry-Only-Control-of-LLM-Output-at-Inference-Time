# ManifoldGuard Report Schema

The release support contract for CLI modes, install modes, examples, and CI gates is captured in `docs/product_readiness_manifest.json`.

Machine-readable schema file:

- `docs/report_schema.json`

`manifold-check --format json` emits one JSON object for a single regulation run.
`manifold-check --input-jsonl ...` emits one JSON object per input line.
`manifold-check --demo` runs the built-in offline demo corpus without requiring an examples checkout.
`--summary` appends a final summary object in batch JSONL mode.
`--format markdown` emits a human-readable single report or batch audit report.
`--format csv` emits one spreadsheet-friendly row per candidate evaluation.
`--explain` adds optional per-candidate decision explanations to text, JSON, and Markdown reports.
`--why` is a friendly alias for `--explain`.

## Single Report

Top-level fields:

- `action`: `emit` or `block`.
- `emitted_text`: emitted candidate text, or `null` when blocked.
- `emitted_index`: zero-based emitted candidate index, or `null` when blocked.
- `emitted_score`: emitted candidate regulator score, or `null` when blocked.
- `evaluations`: per-candidate evaluation objects.

Each evaluation contains:

- `index`: zero-based candidate index.
- `text`: candidate text.
- `status`: `safe` or `blocked`.
- `safe_to_emit`: boolean emit eligibility.
- `pred_hallucinated`: boolean regulator prediction.
- `regulator_score`: final ranking score.
- `mbt5_shock`: semantic geometry shock score.
- `threshold`: active geometry threshold.
- `literal_score`: literal drift score.
- `clamps`: clamp names applied to the candidate.
- `relations`: extracted candidate relation tuples.
- `negated_relations`: extracted negated relation tuples.
- `exact_reference_member`: whether candidate exactly matches a reference after normalization.
- `explanation`: optional decision explanation object when `--explain` is used.
- `token_shock`: optional embedding-backed token shock entries when `--token-shock` is used.

## Batch Input JSONL

Each non-empty line must be a JSON object with:

- `references`: string or list of strings.
- `candidates`: string or list of strings.
- `id`: optional caller-provided identifier copied into the output line.

Singular aliases `reference` and `candidate` are accepted for one-item inputs.

## Batch Output JSONL

Batch output uses the same report fields as single JSON output and adds:

- `line`: one-based input line number.
- `id`: copied from the input when present.
- `references`: normalized list of reference strings used for that line.

## Batch Summary

When `--summary` is used with `--input-jsonl`, the final JSONL record is:

- `record_type`: always `summary`.
- `total`: number of processed input rows.
- `emitted`: number of rows that emitted a safe candidate.
- `blocked`: number of rows that blocked.
- `safe_candidates`: total safe candidate evaluations across all rows.
- `blocked_candidates`: total blocked candidate evaluations across all rows.

The machine-readable schema in `docs/report_schema.json` validates these objects for tooling integrations.

## Exit Codes

- `0`: command completed normally.
- `2`: `--fail-on-block` was used and a single run blocked, or at least one batch row blocked.

## Markdown Output

`--format markdown` is available for single regulation reports and batch JSONL audit reports.

Single regulation Markdown includes:

- report title.
- action, emitted index, and emitted text.
- per-candidate status, score, shock, literal score, clamps, relations, and negated relations.
- optional per-candidate explanation summaries when `--explain` is used.

Batch Markdown includes:

- audit title.
- summary totals.
- one case section per input line.
- the same per-candidate details as single Markdown reports.

## CSV Output

`--format csv` is available for single regulation reports and batch JSONL audit reports.

CSV output contains one row per candidate evaluation with these columns:

- `case_id`: copied input `id` when present.
- `line`: one-based input line number in batch mode.
- `references`: batch references joined with ` || `.
- `action`: case-level `emit` or `block`.
- `emitted_index`: case-level emitted candidate index, blank when blocked.
- `emitted_text`: case-level emitted candidate text, blank when blocked.
- `candidate_index`: zero-based candidate index.
- `candidate_text`: candidate text.
- `status`: `safe` or `blocked`.
- `safe_to_emit`: lowercase boolean.
- `pred_hallucinated`: lowercase boolean.
- `regulator_score`: final ranking score.
- `mbt5_shock`: semantic geometry shock score.
- `threshold`: active geometry threshold.
- `literal_score`: literal drift score.
- `clamps`: clamp names joined with `; `.
- `relations`: relation tuples joined with `; `.
- `negated_relations`: negated relation tuples joined with `; `.
- `exact_reference_member`: lowercase boolean.
- `token_shock`: optional embedding-backed token shock entries joined as `token:score`.

CSV keeps the stable candidate-row columns. Use JSON or Markdown with `--explain`
when machine-readable or human-readable decision explanations are required.

Example machine-validated payloads:

- `examples/single_report_example.json`
- `examples/batch_report_example.jsonl`

Human-readable explanation example:

- `examples/explain_report.md`
