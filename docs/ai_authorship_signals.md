# Exploratory Authorship-Signal Research

Status: research-only. This is not a production AI detector and is not part of
the v0.1.8 release claim surface.

## Purpose

This experiment asks a deliberately narrower question than "who wrote this
text?" It maps a text to a compact offline structural profile, places that
profile against two labelled prototype groups with ManifoldGuard geometry, and
returns one of three outcomes:

- human-like structural signal
- AI-assistance-like structural signal
- uncertain

The signal is not a probability and is not proof of authorship. It describes
similarity to the labelled corpus used for that specific run.

## Why the boundary matters

The same final text can result from a person writing, a model generating, a
person editing a model draft, a model editing a person draft, or copying from
another source. A text-only system that did not observe the writing event cannot
resolve all of those histories. It must therefore be allowed to abstain.

Research on detector robustness links the best achievable detector performance
to the distance between human and model text distributions and demonstrates
that paraphrasing can reduce detector reliability:

- https://arxiv.org/abs/2303.11156
- https://openai.com/index/new-ai-classifier-for-indicating-ai-written-text/
- https://www.nist.gov/publications/reducing-risks-posed-synthetic-content-overview-technical-approaches-digital-content

Do not use this experiment as the sole basis for an accusation, disciplinary
action, employment decision, academic decision, or claim that a person used AI.

## What is implemented

The evaluator is deterministic and offline. Its profile uses:

- mean sentence length and sentence-length variation
- lexical diversity and repeated-bigram rate
- function-word ratio and long-token ratio
- punctuation ratio and mean token length

Training examples are standardised, reduced to robust geometric prototype
centres, and compared using ManifoldGuard squared-distance geometry. A small
distance difference produces uncertain rather than a forced label.

The evaluator does not require sentence-transformers, a model download, a
network call, or a hosted API.

## Included fixture corpus

examples/ai_authorship_fixture_corpus.jsonl exists only to exercise the corpus
format, report structure, abstention path, and test suite. Every row has
provenance_status set to fixture. The human and AI labels in that file are
synthetic class buckets, not records of a real writing event.

Because of that status, the evaluator deliberately suppresses any agreement
metric for the shipped fixture. A high fixture score would have no evidential
meaning and must not be presented as detection performance.

## Running the fixture

Keep generated reports outside the repository:

    python -B scripts/evaluate_authorship_signals.py \
      --corpus examples/ai_authorship_fixture_corpus.jsonl \
      --output D:\Temp\ManifoldGuard\authorship-fixture-report.json

For machine-readable output:

    python -B scripts/evaluate_authorship_signals.py \
      --corpus examples/ai_authorship_fixture_corpus.jsonl \
      --format json \
      --output D:\Temp\ManifoldGuard\authorship-fixture-report.json

The fixture report must say Evidence status: fixture_or_mixed. It is a pipeline
check, not a benchmark result.

## Verified research corpus contract

A real experiment needs records with explicit, reviewable provenance:

- id: stable non-identifying record id
- text: consented or lawfully usable text
- label: human, ai_raw, ai_edited, or ai_paraphrased
- split: train or eval
- provenance_status: verified
- source_notes: a concise non-sensitive explanation of how the writing route was
  recorded

Use the evaluator with --require-verified to reject a fixture or mixed corpus:

    python -B scripts/evaluate_authorship_signals.py \
      --corpus D:\Research\verified-authorship-corpus.jsonl \
      --require-verified \
      --output D:\Temp\ManifoldGuard\verified-authorship-report.json

The report intentionally omits raw corpus text and source notes. Do not put
personal, private, or sensitive writing into a corpus without a clear lawful
basis and the writer's informed permission.

## Before any comparison claim

Do not compare this experiment with commercial detector products until all of
the following are true:

- the corpus contains independently verified human originals and logged model
  outputs
- prompts, model identifiers, edits, genres, languages, and collection rules
  are recorded
- train and evaluation material are separated by genre and source where
  possible
- edited, paraphrased, translated, and adversarially rewritten cases are
  included
- coverage, abstention, false-positive risk, and failures are reported together
- an external holdout corpus has been tested

The best near-term product use is an auditable detector stress-test harness or
provenance-uncertainty evaluator, not a binary authorship verdict.
