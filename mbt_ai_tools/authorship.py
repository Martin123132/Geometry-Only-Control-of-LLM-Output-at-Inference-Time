"""Research-only authorship-signal evaluation utilities.

This module intentionally does not claim to determine who wrote a text. It
maps text to a small, auditable structural profile and compares that profile
with labelled prototype groups using ManifoldGuard geometry. The output is a
similarity signal with an explicit abstention state, suitable for controlled
research corpora only.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .mbt.geometry import geometric_median, shock

FEATURE_NAMES = (
    "mean_sentence_tokens",
    "sentence_length_variation",
    "unique_token_ratio",
    "repeated_bigram_ratio",
    "function_word_ratio",
    "long_token_ratio",
    "punctuation_ratio",
    "mean_token_length",
)

AI_ASSISTED_LABELS = frozenset({"ai_raw", "ai_edited", "ai_paraphrased"})
SUPPORTED_LABELS = frozenset({"human", *AI_ASSISTED_LABELS})
SUPPORTED_SPLITS = frozenset({"train", "eval"})
SUPPORTED_PROVENANCE_STATUSES = frozenset({"fixture", "verified"})

_WORD_PATTERN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
_PUNCTUATION = frozenset(".,;:!?")
_FUNCTION_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "for",
        "from",
        "has",
        "have",
        "if",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "with",
    }
)


class AuthorshipCorpusError(ValueError):
    """Raised when a research corpus is incomplete or has unsafe provenance."""


def label_group(label: str) -> str:
    """Map a detailed construction label to the two prototype groups."""

    if label == "human":
        return "human"
    if label in AI_ASSISTED_LABELS:
        return "ai_assisted"
    raise AuthorshipCorpusError(
        f"Unsupported label {label!r}; expected one of {sorted(SUPPORTED_LABELS)}."
    )


def text_profile(text: str) -> dict[str, float]:
    """Return a compact deterministic structural profile for one non-empty text."""

    tokens = [match.group(0).lower() for match in _WORD_PATTERN.finditer(text)]
    if not tokens:
        raise ValueError("Text must contain at least one alphabetic token.")

    sentences = [part.strip() for part in _SENTENCE_BOUNDARY.split(text) if part.strip()]
    sentence_lengths = [
        len(_WORD_PATTERN.findall(sentence)) for sentence in sentences
    ] or [len(tokens)]
    mean_sentence_tokens = float(np.mean(sentence_lengths))
    sentence_length_variation = float(
        np.std(sentence_lengths) / mean_sentence_tokens
        if mean_sentence_tokens
        else 0.0
    )

    bigrams = list(zip(tokens, tokens[1:]))
    bigram_counts = Counter(bigrams)
    repeated_bigram_ratio = (
        sum(count - 1 for count in bigram_counts.values() if count > 1) / len(bigrams)
        if bigrams
        else 0.0
    )
    visible_characters = [character for character in text if not character.isspace()]
    punctuation_ratio = (
        sum(character in _PUNCTUATION for character in visible_characters)
        / len(visible_characters)
        if visible_characters
        else 0.0
    )

    return {
        "mean_sentence_tokens": mean_sentence_tokens,
        "sentence_length_variation": sentence_length_variation,
        "unique_token_ratio": len(set(tokens)) / len(tokens),
        "repeated_bigram_ratio": float(repeated_bigram_ratio),
        "function_word_ratio": sum(token in _FUNCTION_WORDS for token in tokens)
        / len(tokens),
        "long_token_ratio": sum(len(token) >= 7 for token in tokens) / len(tokens),
        "punctuation_ratio": float(punctuation_ratio),
        "mean_token_length": float(np.mean([len(token) for token in tokens])),
    }


def _profile_vector(profile: dict[str, float]) -> np.ndarray:
    return np.asarray([profile[name] for name in FEATURE_NAMES], dtype=float)


def _required_string(record: dict[str, Any], key: str, line_number: int) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise AuthorshipCorpusError(
            f"Corpus line {line_number} requires a non-empty string field {key!r}."
        )
    return value.strip()


def _validate_record(record: dict[str, Any], line_number: int) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise AuthorshipCorpusError(f"Corpus line {line_number} must be a JSON object.")

    identifier = _required_string(record, "id", line_number)
    text = _required_string(record, "text", line_number)
    label = _required_string(record, "label", line_number)
    split = _required_string(record, "split", line_number)
    provenance_status = _required_string(record, "provenance_status", line_number)
    source_notes = _required_string(record, "source_notes", line_number)

    if label not in SUPPORTED_LABELS:
        raise AuthorshipCorpusError(
            f"Corpus line {line_number} has unsupported label {label!r}; "
            f"expected one of {sorted(SUPPORTED_LABELS)}."
        )
    if split not in SUPPORTED_SPLITS:
        raise AuthorshipCorpusError(
            f"Corpus line {line_number} has unsupported split {split!r}; "
            f"expected one of {sorted(SUPPORTED_SPLITS)}."
        )
    if provenance_status not in SUPPORTED_PROVENANCE_STATUSES:
        raise AuthorshipCorpusError(
            f"Corpus line {line_number} has unsupported provenance_status "
            f"{provenance_status!r}; expected one of "
            f"{sorted(SUPPORTED_PROVENANCE_STATUSES)}."
        )

    return {
        "id": identifier,
        "text": text,
        "label": label,
        "split": split,
        "provenance_status": provenance_status,
        "source_notes": source_notes,
    }


def read_authorship_corpus(path: Path) -> list[dict[str, Any]]:
    """Read a source-labelled JSONL research corpus with explicit provenance."""

    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip():
            continue
        try:
            raw_record = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise AuthorshipCorpusError(
                f"Corpus line {line_number} is not valid JSON: {exc.msg}."
            ) from exc
        record = _validate_record(raw_record, line_number)
        if record["id"] in seen_ids:
            raise AuthorshipCorpusError(
                f"Corpus contains duplicate id {record['id']!r}."
            )
        seen_ids.add(record["id"])
        records.append(record)

    if not records:
        raise AuthorshipCorpusError(f"Corpus {path} did not contain any records.")
    return records


@dataclass(frozen=True)
class AuthorshipSignalModel:
    """Robust profile prototypes plus the abstention boundary used for scoring."""

    feature_means: tuple[float, ...]
    feature_scales: tuple[float, ...]
    human_center: tuple[float, ...]
    ai_assisted_center: tuple[float, ...]
    min_separation: float

    @classmethod
    def fit(
        cls,
        records: Iterable[dict[str, Any]],
        *,
        min_separation: float = 0.15,
    ) -> "AuthorshipSignalModel":
        """Fit two geometry prototypes from the explicitly labelled train split."""

        if not 0.0 < min_separation < 1.0:
            raise ValueError("min_separation must be greater than 0 and less than 1.")

        training_records = [record for record in records if record["split"] == "train"]
        grouped = {
            "human": [
                record
                for record in training_records
                if label_group(record["label"]) == "human"
            ],
            "ai_assisted": [
                record
                for record in training_records
                if label_group(record["label"]) == "ai_assisted"
            ],
        }
        for group, samples in grouped.items():
            if len(samples) < 2:
                raise AuthorshipCorpusError(
                    f"Training split needs at least two {group!r} records."
                )

        raw_vectors = np.asarray(
            [_profile_vector(text_profile(record["text"])) for record in training_records],
            dtype=float,
        )
        means = np.mean(raw_vectors, axis=0)
        scales = np.std(raw_vectors, axis=0)
        scales = np.where(scales < 1e-9, 1.0, scales)

        def normalized_vectors(samples: Sequence[dict[str, Any]]) -> np.ndarray:
            return np.asarray(
                [
                    (_profile_vector(text_profile(sample["text"])) - means) / scales
                    for sample in samples
                ],
                dtype=float,
            )

        human_center = geometric_median(normalized_vectors(grouped["human"]))
        ai_assisted_center = geometric_median(normalized_vectors(grouped["ai_assisted"]))
        return cls(
            feature_means=tuple(float(value) for value in means),
            feature_scales=tuple(float(value) for value in scales),
            human_center=tuple(float(value) for value in human_center),
            ai_assisted_center=tuple(float(value) for value in ai_assisted_center),
            min_separation=min_separation,
        )

    def evaluate(self, text: str) -> dict[str, Any]:
        """Return a profile similarity signal and an explicit abstention outcome."""

        profile = text_profile(text)
        normalized = (
            _profile_vector(profile)
            - np.asarray(self.feature_means, dtype=float)
        ) / np.asarray(self.feature_scales, dtype=float)
        human_distance = shock(normalized, np.asarray(self.human_center, dtype=float))
        ai_assisted_distance = shock(
            normalized,
            np.asarray(self.ai_assisted_center, dtype=float),
        )
        denominator = max(human_distance + ai_assisted_distance, 1e-12)
        ai_assistance_signal = (human_distance - ai_assisted_distance) / denominator
        separation = abs(ai_assistance_signal)

        if separation < self.min_separation:
            classification = "uncertain"
        elif ai_assistance_signal > 0:
            classification = "ai_assistance_like"
        else:
            classification = "human_like"

        return {
            "classification": classification,
            "ai_assistance_signal": float(ai_assistance_signal),
            "separation": float(separation),
            "distances": {
                "human": float(human_distance),
                "ai_assisted": float(ai_assisted_distance),
            },
            "features": profile,
        }


def evaluate_authorship_corpus(
    path: Path,
    *,
    min_separation: float = 0.15,
    require_verified: bool = False,
) -> dict[str, Any]:
    """Evaluate a labelled research corpus without making authorship claims."""

    records = read_authorship_corpus(path)
    provenance_counts = Counter(record["provenance_status"] for record in records)
    fully_verified = set(provenance_counts) == {"verified"}
    if require_verified and not fully_verified:
        raise AuthorshipCorpusError(
            "A verified corpus is required, but this corpus includes fixture records."
        )

    model = AuthorshipSignalModel.fit(records, min_separation=min_separation)
    evaluation_records = [record for record in records if record["split"] == "eval"]
    if not evaluation_records:
        raise AuthorshipCorpusError("Corpus needs at least one eval record.")

    cases: list[dict[str, Any]] = []
    classification_counts: Counter[str] = Counter()
    non_abstained_matches = 0
    non_abstained_total = 0
    for record in evaluation_records:
        prediction = model.evaluate(record["text"])
        source_group = label_group(record["label"])
        expected_bucket = (
            "human_like" if source_group == "human" else "ai_assistance_like"
        )
        classification = prediction["classification"]
        classification_counts[classification] += 1
        if classification != "uncertain":
            non_abstained_total += 1
            non_abstained_matches += int(classification == expected_bucket)
        cases.append(
            {
                "id": record["id"],
                "source_label": record["label"],
                "source_group": source_group,
                "provenance_status": record["provenance_status"],
                "expected_bucket": expected_bucket,
                "prediction": prediction,
            }
        )

    summary: dict[str, Any] = {
        "total_records": len(records),
        "training_records": len(records) - len(evaluation_records),
        "evaluation_records": len(evaluation_records),
        "provenance_counts": dict(sorted(provenance_counts.items())),
        "label_counts": dict(
            sorted(Counter(record["label"] for record in records).items())
        ),
        "classification_counts": dict(sorted(classification_counts.items())),
        "classified_count": non_abstained_total,
        "abstained_count": len(evaluation_records) - non_abstained_total,
        "coverage": non_abstained_total / len(evaluation_records),
        "non_abstained_agreement": (
            non_abstained_matches / non_abstained_total
            if fully_verified and non_abstained_total
            else None
        ),
    }

    return {
        "report_version": "0.1",
        "mode": "research_only",
        "evidence_status": (
            "verified_research_corpus" if fully_verified else "fixture_or_mixed"
        ),
        "claim_boundary": (
            "This is a structural similarity signal with abstention. It does not "
            "determine whether a person or model authored a text."
        ),
        "corpus": str(path),
        "feature_names": list(FEATURE_NAMES),
        "min_separation": min_separation,
        "summary": summary,
        "cases": cases,
    }


def format_authorship_report(report: dict[str, Any]) -> str:
    """Format a compact operator-facing summary without exposing corpus text."""

    summary = report["summary"]
    lines = [
        "ManifoldGuard exploratory authorship-signal report",
        f"Evidence status: {report['evidence_status']}",
        f"Records: {summary['total_records']}",
        f"Training records: {summary['training_records']}",
        f"Evaluation records: {summary['evaluation_records']}",
        f"Classified: {summary['classified_count']}",
        f"Abstained: {summary['abstained_count']}",
        f"Coverage: {summary['coverage']:.3f}",
        f"Claim boundary: {report['claim_boundary']}",
    ]
    if report["evidence_status"] == "fixture_or_mixed":
        lines.append(
            "No agreement metric is reported because fixture or mixed provenance "
            "is not evidence of authorship detection."
        )
    elif summary["non_abstained_agreement"] is not None:
        lines.append(
            "Non-abstained agreement: "
            f"{summary['non_abstained_agreement']:.3f}"
        )
    return "\n".join(lines) + "\n"
