"""MBT-5 v11 guarded candidate-selection tools.

This module implements the public, package-level version of the mechanisms
recorded in the EXP01-EXP20 technical ledger: semantic shock, literal drift
protection, relation clamps, guarded relation-recall patches, reference-member
geometry override, negation/overclaim checks, and abstention when every
candidate is unsafe.

The implementation is intentionally transparent and reference-bound.  It does
not fact-check against an external knowledge base; it only regulates candidates
against the reference text and optional relation tuples supplied by the caller.
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import numpy as np

from .embeddings import embed_texts
from .geometry import geometric_median, shock

THRESHOLD_K = 3.0
THRESHOLD_FLOOR = 0.02
LITERAL_SCORE_WEIGHT = 0.15
LITERAL_BLOCK_THRESHOLD = 1.0

_UNIT_WORDS = {
    "celsius",
    "fahrenheit",
    "kelvin",
    "degree",
    "degrees",
    "meter",
    "meters",
    "metre",
    "metres",
    "kilometer",
    "kilometers",
    "mile",
    "miles",
    "second",
    "seconds",
    "minute",
    "minutes",
    "hour",
    "hours",
    "year",
    "years",
    "kg",
    "g",
    "cm",
    "mm",
    "km",
    "mph",
    "m/s",
    "percent",
    "%",
}

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "based",
    "be",
    "but",
    "by",
    "called",
    "can",
    "city",
    "commonly",
    "described",
    "does",
    "from",
    "has",
    "have",
    "in",
    "into",
    "is",
    "it",
    "its",
    "like",
    "mostly",
    "natural",
    "not",
    "of",
    "on",
    "or",
    "roughly",
    "slightly",
    "the",
    "to",
    "use",
    "with",
    "without",
}

_NEGATORS = {"not", "no", "never", "doesnt", "don't", "doesn't", "cannot", "can't"}
_OVERCLAIM_PATTERNS = (
    "fully solved",
    "complete and experimentally verified",
    "final truth",
    "automatically the final",
    "proves gravity has no connection",
    "proves there is no connection",
)

Predicate = str
Relation = Tuple[str, Predicate, str]


@dataclass(frozen=True)
class LiteralDrift:
    """Literal differences between a candidate and its reference manifold."""

    novel_numbers: Tuple[str, ...] = ()
    novel_units: Tuple[str, ...] = ()
    novel_entities: Tuple[str, ...] = ()
    novel_content: Tuple[str, ...] = ()

    @property
    def score(self) -> float:
        return float(
            len(self.novel_numbers)
            + len(self.novel_units)
            + len(self.novel_entities)
            + 0.5 * len(self.novel_content)
        )

    @property
    def has_protected_drift(self) -> bool:
        return bool(self.novel_numbers or self.novel_units or self.novel_entities)


@dataclass
class CandidateEvaluation:
    """Full MBT-5 v11 decision record for one candidate."""

    text: str
    pred_hallucinated: bool
    safe_to_emit: bool
    regulator_score: float
    mbt5_shock: float
    threshold: float
    literal_score: float
    clamp_summary: Tuple[str, ...] = ()
    relations: Tuple[Relation, ...] = ()
    literal_drift: LiteralDrift = field(default_factory=LiteralDrift)
    exact_reference_member: bool = False


@dataclass
class RegulationResult:
    """Result of regulating a candidate pool."""

    action: str
    emitted_text: Optional[str]
    emitted: Optional[CandidateEvaluation]
    evaluations: List[CandidateEvaluation]


@dataclass
class ReferenceManifold:
    """Reference text, derived literals, relation map, and geometry anchors."""

    references: List[str]
    relations: Set[Relation] = field(default_factory=set)
    center: Optional[np.ndarray] = None
    threshold: float = THRESHOLD_FLOOR
    normalized_reference_members: Set[str] = field(default_factory=set)
    numbers: Set[str] = field(default_factory=set)
    units: Set[str] = field(default_factory=set)
    entities: Set[str] = field(default_factory=set)
    content_tokens: Set[str] = field(default_factory=set)

    @classmethod
    def from_texts(
        cls,
        references: Iterable[str],
        *,
        relations: Optional[Iterable[Relation]] = None,
        threshold_k: float = THRESHOLD_K,
        threshold_floor: float = THRESHOLD_FLOOR,
        use_embeddings: bool = True,
    ) -> "ReferenceManifold":
        refs = [r.strip() for r in references if r and r.strip()]
        supplied_relations = {_normalize_relation(r) for r in (relations or [])}
        extracted_relations = (
            set().union(*(extract_relations(r) for r in refs)) if refs else set()
        )
        relation_set = supplied_relations | extracted_relations

        center: Optional[np.ndarray] = None
        threshold = threshold_floor
        if use_embeddings and refs:
            embeddings = embed_texts(refs)
            center = embeddings[0] if len(embeddings) == 1 else geometric_median(embeddings)
            distances = np.asarray([shock(e, center) for e in embeddings])
            median = float(np.median(distances))
            mad = float(np.median(np.abs(distances - median)))
            threshold = max(threshold_floor, median + threshold_k * mad)

        joined = "\n".join(refs)
        return cls(
            references=refs,
            relations=relation_set,
            center=center,
            threshold=threshold,
            normalized_reference_members={normalize_text(r) for r in refs},
            numbers=set(_numbers(joined)),
            units=set(_units(joined)),
            entities=set(_entities(joined)),
            content_tokens=set(_content_tokens(joined)),
        )


def normalize_text(text: str) -> str:
    """Normalize text for exact reference-member comparisons."""

    return " ".join(text.lower().translate(str.maketrans("", "", string.punctuation)).split())


def extract_relations(text: str) -> Set[Relation]:
    """Extract simple copular and non-copular relation claims from text.

    The parser is deliberately small and auditable.  It targets the relation
    shapes recorded in EXP09-EXP20: capitals/classifications, orbits,
    contains, produces/releases/stores/converts, has-count, and historical-date
    assertions such as signed/occurred/adopted/landed in a year.
    """

    normalized = normalize_text(text)
    relations: Set[Relation] = set()

    # Capital relations: "capital of France is Paris", "France's capital is Paris".
    m = re.search(r"capital of ([a-z][a-z ]+?) is ([a-z][a-z ]+?)(?:$|\b)", normalized)
    if m:
        relations.add((_clean_span(m.group(1)), "capital", _clean_span(m.group(2))))
    m = re.search(r"([a-z][a-z ]+?)s capital (?:city )?is ([a-z][a-z ]+?)(?:$|\b)", normalized)
    if m:
        relations.add((_clean_span(m.group(1)), "capital", _clean_span(m.group(2))))
    m = re.search(r"([a-z][a-z ]+?) is (?:the )?capital (?:city )?of ([a-z][a-z ]+?)(?:$|\b)", normalized)
    if m:
        relations.add((_clean_span(m.group(2)), "capital", _clean_span(m.group(1))))

    # Copular/classification relations.
    copular_pattern = (
        r"\b([a-z][a-z ]+?) (?:is|are) (?:a |an |the )?"
        r"([a-z][a-z ]+?)(?:\.|$| and | but )"
    )
    for m in re.finditer(copular_pattern, normalized):
        subj, obj = _clean_span(m.group(1)), _clean_span(m.group(2))
        if (
            subj
            and obj
            and "capital of" not in subj
            and len(subj.split()) <= 5
            and len(obj.split()) <= 7
        ):
            relations.add((subj, "is", obj))

    # Active verb relations, with coordinated shared-subject repair.
    verbs = (
        "orbits|contains|contain|produces|produce|releases|release|stores|store|"
        "converts|convert|improves|improve|uses|use|needs|need"
    )
    pattern = rf"\b([a-z][a-z ]+?) ({verbs}) ([a-z][a-z0-9 ]+?)(?=\.|$| and (?:{verbs}) )"
    for m in re.finditer(pattern, normalized):
        subj = _clean_span(m.group(1))
        pred = _lemma_predicate(m.group(2))
        obj = _clean_span(m.group(3))
        if subj and obj and subj not in {"and", "or"}:
            relations.add((subj, pred, obj))

    coord = rf"\b([a-z][a-z ]+?) ({verbs}) ([a-z][a-z0-9 ]+?) and ({verbs}) ([a-z][a-z0-9 ]+?)(?:\.|$)"
    for m in re.finditer(coord, normalized):
        subj = _clean_span(m.group(1))
        relations.add((subj, _lemma_predicate(m.group(2)), _clean_span(m.group(3))))
        relations.add((subj, _lemma_predicate(m.group(4)), _clean_span(m.group(5))))

    # Counts: "Mars has two moons".
    for m in re.finditer(
        r"\b([a-z][a-z ]+?) has ([a-z0-9]+) ([a-z][a-z ]+?)(?:\.|,|$)",
        normalized,
    ):
        relations.add(
            (_clean_span(m.group(1)), f"has_count:{m.group(3).strip()}", _clean_span(m.group(2)))
        )

    # Historical/date relations.
    for m in re.finditer(
        r"\b([a-z][a-z ]+?) "
        r"(was signed|occurred|was adopted|landed [a-z ]*moon) in (\d{3,4})",
        normalized,
    ):
        relations.add((_clean_span(m.group(1)), _lemma_predicate(m.group(2)), m.group(3)))

    return {_normalize_relation(r) for r in relations if r[0] and r[2]}


def evaluate_candidate(
    candidate: str,
    reference: Union[ReferenceManifold, Iterable[str]],
    *,
    use_embeddings: bool = True,
) -> CandidateEvaluation:
    """Evaluate one candidate against a supplied reference manifold."""

    manifold = (
        reference
        if isinstance(reference, ReferenceManifold)
        else ReferenceManifold.from_texts(reference, use_embeddings=use_embeddings)
    )
    exact_member = normalize_text(candidate) in manifold.normalized_reference_members
    candidate_relations = extract_relations(candidate)
    literal = literal_drift(candidate, manifold)

    mbt5_shock = 0.0
    if use_embeddings and manifold.center is not None:
        mbt5_shock = shock(embed_texts([candidate])[0], manifold.center)

    clamps: List[str] = []
    if literal.novel_numbers:
        clamps.append("protected_number")
    if literal.novel_units:
        clamps.append("protected_unit")
    if literal.novel_entities:
        clamps.append("protected_entity")
    if literal.has_protected_drift:
        clamps.append("protected_literal_drift")
    if literal.score >= LITERAL_BLOCK_THRESHOLD:
        clamps.append("final_literal_block")
    if _has_unsupported_content(candidate, manifold, literal):
        clamps.append("content_clamp_flag")
    if _has_overclaim(candidate):
        clamps.append("overclaim_flag")
    if _has_unsupported_negation(candidate, manifold):
        clamps.append("negated_positive_support_clamp")

    relation_clamps = _relation_clamps(candidate_relations, manifold.relations)
    clamps.extend(relation_clamps)

    geometric_block = mbt5_shock > manifold.threshold
    if geometric_block:
        clamps.append("final_geometric_block")

    if exact_member:
        clamps.append("exact_reference_member")
        if geometric_block and not _hard_clamps_without_geometry(clamps):
            clamps.remove("final_geometric_block")
            clamps.append("reference_member_geometry_override")
            geometric_block = False

    hard_block = _hard_clamps_without_geometry(clamps)
    pred_hallucinated = hard_block or geometric_block
    regulator_score = mbt5_shock + LITERAL_SCORE_WEIGHT * literal.score

    return CandidateEvaluation(
        text=candidate,
        pred_hallucinated=pred_hallucinated,
        safe_to_emit=not pred_hallucinated,
        regulator_score=regulator_score,
        mbt5_shock=mbt5_shock,
        threshold=manifold.threshold,
        literal_score=literal.score,
        clamp_summary=tuple(dict.fromkeys(clamps)) or ("none",),
        relations=tuple(sorted(candidate_relations)),
        literal_drift=literal,
        exact_reference_member=exact_member,
    )


def regulate_candidates(
    candidates: Sequence[str],
    references: Iterable[str],
    *,
    relations: Optional[Iterable[Relation]] = None,
    use_embeddings: bool = True,
) -> RegulationResult:
    """Emit the safest supported candidate, or block if every candidate is unsafe."""

    manifold = ReferenceManifold.from_texts(
        references, relations=relations, use_embeddings=use_embeddings
    )
    evaluations = [
        evaluate_candidate(c, manifold, use_embeddings=use_embeddings) for c in candidates
    ]
    safe = [e for e in evaluations if e.safe_to_emit]
    if not safe:
        return RegulationResult("block", None, None, evaluations)
    emitted = min(safe, key=lambda e: e.regulator_score)
    return RegulationResult("emit", emitted.text, emitted, evaluations)


def literal_drift(candidate: str, manifold: ReferenceManifold) -> LiteralDrift:
    """Compute number/unit/entity/content novelty against reference text."""

    cand_numbers = set(_numbers(candidate))
    cand_units = set(_units(candidate))
    cand_entities = set(_entities(candidate))
    cand_content = set(_content_tokens(candidate))
    return LiteralDrift(
        novel_numbers=tuple(sorted(cand_numbers - manifold.numbers)),
        novel_units=tuple(sorted(cand_units - manifold.units)),
        novel_entities=tuple(sorted(cand_entities - manifold.entities)),
        novel_content=tuple(sorted(cand_content - manifold.content_tokens)),
    )


def _relation_clamps(
    candidate_relations: Set[Relation], reference_relations: Set[Relation]
) -> List[str]:
    clamps: List[str] = []
    if not candidate_relations or not reference_relations:
        return clamps

    ref_subjects = {r[0] for r in reference_relations}
    ref_objects = {r[2] for r in reference_relations}
    ref_participants = ref_subjects | ref_objects
    ref_predicates = {r[1] for r in reference_relations}

    for rel in candidate_relations:
        if rel in reference_relations:
            continue
        subj, pred, obj = rel
        if (obj, pred, subj) in reference_relations:
            clamps.extend(
                ["role_swapped_relation_clamp", "known_participant_unsupported_relation_clamp"]
            )
        elif pred in ref_predicates and (subj in ref_participants or obj in ref_participants):
            clamps.append("known_participant_unsupported_relation_clamp")
        elif subj in ref_participants and pred not in ref_predicates:
            clamps.extend(
                ["missed_predicate_relation_clamp", "known_participant_unsupported_relation_clamp"]
            )
        if pred.startswith(("was signed", "occurred", "was adopted", "landed")) or obj.isdigit():
            matching_events = [
                r for r in reference_relations if r[0] == subj and r[1] == pred
            ]
            if matching_events and rel not in reference_relations:
                clamps.append("historical_date_relation_clamp")

    if clamps:
        clamps.extend(
            ["guarded_known_participant_unsupported_relation_clamp", "exp19b_guarded_patch_clamp"]
        )
    return clamps


def _hard_clamps_without_geometry(clamps: Sequence[str]) -> bool:
    non_hard = {
        "none",
        "final_geometric_block",
        "exact_reference_member",
        "reference_member_geometry_override",
    }
    return any(c not in non_hard for c in clamps)


def _has_overclaim(text: str) -> bool:
    normalized = normalize_text(text)
    return any(pattern in normalized for pattern in _OVERCLAIM_PATTERNS)


def _has_unsupported_negation(text: str, manifold: ReferenceManifold) -> bool:
    tokens = set(normalize_text(text).split())
    if not (tokens & _NEGATORS):
        return False
    reference_tokens = set(normalize_text(" ".join(manifold.references)).split())
    return not (reference_tokens & _NEGATORS)


def _has_unsupported_content(candidate: str, manifold: ReferenceManifold, literal: LiteralDrift) -> bool:
    if not literal.novel_content:
        return False
    # Keep the content clamp guarded to avoid blocking ordinary paraphrase words.
    return len(literal.novel_content) >= 3 or bool(literal.has_protected_drift)


def _numbers(text: str) -> List[str]:
    return [n.lower() for n in re.findall(r"\b\d+(?:\.\d+)?\b|\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|ninety|hundred|thousand)\b", text.lower())]


def _units(text: str) -> List[str]:
    tokens = normalize_text(text).split()
    return [t for t in tokens if t in _UNIT_WORDS]


def _entities(text: str) -> List[str]:
    entities = []
    for match in re.finditer(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text):
        value = _clean_span(match.group(0).lower())
        if value and value not in {"the", "a", "an"}:
            entities.append(value)
    return entities


def _content_tokens(text: str) -> List[str]:
    tokens = normalize_text(text).split()
    return [t for t in tokens if len(t) > 3 and t not in _STOPWORDS and not t.isdigit()]


def _clean_span(value: str) -> str:
    tokens = [t for t in value.strip().split() if t not in {"the", "a", "an"}]
    return " ".join(tokens[-5:]).strip()


def _lemma_predicate(predicate: str) -> str:
    p = predicate.strip().lower()
    mapping: Dict[str, str] = {
        "orbits": "orbit",
        "contains": "contain",
        "produces": "produce",
        "releases": "release",
        "stores": "store",
        "converts": "convert",
        "improves": "improve",
        "uses": "use",
        "needs": "need",
    }
    return mapping.get(p, p)


def _normalize_relation(relation: Relation) -> Relation:
    subj, pred, obj = relation
    return (_clean_span(normalize_text(subj)), _lemma_predicate(normalize_text(pred)), _clean_span(normalize_text(obj)))
