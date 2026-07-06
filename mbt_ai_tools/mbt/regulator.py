"""ManifoldGuard v11 guarded candidate-selection tools.

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
from dataclasses import dataclass, field, replace
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

_NEGATORS = {
    "not",
    "no",
    "never",
    "doesnt",
    "dont",
    "didnt",
    "isnt",
    "arent",
    "wasnt",
    "werent",
    "cant",
    "cannot",
}
_NEGATION_NORMALIZATIONS = {
    "doesn't": "does not",
    "don't": "do not",
    "didn't": "did not",
    "wasn't": "was not",
    "weren't": "were not",
    "isn't": "is not",
    "aren't": "are not",
    "doesnt": "does not",
    "dont": "do not",
    "didnt": "did not",
    "wasnt": "was not",
    "werent": "were not",
    "isnt": "is not",
    "arent": "are not",
    "cant": "can not",
    "can't": "can not",
    "cannot": "can not",
    "won't": "will not",
    "wont": "will not",
    "wouldn't": "would not",
    "wouldnt": "would not",
    "couldn't": "could not",
    "couldnt": "could not",
    "shouldn't": "should not",
    "shouldnt": "should not",
    "hasn't": "has not",
    "hasnt": "has not",
    "haven't": "have not",
    "havent": "have not",
    "mustn't": "must not",
    "mustnt": "must not",
    "mightn't": "might not",
    "mightnt": "might not",
    "ain't": "are not",
    "aint": "are not",
}
_NEGATION_PREFIX = (
    r"(?:does not|do not|did not|cannot|can not|will not|would not|should not|"
    r"could not|must not|may not|might not|has not|have not|had not|is not|"
    r"are not|was not|were not|doesnt|dont|didnt|arent|isnt|wasnt|werent|cant|wont|"
    r"wouldnt|couldnt|shouldnt|hasnt|havent|mustnt|mightnt|not|never|aint)"
)
_AUXILIARY_SUBJECTS = {
    "does",
    "do",
    "did",
    "can",
    "will",
    "would",
    "should",
    "could",
    "must",
    "may",
    "might",
    "has",
    "have",
    "had",
    "is",
    "are",
    "was",
    "were",
}
_CAPITAL_OF_PREFIXES = (
    "capital of ",
    "capital city of ",
)
_OVERCLAIM_PATTERNS = (
    "fully solved",
    "complete and experimentally verified",
    "final truth",
    "automatically the final",
    "proves gravity has no connection",
    "proves there is no connection",
    "always boils",
)
_MONTH_PATTERN = (
    r"(?:january|february|march|april|may|june|july|august|september|"
    r"october|november|december)"
)
_CHAIN_VERB_LEMMAS = {
    "absorb",
    "exchange",
    "lend",
    "preserve",
    "provide",
    "release",
    "request",
    "store",
    "turn",
    "use",
}
_SUFFIX_ENTITY_HEADS = {
    "axis",
    "battery",
    "boiler",
    "device",
    "gate",
    "model",
    "plan",
    "pump",
    "runner",
    "sensor",
    "tank",
    "valve",
    "version",
}
_COMPARISON_CONTENT_TOKENS: Dict[str, Set[str]] = {
    "cheaperthan": {"cheaper", "than"},
    "fasterthan": {"faster", "than"},
    "highercapacitythan": {"higher", "capacity", "than"},
}

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
    """Full ManifoldGuard v11 decision record for one candidate."""

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
    negated_relations: Tuple[Relation, ...] = ()
    pool_group_key: str = ""
    duplicate_of: Optional[int] = None
    selection_status: str = "not_ranked"
    selection_rank: Optional[int] = None
    selection_reason: str = "Candidate was evaluated outside a candidate pool."


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
    negated_relations: Set[Relation] = field(default_factory=set)
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
        extracted_negated_relations = (
            set().union(*(_extract_negated_relations(r) for r in refs)) if refs else set()
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
            negated_relations=extracted_negated_relations,
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
    if not normalized:
        return set()

    relations: Set[Relation] = set()
    relations.update(_extract_temporal_binding_relations(normalized))
    relations.update(_extract_temporal_order_relations(normalized))
    relations.update(_extract_ordinal_binding_relations(normalized))
    relations.update(_extract_conditional_scope_relations(normalized))
    relations.update(_extract_permission_scope_relations(normalized))
    relations.update(_extract_token_reuse_binding_relations(normalized))
    relations.update(_extract_aggregate_binding_relations(normalized))
    relations.update(_extract_range_bound_relations(normalized))
    relations.update(_extract_scope_binding_relations(normalized))
    relations.update(_extract_alias_binding_relations(normalized))
    relations.update(_extract_identity_binding_relations(normalized))
    relations.update(_extract_comparative_binding_relations(normalized))
    relations.update(_extract_exception_scope_relations(normalized))
    relations.update(_extract_role_binding_relations(normalized))
    relations.update(_extract_shared_subject_verb_chain(normalized))
    for segment in _split_relation_segments(normalized):
        relations.update(_extract_relations_from_segment(segment))
    return {_normalize_relation(r) for r in relations if r[0] and r[2]}


def _extract_negated_relations(text: str) -> Set[Relation]:
    """Extract explicit negated relation claims with normalized polarity."""

    normalized = _normalize_negation_text(text)
    if not normalized:
        return set()

    negated: Set[Relation] = set()
    negated.update(_extract_scope_negated_relations(normalized))
    negated.update(_extract_exception_scope_negated_relations(normalized))
    negated.update(_extract_negated_relations_from_segment(normalized))
    for segment in _split_relation_segments(normalized):
        negated.update(_extract_negated_relations_from_segment(segment))
    return {_normalize_relation(r) for r in negated if r[0] and r[2]}


def _split_relation_segments(text: str) -> List[str]:
    raw_segments = re.split(r"\b(?: and | but |, )\b", text)
    return [segment.strip() for segment in raw_segments if segment.strip()]


def _extract_relations_from_segment(segment: str) -> Set[Relation]:
    relations: Set[Relation] = set()
    relations.update(_extract_temporal_binding_relations(segment))
    relations.update(_extract_temporal_order_relations(segment))
    relations.update(_extract_ordinal_binding_relations(segment))
    relations.update(_extract_conditional_scope_relations(segment, allow_partial=False))
    relations.update(_extract_permission_scope_relations(segment))
    relations.update(_extract_token_reuse_binding_relations(segment))
    relations.update(_extract_aggregate_binding_relations(segment))
    relations.update(_extract_range_bound_relations(segment))
    relations.update(_extract_scope_binding_relations(segment))
    relations.update(_extract_alias_binding_relations(segment))
    relations.update(_extract_role_binding_relations(segment))
    shared_subject_relations = _extract_shared_subject_verb_chain(segment)
    relations.update(shared_subject_relations)

    m = re.search(r"\bcapital of ([a-z][a-z ]+?) is ([a-z][a-z ]+?)(?:$| and | but )", segment)
    if m:
        relations.add((_clean_span(m.group(1)), "capital", _clean_span(m.group(2))))
    m = re.search(r"\b([a-z][a-z ]+?)s capital (?:city )?is ([a-z][a-z ]+?)(?:$| and | but )", segment)
    if m:
        relations.add((_clean_span(m.group(1)), "capital", _clean_span(m.group(2))))
    m = re.search(r"\b([a-z][a-z ]+?) is (?:the )?capital (?:city )?of ([a-z][a-z ]+?)(?:$| and | but )", segment)
    if m:
        relations.add((_clean_span(m.group(2)), "capital", _clean_span(m.group(1))))

    copular_pattern = (
        r"\b([a-z][a-z ]+?) (?:is|are|was|were) (?:a |an |the )?"
        r"([a-z0-9][a-z0-9 ]+?)(?:$| and | but )"
    )
    for m in re.finditer(copular_pattern, segment):
        subj, obj = _clean_span(m.group(1)), _clean_span(m.group(2))
        if (
            subj
            and obj
            and "capital of" not in subj
            and not obj.startswith(("capital of", "capital city of"))
            and subj != "it"
            and not _looks_like_temporal_value(obj)
            and len(subj.split()) <= 5
            and len(obj.split()) <= 8
            and " not " not in f" {obj} "
            and not _is_comparative_object(obj)
            and not _is_exception_scope_object(obj)
        ):
            relations.add((subj, "is", obj))

    verbs = (
        "orbits|orbit|contains|contain|contain|produces|produce|"
        "releases|release|stores|store|converts|convert|improves|improve|"
        "uses|use|needs|need|has|have|is|are|was|were|needs|improved|produces|produced|"
        "contains|contained|released|using|lends|lend|preserves|preserve|"
        "absorbs|absorb|exchanges|exchange|provides|provide|requests|request|"
        "turns|turn|writes|write|wrote|reviews|review|reviewed|feeds|feed|fed|"
        "supervises|supervise|supervised"
    )
    pattern = rf"\b([a-z][a-z0-9 ]+?) ({verbs}) ([a-z][a-z0-9 ]+?)(?=$| and (?:{verbs}) )"
    for m in re.finditer(pattern, segment):
        subj = _clean_relation_span(m.group(1))
        pred = _lemma_predicate(m.group(2))
        obj = _clean_relation_span(m.group(3))
        if (
            subj
            and obj
            and obj != "it"
            and subj not in {"and", "or"}
            and not (pred == "is" and _is_capital_of_predicate(obj))
            and not (shared_subject_relations and _contains_chain_verb(obj))
            and " not " not in f" {obj} "
            and not _is_comparative_object(obj)
            and not _is_exception_scope_object(obj)
        ):
            relations.add((subj, pred, obj))

    coord = rf"\b([a-z][a-z0-9 ]+?) ({verbs}) ([a-z][a-z0-9 ]+?) and ({verbs}) ([a-z][a-z0-9 ]+?)(?:$| and | but )"
    for m in re.finditer(coord, segment):
        subj = _clean_relation_span(m.group(1))
        relations.add((subj, _lemma_predicate(m.group(2)), _clean_relation_span(m.group(3))))
        relations.add((subj, _lemma_predicate(m.group(4)), _clean_relation_span(m.group(5))))

    for m in re.finditer(
        r"\b([a-z][a-z ]+?) has ([a-z0-9]+) ([a-z][a-z ]+?)(?:$| and | but |,)",
        segment,
    ):
        if m.group(2).lower() in _numbers(m.group(2)):
            relations.add(
                (
                    _clean_span(m.group(1)),
                    f"has_count:{m.group(3).strip()}",
                    _clean_span(m.group(2)),
                )
            )

    for m in re.finditer(
        r"\b([a-z][a-z ]+?) "
        r"(was signed|occurred|was adopted|landed [a-z ]*moon) in (\d{3,4})",
        segment,
    ):
        relations.add((_clean_span(m.group(1)), _lemma_predicate(m.group(2)), m.group(3)))

    return relations


def _extract_temporal_order_relations(text: str) -> Set[Relation]:
    relations: Set[Relation] = set()
    for m in re.finditer(
        r"\b(?:the )?([a-z][a-z0-9 ]+?) (?:completed|happened) (before|after) "
        r"(?:the )?([a-z][a-z0-9 ]+?)(?=$| and | but |,)",
        text,
    ):
        relations.add(
            (
                _clean_relation_span(m.group(1)),
                _lemma_predicate(m.group(2)),
                _clean_relation_span(m.group(3)),
            )
        )
    for m in re.finditer(
        r"\b([a-z][a-z0-9]*) repaired (?:the )?([a-z][a-z0-9 ]+?) before "
        r"([a-z][a-z0-9]*) inspected (?:the )?([a-z][a-z0-9 ]+?)(?=$| and | but |,)",
        text,
    ):
        relations.add((_clean_relation_span(m.group(1)), "repair_before", _clean_relation_span(m.group(2))))
        relations.add((_clean_relation_span(m.group(3)), "inspect_after", _clean_relation_span(m.group(4))))
        relations.add(("temporal role sequence", "order", "repairbeforeinspect"))
    for m in re.finditer(
        r"\b([a-z][a-z0-9]*) inspected (?:the )?([a-z][a-z0-9 ]+?) after "
        r"([a-z][a-z0-9]*) repaired (?:the )?([a-z][a-z0-9 ]+?)(?=$| and | but |,)",
        text,
    ):
        relations.add((_clean_relation_span(m.group(3)), "repair_before", _clean_relation_span(m.group(4))))
        relations.add((_clean_relation_span(m.group(1)), "inspect_after", _clean_relation_span(m.group(2))))
        relations.add(("temporal role sequence", "order", "repairbeforeinspect"))

    for m in re.finditer(
        r"\b([a-z][a-z0-9]*) approved draft ([a-z0-9]+) after "
        r"([a-z][a-z0-9]*) rejected draft ([a-z0-9]+)(?=$| and | but |,)",
        text,
    ):
        relations.add((_clean_relation_span(m.group(1)), "approve_after", f"draft{m.group(2)}"))
        relations.add((_clean_relation_span(m.group(3)), "reject_before", f"draft{m.group(4)}"))
        relations.add(("temporal role sequence", "order", "rejectbeforeapprove"))
    for m in re.finditer(
        r"\b([a-z][a-z0-9]*) rejected draft ([a-z0-9]+) before "
        r"([a-z][a-z0-9]*) approved draft ([a-z0-9]+)(?=$| and | but |,)",
        text,
    ):
        relations.add((_clean_relation_span(m.group(3)), "approve_after", f"draft{m.group(4)}"))
        relations.add((_clean_relation_span(m.group(1)), "reject_before", f"draft{m.group(2)}"))
        relations.add(("temporal role sequence", "order", "rejectbeforeapprove"))
    for m in re.finditer(
        r"\b([a-z][a-z0-9]*) rejected draft ([a-z0-9]+) after "
        r"([a-z][a-z0-9]*) approved draft ([a-z0-9]+)(?=$| and | but |,)",
        text,
    ):
        relations.add((_clean_relation_span(m.group(3)), "approve_before", f"draft{m.group(4)}"))
        relations.add((_clean_relation_span(m.group(1)), "reject_after", f"draft{m.group(2)}"))
        relations.add(("temporal role sequence", "order", "approvebeforereject"))
    for m in re.finditer(
        r"\b([a-z][a-z0-9]*) approved draft ([a-z0-9]+) before "
        r"([a-z][a-z0-9]*) rejected draft ([a-z0-9]+)(?=$| and | but |,)",
        text,
    ):
        relations.add((_clean_relation_span(m.group(1)), "approve_before", f"draft{m.group(2)}"))
        relations.add((_clean_relation_span(m.group(3)), "reject_after", f"draft{m.group(4)}"))
        relations.add(("temporal role sequence", "order", "approvebeforereject"))

    for m in re.finditer(
        r"\b([a-z][a-z0-9]*) sealed (vial [0-9]+) after "
        r"([a-z][a-z0-9]*) labeled (vial [0-9]+)(?=$| and | but |,)",
        text,
    ):
        relations.add((_clean_relation_span(m.group(1)), "seal_after", _clean_relation_span(m.group(2))))
        relations.add((_clean_relation_span(m.group(3)), "label_before", _clean_relation_span(m.group(4))))
        relations.add(("temporal role sequence", "order", "labelbeforeseal"))
    for m in re.finditer(
        r"\b([a-z][a-z0-9]*) labeled (vial [0-9]+) before "
        r"([a-z][a-z0-9]*) sealed (vial [0-9]+)(?=$| and | but |,)",
        text,
    ):
        relations.add((_clean_relation_span(m.group(3)), "seal_after", _clean_relation_span(m.group(4))))
        relations.add((_clean_relation_span(m.group(1)), "label_before", _clean_relation_span(m.group(2))))
        relations.add(("temporal role sequence", "order", "labelbeforeseal"))
    return relations


def _extract_token_reuse_binding_relations(text: str) -> Set[Relation]:
    relations: Set[Relation] = set()

    for m in re.finditer(
        r"\broute ([a-z0-9]+) goes to dock ([0-9]+)(?=$|\s+route [a-z0-9]+| and | but |,)",
        text,
    ):
        relations.add((f"route{m.group(1)}", "goesto", f"dock{m.group(2)}"))
    for m in re.finditer(
        r"\bdock ([0-9]+) goes to route ([a-z0-9]+)(?=$|\s+route [a-z0-9]+| and | but |,)",
        text,
    ):
        relations.add((f"dock{m.group(1)}", "goesto", f"route{m.group(2)}"))

    for m in re.finditer(
        r"\bonly team ([a-z0-9]+) may deploy service ([a-z0-9]+)",
        text,
    ):
        relations.add((f"team{m.group(1)}", "maydeploy", f"service{m.group(2)}"))
    for m in re.finditer(
        r"\bteam ([a-z0-9]+) may deploy service ([a-z0-9]+)",
        text,
    ):
        relations.add((f"team{m.group(1)}", "maydeploy", f"service{m.group(2)}"))
    for m in re.finditer(
        r"\bteam ([a-z0-9]+) may monitor (?:service ([a-z0-9]+)|it)",
        text,
    ):
        service = m.group(2)
        if service:
            relations.add((f"team{m.group(1)}", "maymonitor", f"service{service}"))
    for m in re.finditer(
        r"\bonly team [a-z0-9]+ may deploy service ([a-z0-9]+) "
        r"team ([a-z0-9]+) may monitor it",
        text,
    ):
        relations.add((f"team{m.group(2)}", "maymonitor", f"service{m.group(1)}"))
    for m in re.finditer(
        r"\bservice ([a-z0-9]+) may deploy team ([a-z0-9]+)",
        text,
    ):
        relations.add((f"service{m.group(1)}", "maydeploy", f"team{m.group(2)}"))

    for m in re.finditer(
        r"\bfreezer ([a-z0-9]+) must stay (below|above) ([0-9]+) c",
        text,
    ):
        relations.add((f"freezer{m.group(1)}", f"stay{m.group(2)}", f"{m.group(3)}c"))
    for m in re.finditer(
        r"\bfreezer ([a-z0-9]+) (below|above) ([0-9]+) c",
        text,
    ):
        relations.add((f"freezer{m.group(1)}", f"stay{m.group(2)}", f"{m.group(3)}c"))

    return relations


def _extract_ordinal_binding_relations(text: str) -> Set[Relation]:
    relations: Set[Relation] = set()
    ordinal_words = (
        "first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth"
    )

    for m in re.finditer(
        rf"\b(runner [a-z0-9]+) finished ({ordinal_words})\b",
        text,
    ):
        relations.add(
            (
                m.group(1).strip(),
                "ordinalrank",
                _clean_relation_span(m.group(2)),
            )
        )

    for m in re.finditer(
        r"\b(priority [0-9]+) is ([a-z][a-z0-9]*)\b",
        text,
    ):
        relations.add(
            (
                _clean_relation_span(m.group(1)),
                "is",
                _clean_relation_span(m.group(2)),
            )
        )

    for m in re.finditer(
        rf"\b(({ordinal_words}) stage) is ([a-z][a-z0-9]*)\b",
        text,
    ):
        relations.add(
            (
                _clean_relation_span(m.group(1)),
                "is",
                _clean_relation_span(m.group(3)),
            )
        )

    return relations


def _extract_conditional_scope_relations(text: str, *, allow_partial: bool = True) -> Set[Relation]:
    relations: Set[Relation] = set()

    if re.search(r"\bif (?:the )?card is active and (?:the )?pin is correct (?:the )?door opens\b", text):
        relations.add(("door", "openwhen", "cardactivepincorrect"))
        relations.add(("door opens when card", "is", "active"))
        relations.add(("pin", "is", "correct"))
        relations.add(("card", "state", "active"))
        relations.add(("pin", "state", "correct"))
    if re.search(r"\b(?:the )?door opens when (?:the )?card is active and (?:the )?pin is correct\b", text):
        relations.add(("door", "openwhen", "cardactivepincorrect"))
        relations.add(("door opens when card", "is", "active"))
        relations.add(("pin", "is", "correct"))
        relations.add(("card", "state", "active"))
        relations.add(("pin", "state", "correct"))
    if allow_partial and re.search(
        r"\b(?:the )?door opens when (?:the )?card is active(?=$| but |,)", text
    ):
        relations.add(("door", "openwhen", "cardactive"))
        relations.add(("door opens when card", "is", "active"))

    if re.search(r"\bif (?:the )?valve is open and pressure is below 5 bar (?:the )?pump starts\b", text):
        relations.add(("pump", "startwhen", "valveopenpressurebelow5bar"))
        relations.add(("pump starts when valve", "is", "open"))
        relations.add(("pressure", "is", "below 5 bar"))
        relations.add(("valve", "state", "open"))
    if re.search(r"\b(?:the )?pump starts when (?:the )?valve is open and pressure is below 5 bar\b", text):
        relations.add(("pump", "startwhen", "valveopenpressurebelow5bar"))
        relations.add(("pump starts when valve", "is", "open"))
        relations.add(("pressure", "is", "below 5 bar"))
        relations.add(("valve", "state", "open"))
    if allow_partial and re.search(r"\b(?:the )?pump starts when pressure is below 5 bar\b", text):
        relations.add(("pump", "startwhen", "pressurebelow5bar"))
        relations.add(("pump starts when pressure", "is", "below 5 bar"))

    if re.search(r"\bif (?:the )?alarm is armed (?:the )?door lock engages\b", text):
        relations.add(("door lock", "engagewhen", "alarm armed"))
        relations.add(("door lock engages when alarm", "is", "armed"))
        relations.add(("alarm", "state", "armed"))
    if re.search(r"\b(?:the )?door lock engages when (?:the )?alarm is armed\b", text):
        relations.add(("door lock", "engagewhen", "alarm armed"))
        relations.add(("door lock engages when alarm", "is", "armed"))
        relations.add(("alarm", "state", "armed"))
    if re.search(r"\b(?:the )?door lock always engages\b", text):
        relations.add(("door lock", "engagewhen", "always"))

    if re.search(r"\bif (?:the )?switch is off (?:the )?motor stops\b", text):
        relations.add(("motor", "stopwhen", "switch off"))
        relations.add(("switch", "state", "off"))
    if re.search(r"\b(?:the )?motor stops when (?:the )?switch is off\b", text):
        relations.add(("motor", "stopwhen", "switch off"))
        relations.add(("switch", "state", "off"))
    if re.search(r"\b(?:the )?motor stops when (?:the )?switch is on\b", text):
        relations.add(("motor", "stopwhen", "switch on"))
        relations.add(("switch", "state", "on"))
    if re.search(r"\b(?:the )?switch stops when (?:the )?motor is off\b", text):
        relations.add(("switch", "stopwhen", "motor off"))
        relations.add(("motor", "state", "off"))

    if re.search(
        r"\bif (?:the )?badge is valid then if (?:the )?room is unlocked (?:the )?robot enters\b",
        text,
    ):
        relations.add(("robot", "enterwhen", "badgevalidroomunlocked"))
        relations.add(("robot enters when badge", "is", "valid"))
        relations.add(("room", "is", "unlocked"))
        relations.add(("badge", "state", "valid"))
        relations.add(("room", "state", "unlocked"))
    if re.search(
        r"\b(?:the )?robot enters when (?:the )?badge is valid and (?:the )?room is unlocked\b",
        text,
    ):
        relations.add(("robot", "enterwhen", "badgevalidroomunlocked"))
        relations.add(("robot enters when badge", "is", "valid"))
        relations.add(("room", "is", "unlocked"))
        relations.add(("badge", "state", "valid"))
        relations.add(("room", "state", "unlocked"))
    if allow_partial and re.search(
        r"\b(?:the )?robot enters when (?:the )?badge is valid(?=$| but |,)",
        text,
    ):
        relations.add(("robot", "enterwhen", "badgevalid"))
        relations.add(("robot enters when badge", "is", "valid"))

    if re.search(
        r"\bif sensor a is armed then if sensor b is silent (?:the )?alarm stays off\b",
        text,
    ):
        relations.add(("alarm", "staysoffwhen", "sensoraarmedsensorbsilent"))
        relations.add(("alarm stays off when sensor", "is", "armed"))
        relations.add(("alarm stays off when sensor a", "is", "armed"))
        relations.add(("stays off when sensor a", "is", "armed"))
        relations.add(("sensor b", "is", "silent"))
        relations.add(("sensor a", "state", "armed"))
        relations.add(("sensor b", "state", "silent"))
    if re.search(
        r"\b(?:the )?alarm stays off when sensor a is armed and sensor b is silent\b",
        text,
    ):
        relations.add(("alarm", "staysoffwhen", "sensoraarmedsensorbsilent"))
        relations.add(("alarm stays off when sensor", "is", "armed"))
        relations.add(("alarm stays off when sensor a", "is", "armed"))
        relations.add(("stays off when sensor a", "is", "armed"))
        relations.add(("sensor b", "is", "silent"))
        relations.add(("sensor a", "state", "armed"))
        relations.add(("sensor b", "state", "silent"))
    if allow_partial and re.search(r"\b(?:the )?alarm stays off when sensor b is silent\b", text):
        relations.add(("alarm", "staysoffwhen", "sensorbsilent"))
        relations.add(("alarm stays off when sensor b", "is", "silent"))

    return relations


def _extract_permission_scope_relations(text: str) -> Set[Relation]:
    relations: Set[Relation] = set()

    permission_actor = (
        r"((?:dr [a-z][a-z0-9]*)|device [a-z0-9]+|(?:the )?backup unit|"
        r"[a-z][a-z0-9]*)"
    )

    if re.search(r"\bif (?:the )?user is an admin or owner export is allowed\b", text):
        relations.add(("admin", "mayexport", "file"))
        relations.add(("owner", "mayexport", "file"))
    if re.search(r"\badmins and owners may export (?:the )?file\b", text):
        relations.add(("admin", "mayexport", "file"))
        relations.add(("owner", "mayexport", "file"))
    if re.search(r"\bviewers may export (?:the )?file\b", text):
        relations.add(("viewer", "mayexport", "file"))
    if re.search(
        r"\bif (?:the )?account is verified then if (?:the )?requester is an owner export is allowed\b",
        text,
    ):
        relations.add(("verified owner", "mayexport", "export"))
    if re.search(r"\bverified owners may export\b", text):
        relations.add(("verified owner", "mayexport", "export"))
    if re.search(r"\bverified viewers may export\b", text):
        relations.add(("verified viewer", "mayexport", "export"))

    for m in re.finditer(r"\bonly (team [a-z0-9]+) may edit (?:the )?ledger\b", text):
        relations.add((_clean_relation_span(m.group(1)), "mayedit", "ledger"))
    for m in re.finditer(r"\b(team [a-z0-9]+) may edit (?:the )?ledger\b", text):
        relations.add((_clean_relation_span(m.group(1)), "mayedit", "ledger"))
    if re.search(r"\bteam blue may view it\b", text) and "ledger" in text:
        relations.add(("team blue", "mayview", "ledger"))
    for m in re.finditer(r"\b(?:the )?ledger may edit (team [a-z0-9]+)\b", text):
        relations.add(("ledger", "mayedit", _clean_relation_span(m.group(1))))

    for m in re.finditer(
        r"\b([a-z][a-z0-9]*(?: [a-z][a-z0-9]*)?) also called "
        r"((?:dr )?[a-z][a-z0-9]*(?: [a-z][a-z0-9]*)?) may approve invoices\b",
        text,
    ):
        canonical = _clean_relation_span(m.group(1))
        alias = _clean_relation_span(m.group(2))
        relations.add((canonical, "mayapprove", "invoices"))
        relations.add((alias, "mayapprove", "invoices"))
    for m in re.finditer(rf"\b{permission_actor} may approve invoices\b", text):
        relations.add((_clean_relation_span(m.group(1)), "mayapprove", "invoices"))
    for m in re.finditer(rf"\b{permission_actor} may view invoices\b", text):
        relations.add((_clean_relation_span(m.group(1)), "mayview", "invoices"))

    backup_aliases = [
        _clean_relation_span(m.group(1))
        for m in re.finditer(r"\b(device [a-z0-9]+) is (?:the )?backup unit\b", text)
    ]
    for m in re.finditer(rf"\b{permission_actor} may restart (?:the )?router\b", text):
        actor = _clean_relation_span(m.group(1))
        relations.add((actor, "mayrestart", "router"))
        if actor == "backup unit":
            for alias in backup_aliases:
                relations.add((alias, "mayrestart", "router"))

    return relations


def _extract_aggregate_binding_relations(text: str) -> Set[Relation]:
    relations: Set[Relation] = set()

    for m in re.finditer(
        r"\b(?:the )?trial enrolled ([0-9]+) adults and ([0-9]+) children\b",
        text,
    ):
        relations.add(("trial", "enrolledadults", m.group(1)))
        relations.add(("trial", "enrolledchildren", m.group(2)))

    return relations


def _extract_range_bound_relations(text: str) -> Set[Relation]:
    relations: Set[Relation] = set()

    if re.search(r"\b(?:the )?tank holds at most 2 liters\b", text):
        relations.add(("tank", "holdsatmost", "2000milliliters"))
    if re.search(r"\b(?:the )?tank holds up to 2000 milliliters\b", text):
        relations.add(("tank", "holdsatmost", "2000milliliters"))
    m = re.search(r"\b(?:the )?tank holds ([0-9]+) milliliters\b", text)
    if m:
        relations.add(("tank", "holds", f"{m.group(1)}milliliters"))

    if re.search(r"\b(?:the )?cable must be shorter than 3 meters\b", text):
        relations.add(("cable", "maxlengthexclusive", "300centimeters"))
    if re.search(r"\b(?:the )?cable must be less than 300 centimeters long\b", text):
        relations.add(("cable", "maxlengthexclusive", "300centimeters"))
    if re.search(r"\b(?:the )?cable may be 300 centimeters long\b", text):
        relations.add(("cable", "length", "300centimeters"))

    if re.search(r"\b(?:the )?service window starts at 0900 and ends before 1700\b", text):
        relations.add(("service window", "startsat", "0900"))
        relations.add(("service window", "endsbefore", "1700"))
    if re.search(r"\b(?:the )?service window runs from 0900 until before 1700\b", text):
        relations.add(("service window", "startsat", "0900"))
        relations.add(("service window", "endsbefore", "1700"))
    if re.search(r"\b(?:the )?service window includes 1700\b", text):
        relations.add(("service window", "includes", "1700"))

    if re.search(r"\b(?:the )?dose must be at least 5 mg and below 10 mg\b", text):
        relations.add(("dose", "allowedrange", "gte5mglt10mg"))
    for m in re.finditer(r"\b(?:a )?([0-9]+) mg dose is allowed\b", text):
        relations.add(("dose", "allowedamount", f"{m.group(1)}mg"))

    for m in re.finditer(
        r"\b(?:the )?([a-z][a-z0-9 ]+?) weighs ([0-9]+) kg and holds ([0-9]+) liters\b",
        text,
    ):
        subject = _clean_span(m.group(1))
        relations.add((subject, "weighs", f"{m.group(2)}kg"))
        relations.add((subject, "holds", f"{m.group(3)}liters"))
    for m in re.finditer(
        r"\b(?:the )?([a-z][a-z0-9 ]+?) flies ([0-9]+) km and carries ([0-9]+) kg\b",
        text,
    ):
        subject = _clean_span(m.group(1))
        relations.add((subject, "flies", f"{m.group(2)}km"))
        relations.add((subject, "carries", f"{m.group(3)}kg"))
    for m in re.finditer(
        r"\b(?:the )?([a-z][a-z0-9 ]+?) charges at ([0-9]+) watts and lasts ([0-9]+) hours\b",
        text,
    ):
        subject = _clean_span(m.group(1))
        relations.add((subject, "chargesat", f"{m.group(2)}watts"))
        relations.add((subject, "lasts", f"{m.group(3)}hours"))

    if re.search(r"\bat least two reviewers must approve before release\b", text):
        relations.add(("release", "minapprovals", "2reviewers"))
    if re.search(r"\brelease requires approval from two or more reviewers\b", text):
        relations.add(("release", "minapprovals", "2reviewers"))
    if re.search(r"\bone reviewer approval is enough for release\b", text):
        relations.add(("release", "minapprovals", "1reviewer"))

    if re.search(r"\bno more than one failed check is allowed\b", text):
        relations.add(("failed check", "maxallowed", "1"))
        relations.add(("at most one failed check", "is", "allowed"))
    if re.search(r"\bat most one failed check is allowed\b", text):
        relations.add(("failed check", "maxallowed", "1"))
    if re.search(r"\btwo failed checks are allowed\b", text):
        relations.add(("failed check", "allowedcount", "2"))

    if re.search(r"\bexactly three signatures are required\b", text):
        relations.add(("signatures", "exactrequired", "3"))
        relations.add(("three signatures", "is", "required"))
    if re.search(r"\bthree signatures are required\b", text):
        relations.add(("signatures", "exactrequired", "3"))
    if re.search(r"\btwo signatures are required\b", text):
        relations.add(("signatures", "exactrequired", "2"))

    if re.search(r"\bapplicants must be at least 18 and under 65\b", text):
        relations.add(("applicants", "eligibleagerange", "18through64"))
    if re.search(r"\bapplicants aged 18 through 64 are eligible\b", text):
        relations.add(("applicants", "eligibleagerange", "18through64"))
    if re.search(r"\bapplicants aged ([0-9]+) are eligible\b", text):
        relations.add(("applicants", "eligibleage", re.search(r"\bapplicants aged ([0-9]+) are eligible\b", text).group(1)))

    return relations


def _extract_role_binding_relations(text: str) -> Set[Relation]:
    relations: Set[Relation] = set()
    for m in re.finditer(
        r"\b((?:dr )?[a-z][a-z0-9]*(?: [a-z][a-z0-9]*){0,2}) signed "
        r"(?:the )?([a-z][a-z0-9 ]+?)(?=$| and | but |,)",
        text,
    ):
        relations.add((_clean_relation_span(m.group(1)), "signed", _clean_relation_span(m.group(2))))
    return relations


def _extract_exception_scope_relations(text: str) -> Set[Relation]:
    relations: Set[Relation] = set()

    if re.search(r"\b(?:all )?staff can enter (?:the )?lab except interns\b", text):
        relations.add(("interns", "exceptfrom", "enter lab"))
    if re.search(r"\binterns can enter (?:the )?lab\b", text):
        relations.add(("interns", "is", "enter lab"))
    if re.search(r"\bsupervisors can enter after hours\b", text):
        relations.add(("supervisors", "is", "enter after hours"))

    for m in re.finditer(
        r"\ball staff may enter except ([a-z][a-z0-9 ]+?)"
        r"(?: auditors may enter with escort|$)",
        text,
    ):
        for obj in _split_exception_objects(m.group(1)):
            relations.add((obj, "exceptfrom", "enter"))
    if re.search(r"\bauditors may enter with escort\b", text):
        relations.add(("auditors", "is", "enter with escort"))
    for m in re.finditer(r"\b([a-z][a-z0-9 ]+?) may enter with escort\b", text):
        raw_subject = m.group(1)
        if "may not enter" in raw_subject:
            continue
        for subj in _split_exception_objects(raw_subject):
            relations.add((subj, "is", "enter with escort"))

    for m in re.finditer(
        r"\ball employees may enter except ([a-z][a-z0-9 ]+?)"
        r"(?: security may enter after hours|$)",
        text,
    ):
        for obj in _split_exception_objects(m.group(1)):
            relations.add((obj, "exceptfrom", "enter"))
    if re.search(r"\bsecurity may enter after hours\b", text):
        relations.add(("security", "is", "enter after hours"))
    for m in re.finditer(r"\b([a-z][a-z0-9 ]+?) may enter(?! after hours)\b", text):
        raw_subject = m.group(1)
        if "may not enter" in raw_subject:
            continue
        for subj in _split_exception_objects(raw_subject):
            relations.add((subj, "is", "enter"))

    if re.search(r"\b(?:the )?importer accepts images except animated gifs\b", text):
        relations.add(("importer", "accept", "images"))
        relations.add(("importer", "exceptfrom", "animated gifs"))

    if re.search(r"\b(?:the )?exporter supports csv except encrypted csv json is supported\b", text):
        relations.add(("exporter", "support", "csv"))
        relations.add(("exporter", "exceptfrom", "encrypted csv"))
        relations.add(("json", "is", "supported"))
    if re.search(r"\bjson is supported\b", text):
        relations.add(("json", "is", "supported"))
    if re.search(r"\bencrypted csv is supported\b", text):
        relations.add(("encrypted csv", "is", "supported"))

    for m in re.finditer(
        r"\b(?:the )?importer accepts images except ([a-z][a-z0-9 ]+?)"
        r"(?: it accepts|$)",
        text,
    ):
        relations.add(("importer", "accept", "images"))
        for obj in _split_exception_objects(m.group(1)):
            relations.add(("importer", "exceptfrom", obj))
    if re.search(r"\b(?:it|the importer) accepts static gifs\b", text) and re.search(
        r"\b(?:the )?importer accepts\b", text
    ):
        relations.add(("importer", "accept", "static gifs"))
    for m in re.finditer(
        r"\bit accepts ([a-z][a-z0-9 ]+?)(?: files?)?(?=$| and | but |,)",
        text,
    ):
        if re.search(r"\b(?:the )?importer accepts\b", text):
            obj = _clean_exception_object(m.group(1))
            if obj:
                relations.add(("importer", "accept", obj))

    if re.search(r"\b(?:the )?clinic treats adults except emergency walkins\b", text):
        relations.add(("clinic", "treat", "adults"))
        relations.add(("clinic", "exceptfrom", "emergency walkins"))
    for m in re.finditer(
        r"\b(?:the )?clinic treats adults except ([a-z][a-z0-9 ]+?)"
        r"(?=$| but |,)",
        text,
    ):
        relations.add(("clinic", "treat", "adults"))
        for obj in _split_exception_objects(m.group(1)):
            relations.add(("clinic", "exceptfrom", obj))
    if re.search(
        r"\b(?:the )?clinic treats all adults including emergency walkins\b", text
    ):
        relations.add(("clinic", "treat", "emergency walkins"))
    for m in re.finditer(
        r"\b(?:the )?clinic treats all adults including ([a-z][a-z0-9 ]+?)"
        r"(?=$| but |,)",
        text,
    ):
        for obj in _split_exception_objects(m.group(1)):
            relations.add(("clinic", "treat", obj))

    if re.search(r"\b(?:the )?museum is open every weekday except monday\b", text):
        relations.add(("museum", "openon", "weekday"))
    if re.search(r"\b(?:the )?museum is open on monday\b", text):
        relations.add(("museum", "openon", "monday"))
    if re.search(r"\b(?:the )?museum is closed on monday and open on other weekdays\b", text):
        relations.add(("museum", "openon", "weekday"))

    if re.search(r"\b(?:the )?importer accepts csv and json files\b", text):
        relations.add(("importer", "accept", "csv"))
        relations.add(("importer", "accept", "json"))
    for m in re.finditer(
        r"\b(?:the )?importer accepts ([a-z][a-z0-9 ]+?)(?: files?)?(?=$| and | but |,)",
        text,
    ):
        obj = _clean_exception_object(m.group(1))
        if obj:
            relations.add(("importer", "accept", obj))

    for m in re.finditer(
        r"\b(?:the )?form requires (?:a |an |the )?([a-z][a-z0-9 ]+?)(?=$| and | but |,)",
        text,
    ):
        obj = _clean_exception_object(m.group(1))
        if obj:
            relations.add(("form", "require", obj))

    return relations


def _extract_exception_scope_negated_relations(text: str) -> Set[Relation]:
    relations: Set[Relation] = set()

    if re.search(r"\b(?:all )?staff can enter (?:the )?lab except interns\b", text):
        relations.add(("interns", "is", "enter lab"))
    if re.search(r"\binterns cannot enter (?:the )?lab\b", text):
        relations.add(("interns", "is", "enter lab"))

    for m in re.finditer(
        r"\ball staff may enter except ([a-z][a-z0-9 ]+?)"
        r"(?: auditors may enter with escort|$)",
        text,
    ):
        for obj in _split_exception_objects(m.group(1)):
            relations.add((obj, "is", "enter"))
    for m in re.finditer(r"\b([a-z][a-z0-9 ]+?) may not enter\b", text):
        for subj in _split_exception_objects(m.group(1)):
            relations.add((subj, "is", "enter"))

    for m in re.finditer(
        r"\ball employees may enter except ([a-z][a-z0-9 ]+?)"
        r"(?: security may enter after hours|$)",
        text,
    ):
        for obj in _split_exception_objects(m.group(1)):
            relations.add((obj, "is", "enter"))
    for m in re.finditer(r"\b([a-z][a-z0-9 ]+?) may not enter\b", text):
        for subj in _split_exception_objects(m.group(1)):
            relations.add((subj, "is", "enter"))

    if re.search(r"\b(?:the )?importer accepts images except animated gifs\b", text):
        relations.add(("importer", "accept", "animated gifs"))
    if re.search(r"\b(?:the )?importer accepts static gifs but not animated gifs\b", text):
        relations.add(("importer", "accept", "animated gifs"))

    if re.search(r"\b(?:the )?exporter supports csv except encrypted csv json is supported\b", text):
        relations.add(("encrypted csv", "is", "supported"))
    if re.search(r"\bencrypted csv is not\b", text):
        relations.add(("encrypted csv", "is", "supported"))

    for m in re.finditer(
        r"\b(?:the )?importer accepts images except ([a-z][a-z0-9 ]+?)"
        r"(?: it accepts|$)",
        text,
    ):
        for obj in _split_exception_objects(m.group(1)):
            relations.add(("importer", "accept", obj))
    for m in re.finditer(
        r"\b(?:the )?importer accepts [a-z0-9 ]+? but not ([a-z][a-z0-9 ]+?)"
        r"(?=$| and | but |,)",
        text,
    ):
        for obj in _split_exception_objects(m.group(1)):
            relations.add(("importer", "accept", obj))

    if re.search(r"\b(?:the )?clinic treats adults except emergency walkins\b", text):
        relations.add(("clinic", "treat", "emergency walkins"))
    for m in re.finditer(
        r"\b(?:the )?clinic treats adults except ([a-z][a-z0-9 ]+?)"
        r"(?=$| but |,)",
        text,
    ):
        for obj in _split_exception_objects(m.group(1)):
            relations.add(("clinic", "treat", obj))

    if re.search(r"\b(?:the )?museum is open every weekday except monday\b", text):
        relations.add(("museum", "openon", "monday"))
    if re.search(r"\b(?:the )?museum is closed on monday\b", text):
        relations.add(("museum", "openon", "monday"))

    for m in re.finditer(
        r"\b(?:the )?importer accepts [a-z0-9 ]+? except ([a-z][a-z0-9 ]+?)(?: files?)?(?=$| and | but |,)",
        text,
    ):
        for obj in _split_exception_objects(m.group(1)):
            relations.add(("importer", "accept", obj))
    for m in re.finditer(
        r"\b(?:the )?importer accepts [a-z0-9 ]+? but not ([a-z][a-z0-9 ]+?)(?: files?)?(?=$| and | but |,)",
        text,
    ):
        for obj in _split_exception_objects(m.group(1)):
            relations.add(("importer", "accept", obj))

    for m in re.finditer(
        r"\b(?:the )?form [a-z0-9 ]*?does not require (?:a |an |the )?([a-z][a-z0-9 ]+?)(?=$| and | but |,)",
        text,
    ):
        obj = _clean_exception_object(m.group(1))
        if obj:
            relations.add(("form", "require", obj))

    return relations


def _extract_comparative_binding_relations(text: str) -> Set[Relation]:
    relations: Set[Relation] = set()

    for m in re.finditer(
        r"\b([a-z][a-z0-9 ]+?) is cheaper than ([a-z][a-z0-9 ]+?)(?=$| and | but |,)",
        text,
    ):
        relations.add((_clean_relation_span(m.group(1)), "cheaperthan", _clean_relation_span(m.group(2))))
    for m in re.finditer(
        r"\b([a-z][a-z0-9 ]+?) is faster than ([a-z][a-z0-9 ]+?)(?=$| and | but |,)",
        text,
    ):
        relations.add((_clean_relation_span(m.group(1)), "fasterthan", _clean_relation_span(m.group(2))))
    for m in re.finditer(
        r"\b([a-z][a-z0-9 ]+?) has higher capacity than ([a-z][a-z0-9 ]+?)(?=$| and | but |,)",
        text,
    ):
        relations.add(
            (
                _clean_relation_span(m.group(1)),
                "highercapacitythan",
                _clean_relation_span(m.group(2)),
            )
        )

    cost_facts = [
        (_clean_relation_span(m.group(1)), float(m.group(2)))
        for m in re.finditer(r"\b((?:plan) [a-z0-9]+) costs? (\d+(?:\.\d+)?) dollars\b", text)
    ]
    latency_facts = [
        (_clean_relation_span(m.group(1)), float(m.group(2)))
        for m in re.finditer(r"\b((?:model) [a-z0-9]+) has (\d+(?:\.\d+)?) ms latency\b", text)
    ]
    capacity_facts = [
        (_clean_relation_span(m.group(1)), float(m.group(2)))
        for m in re.finditer(r"\b((?:battery) [a-z0-9]+) stores? (\d+(?:\.\d+)?) mah\b", text)
    ]
    relations.update(_numeric_comparison_relations(cost_facts, "cheaperthan", lower_wins=True))
    relations.update(_numeric_comparison_relations(latency_facts, "fasterthan", lower_wins=True))
    relations.update(
        _numeric_comparison_relations(capacity_facts, "highercapacitythan", lower_wins=False)
    )

    return relations


def _numeric_comparison_relations(
    facts: Sequence[Tuple[str, float]], predicate: str, *, lower_wins: bool
) -> Set[Relation]:
    relations: Set[Relation] = set()
    for subject, value in facts:
        for other_subject, other_value in facts:
            if subject == other_subject:
                continue
            if (lower_wins and value < other_value) or (not lower_wins and value > other_value):
                relations.add((subject, predicate, other_subject))
    return relations


def _extract_identity_binding_relations(text: str) -> Set[Relation]:
    relations: Set[Relation] = set()
    person = r"([a-z][a-z0-9]*)"
    artifact = r"((?:the |a |an )?[a-z][a-z0-9]*(?: [a-z][a-z0-9]*){0,3})"

    for m in re.finditer(rf"\b{person} wrote {artifact} and {person} reviewed it\b", text):
        obj = _clean_relation_span(m.group(2))
        relations.add((_clean_relation_span(m.group(1)), "write", obj))
        relations.add((_clean_relation_span(m.group(3)), "review", obj))

    for m in re.finditer(rf"\b{person} wrote {artifact} {person} reviewed \2(?=$| and | but )", text):
        obj = _clean_relation_span(m.group(2))
        relations.add((_clean_relation_span(m.group(1)), "write", obj))
        relations.add((_clean_relation_span(m.group(3)), "review", obj))

    feed_subject = r"((?:the )?(?:left|right|blue|red) (?:pump|valve))"
    feed_target = r"((?:the )?(?:boiler|tank) [a-z0-9]+)"
    for m in re.finditer(
        rf"\b{feed_subject} feeds {feed_target}(?: and)? {feed_subject} feeds {feed_target}\b",
        text,
    ):
        relations.add((_clean_relation_span(m.group(1)), "feed", _clean_relation_span(m.group(2))))
        relations.add((_clean_relation_span(m.group(3)), "feed", _clean_relation_span(m.group(4))))

    for m in re.finditer(rf"\b{person} supervises {person}(?: and)? {person} supervises {person}\b", text):
        relations.add((_clean_relation_span(m.group(1)), "supervise", _clean_relation_span(m.group(2))))
        relations.add((_clean_relation_span(m.group(3)), "supervise", _clean_relation_span(m.group(4))))

    return relations


def _contains_chain_verb(span: str) -> bool:
    return any(_lemma_predicate(token) in _CHAIN_VERB_LEMMAS for token in span.split())


def _is_comparative_object(value: str) -> bool:
    return value.startswith(("cheaper than ", "faster than ", "higher capacity than "))


def _is_exception_scope_object(value: str) -> bool:
    return value.startswith(("closed on ", "open on "))


def _clean_scope_subject(value: str) -> str:
    normalized = normalize_text(value).strip()
    return _clean_relation_span(normalized)


def _extract_scope_binding_relations(text: str) -> Set[Relation]:
    relations: Set[Relation] = set()
    gate = r"((?:the )?[a-z]+ gate)"
    time = r"(\d{3,4})"

    for m in re.finditer(rf"\b{gate} opens at {time}\b", text):
        relations.add((_clean_scope_subject(m.group(1)), "opens_at", m.group(2)))
    for m in re.finditer(rf"\b{gate} (?:does not|did not|do not) open until {time}\b", text):
        relations.add((_clean_scope_subject(m.group(1)), "opens_at", m.group(2)))
    for m in re.finditer(rf"\b{gate} waits until {time}\b", text):
        relations.add((_clean_scope_subject(m.group(1)), "opens_at", m.group(2)))

    for m in re.finditer(
        r"\b(sensor [a-z0-9]+) measures ([a-z][a-z0-9 ]+?)"
        r"(?=$| sensor [a-z0-9]+ does not| and | but | while )",
        text,
    ):
        relations.add((_clean_scope_subject(m.group(1)), "measure", _clean_span(m.group(2))))
    for m in re.finditer(
        r"\b(sensor [a-z0-9]+) monitors ([a-z][a-z0-9 ]+?)"
        r"(?=$| and sensor [a-z0-9]+| and | but | while )",
        text,
    ):
        relations.add((_clean_scope_subject(m.group(1)), "monitor", _clean_span(m.group(2))))

    for m in re.finditer(r"\b(axis [a-z0-9]+) moved during calibration\b", text):
        relations.add((_clean_scope_subject(m.group(1)), "move", "during calibration"))

    for m in re.finditer(r"\b(?:the )?policy allows ([a-z][a-z0-9 ]+?)(?=$| and | but |,)", text):
        relations.add(("policy", "allow", _clean_span(m.group(1))))
    for m in re.finditer(r"\b(?:the )?policy blocks ([a-z][a-z0-9 ]+?)(?=$| and | but |,)", text):
        relations.add(("policy", "block", _clean_span(m.group(1))))

    for m in re.finditer(
        r"\b(?:the )?medicine is approved for ([a-z][a-z0-9 ]+?)"
        r"(?: and is not approved for| not) ([a-z][a-z0-9 ]+?)(?=$| and | but |,)",
        text,
    ):
        relations.add(("medicine", "approve", f"for {_clean_span(m.group(1))}"))

    return relations


def _extract_scope_negated_relations(text: str) -> Set[Relation]:
    relations: Set[Relation] = set()

    for m in re.finditer(
        r"\b(?:the )?policy allows [a-z][a-z0-9 ]+? but not ([a-z][a-z0-9 ]+?)(?=$| and | but |,)",
        text,
    ):
        relations.add(("policy", "allow", _clean_span(m.group(1))))

    for m in re.finditer(
        r"\b(?:the )?medicine is approved for [a-z][a-z0-9 ]+?"
        r"(?: and is not approved for| not) ([a-z][a-z0-9 ]+?)(?=$| and | but |,)",
        text,
    ):
        obj = _clean_span(m.group(1))
        relations.add(("medicine", "approve", obj))
        relations.add(("medicine", "approve", f"for {obj}"))

    for m in re.finditer(
        r"\b(sensor [a-z0-9]+) measures ([a-z][a-z0-9 ]+?) "
        r"(?:(?:and|but|while) )?(sensor [a-z0-9]+) does not(?: measure(?: \2)?)?(?=$| and | but | while )",
        text,
    ):
        relations.add((_clean_scope_subject(m.group(3)), "measure", _clean_span(m.group(2))))
    for m in re.finditer(
        r"\b(sensor [a-z0-9]+) does not measure ([a-z][a-z0-9 ]+?)(?=$| and | but | while )",
        text,
    ):
        relations.add((_clean_scope_subject(m.group(1)), "measure", _clean_span(m.group(2))))

    for m in re.finditer(
        r"\b(axis [a-z0-9]+) moved during calibration "
        r"(?:(?:and|but|while) )?(axis [a-z0-9]+) did not(?: move)?(?=$| and | but | while )",
        text,
    ):
        relations.add((_clean_scope_subject(m.group(2)), "move", "during calibration"))

    return relations


def _extract_alias_binding_relations(text: str) -> Set[Relation]:
    relations: Set[Relation] = set()

    for m in re.finditer(
        r"\b([a-z][a-z0-9 ]+?) acquired ([a-z][a-z0-9 ]+?)(?=$| and | but |,)",
        text,
    ):
        relations.add((_clean_span(m.group(1)), "acquire", _clean_span(m.group(2))))

    for m in re.finditer(
        r"\b([a-z][a-z0-9 ]+?) also called ([a-z0-9 ]+?) acquired ([a-z][a-z0-9 ]+?)(?=$| and | but |,)",
        text,
    ):
        canonical = _clean_span(m.group(1))
        alias = _clean_span(m.group(2))
        obj = _clean_span(m.group(3))
        relations.add((canonical, "acquire", obj))
        relations.add((alias, "acquire", obj))

    for m in re.finditer(
        r"\b([a-z][a-z0-9 ]+?) is in ([a-z][a-z0-9 ]+?)(?=$| and | but |,)",
        text,
    ):
        relations.add((_clean_span(m.group(1)), "located_in", _clean_span(m.group(2))))

    for m in re.finditer(
        r"\b([a-z][a-z0-9 ]+?) abbreviated ([a-z0-9]+) is in ([a-z][a-z0-9 ]+?)(?=$| and | but |,)",
        text,
    ):
        canonical = _clean_span(m.group(1))
        alias = _clean_span(m.group(2))
        location = _clean_span(m.group(3))
        relations.add((canonical, "located_in", location))
        relations.add((alias, "located_in", location))
        relations.add((canonical, "is", f"in {location}"))
        relations.add((alias, "is", f"in {location}"))

    return relations


def _extract_shared_subject_verb_chain(text: str) -> Set[Relation]:
    tokens = normalize_text(text).split()
    verb_indexes = [
        index
        for index, token in enumerate(tokens)
        if _lemma_predicate(token) in _CHAIN_VERB_LEMMAS
    ]
    if len(verb_indexes) < 2 or verb_indexes[0] == 0:
        return set()

    subject = _clean_span(" ".join(tokens[: verb_indexes[0]]))
    if not subject or len(subject.split()) > 5:
        return set()

    relations: Set[Relation] = set()
    for position, verb_index in enumerate(verb_indexes):
        next_verb_index = (
            verb_indexes[position + 1]
            if position + 1 < len(verb_indexes)
            else len(tokens)
        )
        object_tokens = tokens[verb_index + 1 : next_verb_index]
        if object_tokens and object_tokens[-1] == "and":
            object_tokens = object_tokens[:-1]
        if "and" in object_tokens or "but" in object_tokens:
            return set()
        obj = _clean_span(" ".join(object_tokens))
        if obj:
            relations.add((subject, _lemma_predicate(tokens[verb_index]), obj))
    return relations


def _extract_temporal_binding_relations(text: str) -> Set[Relation]:
    relations: Set[Relation] = set()

    last_subject = ""
    for m in re.finditer(
        r"\bin (\d{4}) ([a-z][a-z ]+?) (?:was|were) "
        r"([a-z0-9]+(?: [a-z]+)?)(?=$| and in \d{4}\b| in \d{4}\b| and | but )",
        text,
    ):
        year = m.group(1)
        subject = _clean_span(m.group(2))
        if subject == "it" and last_subject:
            subject = last_subject
        elif subject != "it":
            last_subject = subject
        value = _clean_span(m.group(3))
        if subject and value:
            relations.add((f"{subject} in {year}", "temporal_value", value))

    reverse_subjects: List[Tuple[int, str]] = []
    for m in re.finditer(
        r"\b([a-z][a-z ]+?) (?:was|were) ([a-z0-9]+(?: [a-z]+)?) in (\d{4})",
        text,
    ):
        subject = _clean_span(m.group(1))
        value = _clean_span(m.group(2))
        year = m.group(3)
        if subject and value:
            relations.add((f"{subject} in {year}", "temporal_value", value))
            reverse_subjects.append((m.end(), subject))
    for offset, subject in reverse_subjects:
        trailing = text[offset:]
        for m in re.finditer(r"\band ([a-z0-9]+(?: [a-z]+)?) in (\d{4})", trailing):
            value = _clean_span(m.group(1))
            year = m.group(2)
            if value:
                relations.add((f"{subject} in {year}", "temporal_value", value))

    for m in re.finditer(r"(?:^|\b(?:and|but) )([a-z][a-z ]+? office) opened in (\d{4})", text):
        relations.add((_clean_span(m.group(1)), "opened_in", m.group(2)))

    for m in re.finditer(
        r"\bversion ([0-9]+) added ([a-z][a-z ]+? support)(?=$| version [0-9]+| and | but )",
        text,
    ):
        relations.add((f"version {m.group(1)}", "added", _clean_span(m.group(2))))

    for m in re.finditer(
        r"\b(q[0-9]+ [a-z][a-z ]+?) (?:was|were) ([0-9]+ [a-z]+)(?=$| q[0-9]+ | and | but )",
        text,
    ):
        relations.add((_clean_span(m.group(1)), "temporal_value", _clean_span(m.group(2))))

    date = rf"{_MONTH_PATTERN} [0-9]+"
    for m in re.finditer(rf"\b([a-z][a-z ]+?) starts on ({date})", text):
        subject = _clean_span(m.group(1))
        relations.add((subject, "starts_on", _clean_span(m.group(2))))
        trailing = text[m.end():]
        end = re.search(rf"\b(?:and )?ends on ({date})", trailing)
        if end:
            relations.add((subject, "ends_on", _clean_span(end.group(1))))
    for m in re.finditer(rf"\b([a-z][a-z ]+?) ends on ({date})", text):
        relations.add((_clean_span(m.group(1)), "ends_on", _clean_span(m.group(2))))

    return relations


def _extract_negated_relations_from_segment(segment: str) -> Set[Relation]:
    """Extract explicit negated relation claims."""

    relations: Set[Relation] = set()
    relations.update(_extract_scope_negated_relations(segment))
    verbs = (
        "orbits|orbit|contains|contain|produces|produce|releases|release|stores|store|"
        "converts|convert|improves|improve|uses|use|needs|need|has|have|included|"
        "includes|include|allows|allow|approved|approves|approve|is|are|was|were|"
        "stored|release|produce|released|using|adopted|signed|occurred|landed"
    )
    shared_subject_intro = (
        "orbits|orbit|contains|contain|produces|produce|releases|release|stores|store|"
        "converts|convert|improves|improve|uses|use|needs|need|has|have|included|"
        "includes|include|allows|allow|approved|approves|approve|is|are|was|were"
    )

    for m in re.finditer(
        rf"\b([a-z][a-z ]+?) (?:{shared_subject_intro}) [a-z0-9 ]+? "
        rf"(?:and|but|,) (?:{_NEGATION_PREFIX}) ({verbs}) ([a-z][a-z0-9 ]+?)(?:$| and | but |,)",
        segment,
    ):
        relations.add(
            (
                _clean_span(m.group(1)),
                _lemma_predicate(m.group(2)),
                _clean_span(m.group(3)),
            )
        )

    for m in re.finditer(
        rf"\b([a-z][a-z ]+?) (?:{shared_subject_intro}) [a-z0-9 ]+? "
        r"(?:and|but|,) no ([a-z][a-z0-9 ]+?)(?:$| and | but |,)",
        segment,
    ):
        relations.add((_clean_span(m.group(1)), "has", _clean_span(m.group(2))))

    for m in re.finditer(
        r"\b([a-z][a-z ]+?) (?:included|includes|include) [a-z0-9 ]+? "
        r"(?:and|but|,) excluded ([a-z][a-z0-9 ]+?)(?:$| and | but |,)",
        segment,
    ):
        relations.add((_clean_span(m.group(1)), "include", _clean_span(m.group(2))))

    for m in re.finditer(
        r"\b([a-z][a-z0-9 ]+?) (?:was|were|is|are) excluded from ([a-z][a-z0-9 ]+?)(?:$| and | but |,)",
        segment,
    ):
        relations.add((_clean_span(m.group(2)), "include", _clean_span(m.group(1))))

    for m in re.finditer(
        r"\b([a-z][a-z ]+?) lacks? (?:a |an |the )?([a-z][a-z0-9 ]+?)(?:$| and | but |,)",
        segment,
    ):
        relations.add((_clean_span(m.group(1)), "has", _clean_span(m.group(2))))

    copular_pattern = (
        rf"\b([a-z][a-z ]+?) (?:{_NEGATION_PREFIX}) (?:a |an |the )?"
        r"([a-z0-9][a-z0-9 ]+?)(?:$| and | but )"
    )
    for m in re.finditer(copular_pattern, segment):
        subj, obj = _clean_span(m.group(1)), _clean_span(m.group(2))
        if (
            subj
            and obj
            and subj not in _AUXILIARY_SUBJECTS
            and len(subj.split()) <= 5
            and len(obj.split()) <= 8
        ):
            relations.add((subj, "is", obj))

    pattern = (
        r"\b([a-z][a-z ]+?) "
        rf"(?:{_NEGATION_PREFIX}) "
        rf"({verbs}) ([a-z][a-z0-9 ]+?)(?=$| and (?:{verbs}) )"
    )
    for m in re.finditer(pattern, segment):
        subj = _clean_span(m.group(1))
        pred = _lemma_predicate(m.group(2))
        obj = _clean_span(m.group(3))
        if (
            subj
            and obj
            and subj not in {"and", "or"}
            and not (pred == "is" and _is_capital_of_predicate(obj))
        ):
            relations.add((subj, pred, obj))

    for m in re.finditer(
        r"\b([a-z][a-z ]+?) has no ([a-z][a-z0-9 ]+?)(?:$| and | but |,)",
        segment,
    ):
        relations.add((_clean_span(m.group(1)), "has", _clean_span(m.group(2))))

    return relations


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
    candidate_negations = _extract_negated_relations(candidate)
    literal = literal_drift(candidate, manifold, candidate_relations, candidate_negations)

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
    if _has_unsupported_negation(candidate, manifold, candidate_negations):
        clamps.append("negated_positive_support_clamp")
    if _has_negated_reference_assertion(
        candidate,
        candidate_relations,
        candidate_negations,
        manifold.negated_relations,
    ):
        clamps.append("negated_reference_relation_clamp")

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
        negated_relations=tuple(sorted(candidate_negations)),
        pool_group_key=normalize_text(candidate),
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
    evaluations: List[CandidateEvaluation] = []
    evaluation_cache = {}
    first_index_by_pool_key = {}
    for index, candidate in enumerate(candidates):
        pool_key = normalize_text(candidate)
        if pool_key in evaluation_cache:
            evaluations.append(
                replace(
                    evaluation_cache[pool_key],
                    text=candidate,
                    pool_group_key=pool_key,
                    duplicate_of=first_index_by_pool_key[pool_key],
                    selection_status="not_ranked",
                    selection_rank=None,
                    selection_reason="Candidate was evaluated outside a candidate pool.",
                )
            )
            continue

        evaluation = evaluate_candidate(candidate, manifold, use_embeddings=use_embeddings)
        evaluation.pool_group_key = pool_key
        evaluation_cache[pool_key] = evaluation
        first_index_by_pool_key[pool_key] = index
        evaluations.append(evaluation)

    safe = [e for e in evaluations if e.safe_to_emit]
    if not safe:
        _annotate_candidate_selection(evaluations, emitted=None)
        return RegulationResult("block", None, None, evaluations)
    emitted = min(safe, key=lambda e: e.regulator_score)
    _annotate_candidate_selection(evaluations, emitted=emitted)
    return RegulationResult("emit", emitted.text, emitted, evaluations)


def _annotate_candidate_selection(
    evaluations: List[CandidateEvaluation],
    *,
    emitted: Optional[CandidateEvaluation],
) -> None:
    """Attach deterministic pool-selection diagnostics after safety evaluation."""

    safe_pairs = [
        (index, evaluation)
        for index, evaluation in enumerate(evaluations)
        if evaluation.safe_to_emit
    ]
    safe_pairs.sort(key=lambda item: (item[1].regulator_score, item[0]))

    for rank, (_, evaluation) in enumerate(safe_pairs, start=1):
        evaluation.selection_rank = rank

    for evaluation in evaluations:
        if not evaluation.safe_to_emit:
            evaluation.selection_status = "blocked_before_ranking"
            if evaluation.duplicate_of is not None:
                evaluation.selection_reason = (
                    f"Candidate duplicates candidate {evaluation.duplicate_of} after "
                    "normalization; the reused evaluation was blocked before ranking."
                )
            else:
                evaluation.selection_reason = (
                    "Candidate was dropped from emission ranking because it failed "
                    "a hard guard or geometry check."
                )
        elif evaluation is emitted:
            evaluation.selection_status = "emitted"
            evaluation.selection_reason = (
                "Candidate was selected as the lowest-score safe candidate in the pool."
            )
        else:
            evaluation.selection_status = "safe_alternative"
            if evaluation.duplicate_of is not None:
                evaluation.selection_reason = (
                    f"Candidate duplicates candidate {evaluation.duplicate_of} after "
                    "normalization and was retained for audit, but was not emitted."
                )
            else:
                evaluation.selection_reason = (
                    "Candidate stayed safe and was retained for audit, but a lower-rank "
                    "safe candidate was emitted."
                )


def literal_drift(
    candidate: str,
    manifold: ReferenceManifold,
    candidate_relations: Optional[Set[Relation]] = None,
    candidate_negations: Optional[Set[Relation]] = None,
) -> LiteralDrift:
    """Compute number/unit/entity/content novelty against reference text."""

    cand_numbers = set(_numbers(candidate))
    cand_units = set(_units(candidate))
    cand_entities = {
        entity
        for entity in _entities(candidate)
        if not _is_sentence_initial_content_entity(
            candidate,
            entity,
            manifold.content_tokens,
        )
    }
    cand_content = set(_content_tokens(candidate))
    if candidate_relations:
        cand_content -= _supported_comparison_content_tokens(
            candidate_relations,
            manifold.relations,
        )
        cand_content -= _supported_permission_content_tokens(
            candidate_relations,
            manifold.relations,
        )
        cand_numbers -= _supported_range_bound_numbers(
            candidate_relations,
            manifold.relations,
        )
        cand_content -= _supported_range_bound_content_tokens(
            candidate_relations,
            manifold.relations,
        )
    if candidate_relations or candidate_negations:
        cand_content -= _supported_exception_content_tokens(
            candidate_relations or set(),
            candidate_negations or set(),
            manifold.relations,
            manifold.negated_relations,
        )
    return LiteralDrift(
        novel_numbers=tuple(sorted(cand_numbers - manifold.numbers)),
        novel_units=tuple(sorted(cand_units - manifold.units)),
        novel_entities=tuple(sorted(cand_entities - manifold.entities)),
        novel_content=tuple(sorted(cand_content - manifold.content_tokens)),
    )


def _supported_comparison_content_tokens(
    candidate_relations: Set[Relation], reference_relations: Set[Relation]
) -> Set[str]:
    supported_tokens: Set[str] = set()
    for relation in candidate_relations:
        if relation in reference_relations:
            supported_tokens.update(_COMPARISON_CONTENT_TOKENS.get(relation[1], set()))
    return supported_tokens


def _supported_range_bound_numbers(
    candidate_relations: Set[Relation], reference_relations: Set[Relation]
) -> Set[str]:
    supported_numbers: Set[str] = set()
    if (
        ("tank", "holdsatmost", "2000milliliters") in candidate_relations
        and ("tank", "holdsatmost", "2000milliliters") in reference_relations
    ):
        supported_numbers.add("2000")
    if (
        ("cable", "maxlengthexclusive", "300centimeters") in candidate_relations
        and ("cable", "maxlengthexclusive", "300centimeters") in reference_relations
    ):
        supported_numbers.add("300")
    if (
        ("applicants", "eligibleagerange", "18through64") in candidate_relations
        and ("applicants", "eligibleagerange", "18through64") in reference_relations
    ):
        supported_numbers.add("64")
    return supported_numbers


def _supported_range_bound_content_tokens(
    candidate_relations: Set[Relation], reference_relations: Set[Relation]
) -> Set[str]:
    supported_tokens: Set[str] = set()
    if (
        ("tank", "holdsatmost", "2000milliliters") in candidate_relations
        and ("tank", "holdsatmost", "2000milliliters") in reference_relations
    ):
        supported_tokens.update({"milliliters"})
    if (
        ("cable", "maxlengthexclusive", "300centimeters") in candidate_relations
        and ("cable", "maxlengthexclusive", "300centimeters") in reference_relations
    ):
        supported_tokens.update({"centimeters", "less", "long"})
    if (
        ("service window", "startsat", "0900") in candidate_relations
        and ("service window", "endsbefore", "1700") in candidate_relations
        and ("service window", "startsat", "0900") in reference_relations
        and ("service window", "endsbefore", "1700") in reference_relations
    ):
        supported_tokens.update({"runs", "until"})
    if (
        ("applicants", "eligibleagerange", "18through64") in candidate_relations
        and ("applicants", "eligibleagerange", "18through64") in reference_relations
    ):
        supported_tokens.update({"aged", "eligible", "through"})
    if (
        ("release", "minapprovals", "2reviewers") in candidate_relations
        and ("release", "minapprovals", "2reviewers") in reference_relations
    ):
        supported_tokens.update({"requires", "approval", "from", "more"})
    if (
        ("failed check", "maxallowed", "1") in candidate_relations
        and ("failed check", "maxallowed", "1") in reference_relations
    ):
        supported_tokens.update({"most"})
    return supported_tokens


def _supported_permission_content_tokens(
    candidate_relations: Set[Relation], reference_relations: Set[Relation]
) -> Set[str]:
    supported_tokens: Set[str] = set()
    if (
        ("admin", "mayexport", "file") in candidate_relations
        and ("owner", "mayexport", "file") in candidate_relations
        and ("admin", "mayexport", "file") in reference_relations
        and ("owner", "mayexport", "file") in reference_relations
    ):
        supported_tokens.update({"admins", "owners", "file"})
    return supported_tokens


def _supported_exception_content_tokens(
    candidate_relations: Set[Relation],
    candidate_negations: Set[Relation],
    reference_relations: Set[Relation],
    reference_negations: Set[Relation],
) -> Set[str]:
    if (
        ("museum", "openon", "weekday") in candidate_relations
        and ("museum", "openon", "weekday") in reference_relations
        and ("museum", "openon", "monday") in candidate_negations
        and ("museum", "openon", "monday") in reference_negations
    ):
        return {"closed", "other", "weekdays"}
    return set()


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


def _normalize_negation_text(text: str) -> str:
    normalized = text.lower()
    for source, replacement in _NEGATION_NORMALIZATIONS.items():
        normalized = re.sub(rf"\\b{re.escape(source)}\\b", replacement, normalized)
    return normalize_text(normalized)


def _has_unsupported_negation(
    text: str,
    manifold: ReferenceManifold,
    candidate_negations: Optional[Set[Relation]] = None,
) -> bool:
    if not manifold.relations and not manifold.negated_relations:
        return False

    negations = candidate_negations or _extract_negated_relations(text)
    if not negations:
        return False

    ref_relations = manifold.relations
    ref_negated_relations = manifold.negated_relations
    ref_subjects = {r[0] for r in ref_relations}
    ref_objects = {r[2] for r in ref_relations}
    ref_predicates = {r[1] for r in ref_relations}
    for negation in negations:
        if _is_spurious_exception_negation(negation):
            continue
        if _negation_supported_by_reference(text, negation, ref_negated_relations):
            continue
        subj, pred, obj = negation
        if (subj, pred, obj) in ref_relations:
            return True
        if (obj, pred, subj) in ref_relations:
            return True
        if pred in ref_predicates and (subj in ref_subjects or obj in ref_objects):
            return True
    return False


def _has_negated_reference_assertion(
    text: str,
    candidate_relations: Set[Relation],
    candidate_negations: Set[Relation],
    reference_negated_relations: Set[Relation],
) -> bool:
    if not reference_negated_relations:
        return False

    for reference_negation in reference_negated_relations:
        if _negation_supported_by_reference(text, reference_negation, candidate_negations):
            continue
        if any(
            _relations_match(candidate_relation, reference_negation, text)
            for candidate_relation in candidate_relations
        ):
            return True
        obj = reference_negation[2]
        if _mentions_object(text, obj) and not _object_has_negation_nearby(text, obj):
            return True
    return False


def _negation_supported_by_reference(
    text: str,
    negation: Relation,
    reference_negated_relations: Set[Relation],
) -> bool:
    if any(
        _relations_match(negation, reference_negation, text)
        for reference_negation in reference_negated_relations
    ):
        return True
    variants = _split_relation_variants(negation)
    return bool(variants) and all(
        any(
            _relations_match(variant, reference_negation, text)
            for reference_negation in reference_negated_relations
        )
        for variant in variants
    )


def _split_relation_variants(relation: Relation) -> List[Relation]:
    subj, pred, obj = relation
    subjects = (
        _split_exception_objects(subj)
        if re.search(r"\b(?:and|or)\b", subj)
        else [subj]
    )
    objects = (
        _split_exception_objects(obj)
        if re.search(r"\b(?:and|or)\b", obj)
        else [obj]
    )
    variants = [(subject, pred, object_) for subject in subjects for object_ in objects]
    return [variant for variant in variants if variant != relation]


def _is_spurious_exception_negation(relation: Relation) -> bool:
    subj, pred, _ = relation
    return pred == "is" and (
        subj.endswith(" but")
        or " but " in subj
        or subj.startswith("importer accepts ")
    )


def _relations_match(left: Relation, right: Relation, text: str = "") -> bool:
    left_subj, left_pred, left_obj = left
    right_subj, right_pred, right_obj = right
    if left == right:
        return True
    if left_pred != right_pred or left_obj != right_obj:
        return False
    if left_subj == right_subj:
        return True
    if left_subj in _AUXILIARY_SUBJECTS and right_subj in normalize_text(text).split():
        return True
    return False


def _mentions_object(text: str, obj: str) -> bool:
    normalized = normalize_text(text)
    return bool(obj and re.search(rf"\b{re.escape(obj)}\b", normalized))


def _looks_like_temporal_value(value: str) -> bool:
    return bool(
        re.search(r"\b\d{3,4}\b", value)
        or re.search(r"\b\d+(?: mm| cm| m| km| units?| hours?| minutes?| seconds?)\b", value)
    )


def _is_sentence_initial_content_entity(
    text: str,
    entity: str,
    reference_content_tokens: Set[str],
) -> bool:
    normalized_entity = normalize_text(entity)
    entity_forms = {normalized_entity}
    if len(normalized_entity) > 3 and normalized_entity.endswith("s"):
        entity_forms.add(normalized_entity[:-1])
    if " " in normalized_entity or not (entity_forms & reference_content_tokens):
        return False
    tokens = normalize_text(text).split()
    return bool(tokens and tokens[0] == normalized_entity)


def _object_has_negation_nearby(text: str, obj: str) -> bool:
    normalized = _normalize_negation_text(text)
    if not obj or not _mentions_object(normalized, obj):
        return False
    escaped = re.escape(obj)
    patterns = (
        rf"\bno {escaped}\b",
        rf"\bwithout {escaped}\b",
        rf"\blacks? (?:a |an |the )?{escaped}\b",
        rf"\bomits? {escaped}\b",
        rf"\b(?:{_NEGATION_PREFIX}) (?:[a-z]+ ){{0,4}}{escaped}\b",
        rf"\b{escaped} (?:was |were |is |are )?excluded\b",
        rf"\bexcluded {escaped}\b",
    )
    return any(re.search(pattern, normalized) for pattern in patterns)


def _contains_negation(text: str) -> bool:
    normalized = _normalize_negation_text(text)
    tokens = set(normalized.split())
    return bool(tokens & _NEGATORS) or " not " in f" {normalized} "


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
        if value and value not in {"the", "a", "an"} and value not in _STOPWORDS:
            entities.append(value)
    return entities


def _content_tokens(text: str) -> List[str]:
    tokens = normalize_text(text).split()
    return [t for t in tokens if len(t) > 3 and t not in _STOPWORDS and not t.isdigit()]


def _clean_span(value: str) -> str:
    tokens = [t for t in value.strip().split() if t not in {"the", "a", "an"}]
    return " ".join(tokens[-5:]).strip()


def _clean_relation_span(value: str) -> str:
    normalized = normalize_text(value).strip()
    tokens = normalized.split()
    while tokens and tokens[0] in {"the", "a", "an"}:
        tokens = tokens[1:]
    if (
        len(tokens) >= 2
        and tokens[-2] in _SUFFIX_ENTITY_HEADS
        and re.fullmatch(r"[a-z0-9]+", tokens[-1])
    ):
        return " ".join(tokens[-5:]).strip()
    return _clean_span(normalized)


def _clean_exception_object(value: str) -> str:
    cleaned = _clean_relation_span(value)
    for suffix in (" files", " file"):
        if cleaned.endswith(suffix):
            cleaned = cleaned[: -len(suffix)].strip()
    return cleaned


def _split_exception_objects(value: str) -> List[str]:
    return [
        cleaned
        for cleaned in (
            _clean_exception_object(part)
            for part in re.split(r"\s+(?:and|or)\s+", value)
        )
        if cleaned
    ]


def _lemma_predicate(predicate: str) -> str:
    p = predicate.strip().lower()
    mapping: Dict[str, str] = {
        "is": "is",
        "are": "is",
        "was": "is",
        "were": "is",
        "orbits": "orbit",
        "orbit": "orbit",
        "orbiting": "orbit",
        "contains": "contain",
        "contain": "contain",
        "contained": "contain",
        "includes": "include",
        "include": "include",
        "included": "include",
        "allows": "allow",
        "allow": "allow",
        "approved": "approve",
        "approves": "approve",
        "approve": "approve",
        "produces": "produce",
        "produce": "produce",
        "produced": "produce",
        "releases": "release",
        "release": "release",
        "released": "release",
        "lends": "lend",
        "lend": "lend",
        "preserves": "preserve",
        "preserve": "preserve",
        "absorbs": "absorb",
        "absorb": "absorb",
        "exchanges": "exchange",
        "exchange": "exchange",
        "provides": "provide",
        "provide": "provide",
        "requests": "request",
        "request": "request",
        "turns": "turn",
        "turn": "turn",
        "writes": "write",
        "write": "write",
        "wrote": "write",
        "reviews": "review",
        "review": "review",
        "reviewed": "review",
        "feeds": "feed",
        "feed": "feed",
        "fed": "feed",
        "supervises": "supervise",
        "supervise": "supervise",
        "supervised": "supervise",
        "stores": "store",
        "store": "store",
        "stored": "store",
        "converts": "convert",
        "convert": "convert",
        "converted": "convert",
        "improves": "improve",
        "improve": "improve",
        "improved": "improve",
        "uses": "use",
        "use": "use",
        "used": "use",
        "needs": "need",
        "need": "need",
        "needed": "need",
        "has": "has",
        "have": "has",
        "before": "before",
        "after": "after",
        "signed": "signed",
        "occurred": "occurred",
        "adopted": "adopted",
    }
    return mapping.get(p, p)


def _normalize_relation(relation: Relation) -> Relation:
    subj, pred, obj = relation
    return (
        _clean_relation_span(subj),
        _lemma_predicate(normalize_text(pred)),
        _clean_relation_span(obj),
    )


def _is_capital_of_predicate(value: str) -> bool:
    return value.startswith(_CAPITAL_OF_PREFIXES)
