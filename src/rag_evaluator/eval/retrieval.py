from __future__ import annotations

import unicodedata
from collections.abc import Sequence
from pathlib import PurePosixPath

from rag_evaluator.adapters.base import SourceRef
from rag_evaluator.dataset.models import EvidenceRef


def normalize_document(name: str) -> str:
    name = unicodedata.normalize("NFC", name).replace("\\", "/")
    return PurePosixPath(name).stem.casefold()


def refs_match(evidence: EvidenceRef, source: SourceRef) -> bool:
    if evidence.file_hash and source.file_hash:
        return evidence.file_hash == source.file_hash and evidence.page == source.page
    if evidence.page != source.page:
        return False
    if (
        evidence.collection
        and source.collection
        and evidence.collection != source.collection
    ):
        return False
    return normalize_document(evidence.document) == normalize_document(source.document)


def evidence_hits(
    evidence: Sequence[EvidenceRef], sources: Sequence[SourceRef]
) -> list[bool]:
    return [any(refs_match(e, s) for s in sources) for e in evidence]


def hit_at_k(
    evidence: Sequence[EvidenceRef], sources: Sequence[SourceRef], k: int
) -> bool:
    return any(any(refs_match(e, s) for e in evidence) for s in sources[:k])


def evidence_recall(
    evidence: Sequence[EvidenceRef], sources: Sequence[SourceRef]
) -> float:
    hits = evidence_hits(evidence, sources)
    return sum(hits) / len(hits) if hits else 0.0


def all_evidence_hit(
    evidence: Sequence[EvidenceRef], sources: Sequence[SourceRef]
) -> bool:
    return bool(evidence) and all(evidence_hits(evidence, sources))


def mrr(evidence: Sequence[EvidenceRef], sources: Sequence[SourceRef]) -> float:
    for i, s in enumerate(sources):
        if any(refs_match(e, s) for e in evidence):
            return 1.0 / (i + 1)
    return 0.0


def citation_precision(
    evidence: Sequence[EvidenceRef], sources: Sequence[SourceRef]
) -> float:
    if not sources:
        return 0.0
    cited = sum(1 for s in sources if any(refs_match(e, s) for e in evidence))
    return cited / len(sources)
