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


LOST_TO_CUTOFF = "lost_to_cutoff"
RANKED_BELOW_TOP_K = "ranked_below_top_k"
NOT_IN_PROBE = "not_in_probe"


def attribute_misses(
    evidence: Sequence[EvidenceRef],
    sources: Sequence[SourceRef],
    probe_sources: Sequence[SourceRef],
    top_k: int,
) -> dict[str, str]:
    """依 probe 名次歸因未命中的 gold 頁。

    reranker 對每篇文件的打分與 top_n 無關,故 probe(較大 top_k)中的名次
    可判別:名次 ≤ top_k 表示本應入選卻被最大斷層截掉;> top_k 表示排名太低。
    probe 回應自身也經過截斷,lost_to_cutoff 為保守低估。
    """
    out: dict[str, str] = {}
    for e in evidence:
        if any(refs_match(e, s) for s in sources):
            continue
        key = f"{normalize_document(e.document)}#p{e.page}"
        rank = next(
            (i + 1 for i, s in enumerate(probe_sources) if refs_match(e, s)), None
        )
        if rank is None:
            out[key] = NOT_IN_PROBE
        elif rank <= top_k:
            out[key] = LOST_TO_CUTOFF
        else:
            out[key] = RANKED_BELOW_TOP_K
    return out
