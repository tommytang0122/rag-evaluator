from rag_evaluator.adapters.base import SourceRef
from rag_evaluator.dataset.models import EvidenceRef
from rag_evaluator.eval.retrieval import (
    all_evidence_hit,
    citation_precision,
    evidence_recall,
    hit_at_k,
    mrr,
    normalize_document,
    refs_match,
)


def test_normalize_document():
    # Windows path, extension, case, NFC all normalized away
    assert (
        normalize_document("C:\\data\\營收月報.PDF")
        == normalize_document("/mnt/nas/營收月報.pdf")
        == "營收月報"
    )


def test_refs_match_by_name_page_collection():
    e = EvidenceRef(document="營收月報.pdf", page=3, collection="hr")
    assert refs_match(e, SourceRef(document="/abs/營收月報.pdf", page=3, collection="hr"))
    assert not refs_match(e, SourceRef(document="/abs/營收月報.pdf", page=4, collection="hr"))
    assert not refs_match(e, SourceRef(document="/abs/營收月報.pdf", page=3, collection="fin"))
    # collection missing on either side → not part of the key
    assert refs_match(e, SourceRef(document="營收月報.pdf", page=3))


def test_refs_match_prefers_file_hash():
    e = EvidenceRef(document="x.pdf", page=3, file_hash="h1")
    assert refs_match(e, SourceRef(document="renamed.pdf", page=3, file_hash="h1"))
    assert not refs_match(e, SourceRef(document="x.pdf", page=3, file_hash="h2"))


EV = [EvidenceRef(document="a.pdf", page=1), EvidenceRef(document="b.pdf", page=2)]
SRC = [
    SourceRef(document="z.pdf", page=9),
    SourceRef(document="a.pdf", page=1),
    SourceRef(document="c.pdf", page=5),
]


def test_metrics():
    assert hit_at_k(EV, SRC, 1) is False
    assert hit_at_k(EV, SRC, 3) is True
    assert evidence_recall(EV, SRC) == 0.5          # only a.pdf#1 found
    assert all_evidence_hit(EV, SRC) is False
    assert mrr(EV, SRC) == 0.5                       # first hit at rank 2
    assert citation_precision(EV, SRC) == 1 / 3


def test_metrics_empty_inputs():
    assert mrr(EV, []) == 0.0
    assert citation_precision(EV, []) == 0.0
    assert evidence_recall([], SRC) == 0.0
    assert all_evidence_hit([], SRC) is False
