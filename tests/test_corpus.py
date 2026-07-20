import json

import pytest

from rag_evaluator.dataset.models import CorpusPage
from rag_evaluator.dataset.corpus import (
    Corpus,
    CorpusError,
    convert_nas_rag_manifest,
    load_corpus,
    write_corpus,
)

MANIFEST_ROWS = [
    {  # image flow: no text at all
        "collection": "gavin_test",
        "source": "C:\\nas\\營收月報.pdf",
        "file_hash": "h-aaa",
        "page": 3,
        "image_path": "output/images/gavin_test/營收月報/page_3.png",
        "point_id": "p1",
        "flow": "image",
    },
    {  # table flow: schema_text only
        "collection": "gavin_test",
        "source": "C:\\nas\\營收月報.pdf",
        "file_hash": "h-aaa",
        "page": 4,
        "image_path": "output/images/gavin_test/營收月報/page_4.png",
        "point_id": "p2",
        "flow": "table",
        "schema_text": "【報表標題】營收月報",
    },
    {  # portrait_table flow: content wins over schema_text
        "collection": "gavin_test",
        "source": "C:\\nas\\辦法.pdf",
        "file_hash": "h-bbb",
        "page": 1,
        "image_path": "output/images/gavin_test/辦法/page_1.png",
        "point_id": "p3",
        "flow": "portrait_table",
        "schema_text": "摘要",
        "content": "第一條 本辦法依...",
    },
]


def test_convert_nas_rag_manifest(tmp_path):
    mp = tmp_path / "manifest.jsonl"
    mp.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in MANIFEST_ROWS),
        encoding="utf-8",
    )
    pages = convert_nas_rag_manifest(mp)
    assert [p.text_source for p in pages] == ["none", "schema_text", "content"]
    assert [p.type for p in pages] == ["table_figure", "table_figure", "table_text"]
    assert pages[0].file_hash == "h-aaa"
    assert pages[2].text.startswith("第一條")


def test_corpus_lookup_normalizes_document():
    pages = [
        CorpusPage(collection="hr", document="/mnt/nas/營收月報.pdf", page=3, type="table_figure")
    ]
    c = Corpus(pages)
    assert c.lookup("營收月報.PDF", 3, collection="hr") is not None
    assert c.lookup("營收月報.pdf", 3) is not None          # cross-collection search
    assert c.lookup("別的.pdf", 3, collection="hr") is None


def test_corpus_collision_detected():
    pages = [
        CorpusPage(collection="hr", document="a\\x.pdf", page=1, file_hash="h1"),
        CorpusPage(collection="hr", document="b/x.PDF", page=1, file_hash="h2"),
    ]
    with pytest.raises(CorpusError, match="collision"):
        Corpus(pages)


def test_corpus_collision_both_hashes_none_raises():
    pages = [
        CorpusPage(collection="hr", document="a/report.pdf", page=1, file_hash=None),
        CorpusPage(collection="hr", document="b/REPORT.PDF", page=1, file_hash=None),
    ]
    with pytest.raises(CorpusError, match="collision"):
        Corpus(pages)


def test_corpus_collision_one_hash_missing_raises():
    pages = [
        CorpusPage(collection="hr", document="a/report.pdf", page=1, file_hash="h1"),
        CorpusPage(collection="hr", document="b/REPORT.PDF", page=1, file_hash=None),
    ]
    with pytest.raises(CorpusError, match="collision"):
        Corpus(pages)


def test_corpus_same_hash_dedupes():
    pages = [
        CorpusPage(collection="hr", document="a/report.pdf", page=1, file_hash="h1"),
        CorpusPage(collection="hr", document="b/REPORT.PDF", page=1, file_hash="h1"),
    ]
    c = Corpus(pages)
    assert c.lookup("report.pdf", 1, collection="hr") is not None


def test_corpus_roundtrip(tmp_path):
    pages = [CorpusPage(collection="hr", document="x.pdf", page=1, text="hi", text_source="content")]
    out = tmp_path / "corpus.jsonl"
    write_corpus(pages, out)
    c = load_corpus(out)
    assert c.pages[0].text == "hi"
