import json

import httpx
import pytest

from rag_evaluator.dataset.models import CorpusPage
from rag_evaluator.dataset.corpus import (
    Corpus,
    CorpusError,
    convert_nas_rag_manifest,
    fetch_qdrant_pages,
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


# --- corpus from-qdrant:REST scroll → CorpusPage ---

QDRANT_PAYLOAD_FIGURE = {
    "source": "C:\\nas\\營收月報.pdf",
    "page": 3,
    "type": "table_figure",
    "file_hash": "h-aaa",
    "image_path": "output/images/gavin_test/營收月報/page_3.png",
    "schema_text": "【報表標題】營收月報",
}

QDRANT_PAYLOAD_TEXT = {
    "source": "C:\\nas\\辦法.pdf",
    "page": 1,
    "type": "table_text",
    "file_hash": "h-bbb",
    "schema_text": "摘要",
    "content": "第一條 本辦法依...",
}


def _scroll_response(points, next_page_offset=None):
    return {"result": {"points": points, "next_page_offset": next_page_offset},
            "status": "ok"}


def _qdrant_client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_fetch_qdrant_pages_maps_payload():
    requests = []

    def handler(request):
        requests.append((request.url.path, json.loads(request.content)))
        return httpx.Response(200, json=_scroll_response([
            {"id": "p1", "payload": QDRANT_PAYLOAD_FIGURE},
            {"id": "p2", "payload": QDRANT_PAYLOAD_TEXT},
        ]))

    pages = fetch_qdrant_pages(
        "http://localhost:6333", ["gavin_test"], client=_qdrant_client(handler)
    )
    path, body = requests[0]
    assert path == "/collections/gavin_test/points/scroll"
    assert body["with_payload"] is True and body["with_vectors"] is False
    assert "offset" not in body

    fig, txt = pages
    assert fig.collection == "gavin_test" and fig.document.endswith("營收月報.pdf")
    assert fig.page == 3 and fig.type == "table_figure"
    assert fig.text == "【報表標題】營收月報" and fig.text_source == "schema_text"
    assert fig.file_hash == "h-aaa" and fig.image_path.endswith("page_3.png")
    assert txt.text.startswith("第一條") and txt.text_source == "content"
    assert txt.image_path is None


def test_fetch_qdrant_pages_paginates_until_offset_exhausted():
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            return httpx.Response(200, json=_scroll_response(
                [{"id": "p1", "payload": QDRANT_PAYLOAD_FIGURE}], next_page_offset="p2"
            ))
        return httpx.Response(200, json=_scroll_response(
            [{"id": "p2", "payload": QDRANT_PAYLOAD_TEXT}]
        ))

    pages = fetch_qdrant_pages(
        "http://localhost:6333", ["gavin_test"], client=_qdrant_client(handler)
    )
    assert len(pages) == 2
    assert requests[1]["offset"] == "p2"


def test_fetch_qdrant_pages_iterates_collections():
    seen_paths = []

    def handler(request):
        seen_paths.append(request.url.path)
        return httpx.Response(200, json=_scroll_response(
            [{"id": "p1", "payload": QDRANT_PAYLOAD_TEXT}]
        ))

    pages = fetch_qdrant_pages(
        "http://localhost:6333", ["hr", "fin"], client=_qdrant_client(handler)
    )
    assert seen_paths == [
        "/collections/hr/points/scroll",
        "/collections/fin/points/scroll",
    ]
    assert [p.collection for p in pages] == ["hr", "fin"]


def test_fetch_qdrant_pages_http_error_raises_corpus_error():
    def handler(request):
        return httpx.Response(404, json={"status": {"error": "not found"}})

    with pytest.raises(CorpusError, match="gavin_test"):
        fetch_qdrant_pages(
            "http://localhost:6333", ["gavin_test"], client=_qdrant_client(handler)
        )


def test_fetch_qdrant_pages_missing_required_field_raises():
    def handler(request):
        return httpx.Response(200, json=_scroll_response(
            [{"id": "p9", "payload": {"source": "a.pdf"}}]  # page 缺漏
        ))

    with pytest.raises(CorpusError, match="p9"):
        fetch_qdrant_pages(
            "http://localhost:6333", ["gavin_test"], client=_qdrant_client(handler)
        )


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
