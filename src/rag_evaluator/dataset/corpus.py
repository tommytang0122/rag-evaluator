from __future__ import annotations

import json
from pathlib import Path

import httpx

from rag_evaluator.dataset.models import CorpusPage
from rag_evaluator.eval.retrieval import normalize_document


class CorpusError(Exception):
    pass


FLOW_TO_TYPE = {
    "portrait_table": "table_text",
    "table": "table_figure",
    "image": "table_figure",
}


class Corpus:
    def __init__(self, pages: list[CorpusPage]):
        self.pages = pages
        self._index: dict[tuple[str, str, int], CorpusPage] = {}
        for p in pages:
            key = (p.collection, normalize_document(p.document), p.page)
            existing = self._index.get(key)
            if existing is not None:
                same_page = (
                    existing.file_hash is not None
                    and existing.file_hash == p.file_hash
                )
                if not same_page:
                    raise CorpusError(
                        f"document name collision after normalization: {key} "
                        "(add file_hash to disambiguate, or dedupe the corpus)"
                    )
            self._index[key] = p

    def lookup(
        self, document: str, page: int, collection: str | None = None
    ) -> CorpusPage | None:
        nd = normalize_document(document)
        if collection is not None:
            return self._index.get((collection, nd, page))
        matches = [
            p for (c, d, pg), p in self._index.items() if d == nd and pg == page
        ]
        if len(matches) > 1:
            raise CorpusError(
                f"ambiguous lookup {nd}#p{page}: found in multiple collections"
            )
        return matches[0] if matches else None


def load_corpus(path: Path) -> Corpus:
    pages: list[CorpusPage] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                pages.append(CorpusPage.model_validate(json.loads(line)))
    return Corpus(pages)


def write_corpus(pages: list[CorpusPage], path: Path) -> None:
    with Path(path).open("w", encoding="utf-8") as fh:
        for p in pages:
            fh.write(p.model_dump_json(exclude_none=True) + "\n")


def _payload_to_page(collection: str, point_id: object, payload: dict) -> CorpusPage:
    if "source" not in payload or "page" not in payload:
        raise CorpusError(
            f"qdrant point {point_id} in {collection} lacks source/page in payload"
        )
    content = payload.get("content")
    schema_text = payload.get("schema_text")
    if content:
        text, text_source = content, "content"
    elif schema_text:
        text, text_source = schema_text, "schema_text"
    else:
        text, text_source = "", "none"
    return CorpusPage(
        collection=collection,
        document=payload["source"],
        page=payload["page"],
        text=text,
        text_source=text_source,
        type=payload.get("type"),
        file_hash=payload.get("file_hash"),
        image_path=payload.get("image_path"),
    )


def fetch_qdrant_pages(
    base_url: str,
    collections: list[str],
    client: httpx.Client | None = None,
    page_size: int = 256,
) -> list[CorpusPage]:
    """Scroll 每個 collection 的 Qdrant REST API,把 point payload 轉成 CorpusPage。

    payload 由 nas-rag uploader 寫入(source/page/type/file_hash 必有,
    image_path/schema_text/content 視頁面型態而定),是比 verification manifest
    更乾淨的語料來源:無 append 殘留、type 已是最終值。
    """
    client = client or httpx.Client(timeout=30)
    pages: list[CorpusPage] = []
    for collection in collections:
        offset = None
        while True:
            body: dict = {
                "limit": page_size,
                "with_payload": True,
                "with_vectors": False,
            }
            if offset is not None:
                body["offset"] = offset
            try:
                resp = client.post(
                    f"{base_url}/collections/{collection}/points/scroll", json=body
                )
                resp.raise_for_status()
            except httpx.HTTPError as e:
                raise CorpusError(
                    f"qdrant scroll failed for collection {collection}: {e}"
                ) from e
            result = resp.json()["result"]
            for point in result["points"]:
                pages.append(
                    _payload_to_page(collection, point.get("id"), point.get("payload") or {})
                )
            offset = result.get("next_page_offset")
            if offset is None:
                break
    return pages


def convert_nas_rag_manifest(manifest_path: Path) -> list[CorpusPage]:
    pages: list[CorpusPage] = []
    with Path(manifest_path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            content = row.get("content")
            schema_text = row.get("schema_text")
            if content:
                text, text_source = content, "content"
            elif schema_text:
                text, text_source = schema_text, "schema_text"
            else:
                text, text_source = "", "none"
            pages.append(
                CorpusPage(
                    collection=row["collection"],
                    document=row["source"],
                    page=row["page"],
                    text=text,
                    text_source=text_source,
                    type=FLOW_TO_TYPE.get(row.get("flow", "image")),
                    file_hash=row.get("file_hash"),
                    image_path=row.get("image_path"),
                )
            )
    return pages
