from __future__ import annotations

import json
from pathlib import Path

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
            if existing is not None and existing.file_hash != p.file_hash:
                raise CorpusError(f"document name collision after normalization: {key}")
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
