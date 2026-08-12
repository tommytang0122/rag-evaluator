"""Review-CSV finalization for the golden dataset.

Question drafting is agent-driven: the `generate-golden-questions` skill
(.claude/skills/) has a vision-capable agent read the actual page images and
author questions with row-by-column verification notes, writing a review CSV
in the REVIEW_COLUMNS format below. A human fills the `approved` column, then
`rag-eval dataset finalize` turns approved rows into a validated dataset.

The former one-shot LLM generator (`rag-eval dataset generate`) was removed:
it never saw page images in practice (corpus rows lacked image_path) and
drafted gold values from lossy schema_text summaries, inventing numbers.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from rag_evaluator.dataset.models import DatasetItem

REVIEW_COLUMNS = [
    "question", "gold_answer", "gold_value", "answer_type", "tags",
    "evidence", "source_excerpt", "image_path", "generation_basis", "approved",
]


def write_review_csv(rows: list[dict], path: Path) -> None:
    with Path(path).open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _parse_evidence(cell: str) -> list[dict]:
    out = []
    for part in cell.split(";"):
        part = part.strip()
        if not part:
            continue
        collection, document, page = part.split("|")
        out.append({"collection": collection, "document": document, "page": int(page)})
    return out


def finalize_dataset(review_path: Path, out_path: Path) -> int:
    approved_values = {"yes", "y", "true", "1"}
    items: list[DatasetItem] = []
    seen: set[str] = set()
    with Path(review_path).open(encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("approved", "").strip().casefold() not in approved_values:
                continue
            question = row["question"].strip()
            qid = "q-" + hashlib.sha256(question.encode("utf-8")).hexdigest()[:8]
            if qid in seen:
                raise ValueError(f"duplicate question (qid {qid}): {question}")
            seen.add(qid)
            gold_value = (
                json.loads(row["gold_value"]) if row.get("gold_value", "").strip() else None
            )
            items.append(
                DatasetItem.model_validate(
                    {
                        "id": qid,
                        "question": question,
                        "gold_answer": row.get("gold_answer", ""),
                        "gold_value": gold_value,
                        "answer_type": row["answer_type"],
                        "tags": [t.strip() for t in row.get("tags", "").split(",") if t.strip()],
                        "evidence": _parse_evidence(row.get("evidence", "")),
                    }
                )
            )
    with Path(out_path).open("w", encoding="utf-8") as fh:
        for item in items:
            fh.write(item.model_dump_json(exclude_none=True) + "\n")
    return len(items)
