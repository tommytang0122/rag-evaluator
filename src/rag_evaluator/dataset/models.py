from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class EvidenceRef(BaseModel):
    document: str
    page: int = Field(ge=1)
    collection: str | None = None
    file_hash: str | None = None


class GoldValue(BaseModel):
    number: Decimal
    unit: str | None = None


class DatasetItem(BaseModel):
    id: str
    question: str
    answer_type: Literal["answerable", "refusal"]
    gold_answer: str = ""
    gold_value: GoldValue | None = None
    tags: list[str] = Field(default_factory=list)
    evidence: list[EvidenceRef] = Field(default_factory=list)

    @model_validator(mode="after")
    def _refusal_is_empty(self) -> "DatasetItem":
        if self.answer_type == "refusal" and (self.evidence or self.gold_value):
            raise ValueError("refusal items must have no evidence and no gold_value")
        return self


class CorpusPage(BaseModel):
    collection: str
    document: str
    page: int = Field(ge=1)
    text: str = ""
    text_source: Literal["content", "schema_text", "none"] = "none"
    type: str | None = None
    file_hash: str | None = None
    image_path: str | None = None


def load_dataset(path: Path) -> list[DatasetItem]:
    items: list[DatasetItem] = []
    seen: set[str] = set()
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            item = DatasetItem.model_validate(json.loads(line))
            if item.id in seen:
                raise ValueError(f"duplicate dataset id: {item.id}")
            seen.add(item.id)
            items.append(item)
    return items
