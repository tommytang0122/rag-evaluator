import json
from decimal import Decimal

import pytest
from pydantic import ValidationError

from rag_evaluator.dataset.models import (
    CorpusPage,
    DatasetItem,
    EvidenceRef,
    GoldValue,
    load_dataset,
)

ITEM = {
    "id": "q-3fa9c2d1",
    "question": "2025年1月台塑廠區的營業收入是多少?",
    "gold_answer": "12,415 NTD千元",
    "gold_value": {"number": 12415, "unit": "NTD千元"},
    "answer_type": "answerable",
    "tags": ["numeric", "single-page"],
    "evidence": [{"collection": "hr", "document": "營收月報.pdf", "page": 3}],
}


def test_dataset_item_parses():
    item = DatasetItem.model_validate(ITEM)
    assert item.gold_value.number == Decimal("12415")
    assert item.evidence[0].page == 3


def test_refusal_item_must_be_empty():
    with pytest.raises(ValidationError):
        DatasetItem.model_validate(
            {**ITEM, "answer_type": "refusal"}  # keeps evidence/gold_value → invalid
        )
    ok = DatasetItem.model_validate(
        {
            "id": "q-x",
            "question": "??",
            "answer_type": "refusal",
            "tags": ["unanswerable"],
        }
    )
    assert ok.evidence == [] and ok.gold_value is None


def test_page_must_be_positive():
    with pytest.raises(ValidationError):
        EvidenceRef(document="a.pdf", page=0)
    with pytest.raises(ValidationError):
        CorpusPage(collection="hr", document="a.pdf", page=0)


def test_load_dataset_rejects_duplicate_ids(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text(
        json.dumps(ITEM, ensure_ascii=False)
        + "\n"
        + json.dumps(ITEM, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_dataset(p)


def test_load_dataset_ok(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text(json.dumps(ITEM, ensure_ascii=False) + "\n", encoding="utf-8")
    items = load_dataset(p)
    assert len(items) == 1 and items[0].id == "q-3fa9c2d1"
