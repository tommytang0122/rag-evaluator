import json

import pytest

from rag_evaluator.dataset.models import load_dataset
from rag_evaluator.dataset.generator import (
    finalize_dataset,
    write_review_csv,
)


def test_review_csv_roundtrip_and_finalize(tmp_path):
    rows = [
        {"question": "住宿補助上限?", "gold_answer": "每日 2,500 元",
         "gold_value": json.dumps({"number": 2500, "unit": "元"}),
         "answer_type": "answerable", "tags": "numeric,single-page",
         "evidence": "hr|a.pdf|1", "source_excerpt": "…", "generation_basis": "hr|a.pdf|1",
         "approved": "yes"},
        {"question": "沒被核可的題", "gold_answer": "x", "gold_value": "",
         "answer_type": "answerable", "tags": "single-page",
         "evidence": "hr|a.pdf|2", "source_excerpt": "", "generation_basis": "",
         "approved": ""},
        {"question": "不可答題", "gold_answer": "", "gold_value": "",
         "answer_type": "refusal", "tags": "unanswerable",
         "evidence": "", "source_excerpt": "", "generation_basis": "hr|a.pdf|1",
         "approved": "yes"},
    ]
    review = tmp_path / "review.csv"
    write_review_csv(rows, review)
    out = tmp_path / "dataset.jsonl"
    n = finalize_dataset(review, out)
    assert n == 2  # unapproved row dropped
    items = load_dataset(out)
    assert items[0].id.startswith("q-") and len(items[0].id) == 10
    assert items[0].gold_value.number == 2500
    assert items[0].evidence[0].collection == "hr"
    assert items[1].answer_type == "refusal" and items[1].evidence == []


def test_finalize_rejects_duplicate_questions(tmp_path):
    row = {"question": "同一題", "gold_answer": "x", "gold_value": "",
           "answer_type": "answerable", "tags": "", "evidence": "hr|a.pdf|1",
           "source_excerpt": "", "generation_basis": "", "approved": "yes"}
    review = tmp_path / "review.csv"
    write_review_csv([row, dict(row)], review)
    with pytest.raises(ValueError, match="duplicate"):
        finalize_dataset(review, tmp_path / "d.jsonl")


def test_finalize_multi_page_evidence(tmp_path):
    row = {"question": "跨頁題?", "gold_answer": "A 8,084;B 75,626", "gold_value": "",
           "answer_type": "answerable", "tags": "numeric,cross-page",
           "evidence": "hr|a.pdf|1;hr|a.pdf|4",
           "source_excerpt": "", "generation_basis": "", "approved": "yes"}
    review = tmp_path / "review.csv"
    write_review_csv([row], review)
    out = tmp_path / "d.jsonl"
    assert finalize_dataset(review, out) == 1
    items = load_dataset(out)
    assert [e.page for e in items[0].evidence] == [1, 4]
