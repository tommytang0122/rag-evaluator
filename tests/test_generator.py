import csv
import json
import random

import pytest

from rag_evaluator.dataset.corpus import Corpus
from rag_evaluator.dataset.models import CorpusPage, load_dataset
from rag_evaluator.dataset.generator import (
    GeneratedQA,
    GeneratedQASet,
    finalize_dataset,
    generate_review_rows,
    sample_pages,
    write_review_csv,
)


class ScriptedGenJudge:
    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    def judge(self, prompt, schema, images=()):
        self.calls.append({"prompt": prompt, "images": list(images)})
        return self.results.pop(0)


def _corpus(tmp_path):
    img = tmp_path / "page_2.png"
    img.write_bytes(b"png")
    return Corpus([
        CorpusPage(collection="hr", document="a.pdf", page=1,
                   text="差旅住宿補助每日上限 2,500 元。" * 10, text_source="content"),
        CorpusPage(collection="hr", document="b.pdf", page=2,
                   text="", text_source="none", image_path=str(img)),
    ])


def test_sample_pages_covers_every_document():
    corpus = Corpus([
        CorpusPage(collection="hr", document="a.pdf", page=i) for i in range(1, 4)
    ] + [CorpusPage(collection="hr", document="b.pdf", page=1)])
    picked = sample_pages(corpus, 2, random.Random(0))
    assert {p.document for p in picked} == {"a.pdf", "b.pdf"}  # ≥1 per doc wins over cap


def test_generate_routes_text_and_vision(tmp_path):
    judge = ScriptedGenJudge([
        GeneratedQASet(items=[GeneratedQA(
            question="住宿補助上限?", gold_answer="每日 2,500 元",
            gold_value={"number": 2500, "unit": "元"}, tags=["numeric", "single-page"],
        )]),
        GeneratedQASet(items=[GeneratedQA(
            question="圖表頁的營收?", gold_answer="12,415 千元", tags=["single-page"],
        )]),
        GeneratedQASet(items=[GeneratedQA(
            question="聽起來相關但沒答案?", answer_type="refusal", tags=["unanswerable"],
        )]),
    ])
    rows = generate_review_rows(_corpus(tmp_path), judge, rng=random.Random(0))
    assert len(rows) == 3
    assert judge.calls[1]["images"]  # vision call for the textless page
    refusals = [r for r in rows if r["answer_type"] == "refusal"]
    assert refusals[0]["evidence"] == "" and refusals[0]["approved"] == ""
    answerable = [r for r in rows if r["answer_type"] == "answerable"]
    assert answerable[0]["evidence"] == "hr|a.pdf|1"


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
