from decimal import Decimal

import pytest
from pydantic import BaseModel

from rag_evaluator.dataset.models import DatasetItem
from rag_evaluator.eval.generation import (
    CORRECTNESS_PROMPT,
    CorrectnessVerdict,
    prompt_hash,
    score_correctness,
)
from rag_evaluator.judge import JudgeError


class FakeJudge:
    def __init__(self, result=None, error=False):
        self.result = result
        self.error = error
        self.calls = []

    def judge(self, prompt, schema, images=()):
        self.calls.append({"prompt": prompt, "schema": schema, "images": list(images)})
        if self.error:
            raise JudgeError("boom")
        return self.result


def _item(**over):
    base = {
        "id": "q-1",
        "question": "營收多少?",
        "answer_type": "answerable",
        "gold_answer": "12,415 千元",
        "gold_value": {"number": 12415, "unit": "千元"},
        "tags": ["numeric"],
        "evidence": [{"document": "a.pdf", "page": 1}],
    }
    base.update(over)
    return DatasetItem.model_validate(base)


def test_numeric_rule_match_scores_2_without_judge():
    judge = FakeJudge()
    r = score_correctness(_item(), "營收為 12,415 千元", judge)
    assert r.correctness == 2 and r.method == "rule_numeric"
    assert r.numeric_status == "match"
    assert judge.calls == []


def test_unit_mismatch_flagged():
    r = score_correctness(_item(), "營收為 12,415 元", FakeJudge())
    assert r.correctness == 0 and r.unit_mismatch is True
    assert r.numeric_status == "unit_mismatch"


def test_trap_tag_forces_judge_even_with_gold_value():
    judge = FakeJudge(CorrectnessVerdict(score=0, reason="廠區指錯"))
    item = _item(tags=["numeric", "company-match"])
    r = score_correctness(item, "南亞廠區營收 12,415 千元", judge)
    assert r.method == "judge" and r.correctness == 0
    assert len(judge.calls) == 1


def test_ambiguous_degrades_to_judge():
    judge = FakeJudge(CorrectnessVerdict(score=1, reason="部分正確"))
    r = score_correctness(
        _item(), "上半年 12,415 千元,下半年 13,000 千元", judge
    )
    assert r.method == "judge" and r.correctness == 1
    assert r.numeric_status == "ambiguous"


def test_answerable_pure_refusal_is_false_refusal():
    r = score_correctness(_item(), "找不到相關資訊", FakeJudge())
    assert r.correctness == 0 and r.false_refusal is True
    assert r.method == "refusal_rule"


def test_refusal_item_pure_refusal_correct():
    item = _item(
        answer_type="refusal", gold_answer="", gold_value=None, evidence=[],
        tags=["unanswerable"],
    )
    r = score_correctness(item, "找不到相關資訊", FakeJudge())
    assert r.correctness == 2 and not r.hallucinated_answer


def test_refusal_item_mixed_is_hallucinated():
    item = _item(
        answer_type="refusal", gold_answer="", gold_value=None, evidence=[],
        tags=["unanswerable"],
    )
    r = score_correctness(item, "找不到原始資料,但依推測是 12,415 千元", FakeJudge())
    assert r.correctness == 0 and r.hallucinated_answer is True


def test_narrative_goes_to_judge_without_context():
    judge = FakeJudge(CorrectnessVerdict(score=2, reason="一致"))
    item = _item(gold_value=None, tags=["single-page"], gold_answer="每日 2,500 元")
    r = score_correctness(item, "補助上限為每日兩千五百元", judge)
    assert r.correctness == 2 and r.judge_reason == "一致"
    prompt = judge.calls[0]["prompt"]
    # question, gold answer, and system answer are in the prompt — but no
    # retrieved context/sources block (Correctness judge is context-free by spec)
    assert "補助上限" in prompt and "每日 2,500 元" in prompt
    assert "來源清單" not in prompt
    assert judge.calls[0]["images"] == []


def test_judge_error_marks_row():
    judge = FakeJudge(error=True)
    item = _item(gold_value=None)
    r = score_correctness(item, "某敘述回答", judge)
    assert r.correctness is None and r.judge_error is True


def test_prompt_hash_is_stable_sha256():
    assert len(prompt_hash()) == 64
    assert "{question}" in CORRECTNESS_PROMPT
