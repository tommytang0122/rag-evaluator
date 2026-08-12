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


def test_refusal_item_mixed_fabrication_judged_hallucinated():
    judge = FakeJudge(CorrectnessVerdict(score=0, reason="給出了編造數字"))
    item = _item(
        answer_type="refusal", gold_answer="", gold_value=None, evidence=[],
        tags=["unanswerable"],
    )
    r = score_correctness(item, "找不到原始資料,但依推測是 12,415 千元", judge)
    assert r.method == "judge"
    assert r.correctness == 0 and r.hallucinated_answer is True
    assert "拒答" in judge.calls[0]["prompt"]


def test_refusal_item_explained_refusal_judged_correct():
    judge = FakeJudge(CorrectnessVerdict(score=2, reason="實質拒答並說明語料範圍"))
    item = _item(
        answer_type="refusal", gold_answer="", gold_value=None, evidence=[],
        tags=["unanswerable"],
    )
    r = score_correctness(
        item, "找不到相關資訊。所提供報表為 2023年09月資料,並非詢問的月份。", judge
    )
    assert r.method == "judge"
    assert r.correctness == 2 and r.hallucinated_answer is False


def test_unknown_gold_unit_degrades_to_judge():
    judge = FakeJudge(CorrectnessVerdict(score=2, reason="一致"))
    item = _item(
        gold_answer="3.70 億美元",
        gold_value={"number": "3.7", "unit": "光年"},  # 字典外單位
    )
    r = score_correctness(item, "累計虧損 3.70 光年", judge)
    assert r.method == "judge" and r.correctness == 2
    assert r.numeric_status == "unknown_unit" and r.unit_mismatch is False


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


from rag_evaluator.adapters.base import SourceRef  # noqa: E402
from rag_evaluator.dataset.corpus import Corpus  # noqa: E402
from rag_evaluator.dataset.models import CorpusPage  # noqa: E402
from rag_evaluator.eval.generation import (  # noqa: E402
    AlignedClaim,
    ClaimExtraction,
    ImageVerdict,
    TextVerdicts,
    VerdictItem,
    score_faithfulness,
)


class ScriptedJudge:
    """Returns queued results in order; records calls."""

    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    def judge(self, prompt, schema, images=()):
        self.calls.append({"prompt": prompt, "schema": schema, "images": list(images)})
        r = self.results.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def test_faithfulness_not_applicable_for_refusal_answer():
    item = _item()
    r = score_faithfulness(item, "找不到相關資訊", [SourceRef(document="a.pdf", page=1)], None, ScriptedJudge([]))
    assert r.status == "not_applicable" and r.faithfulness is None


def test_faithfulness_no_sources_scores_zero():
    r = score_faithfulness(_item(), "營收 12,415 千元", [], None, ScriptedJudge([]))
    assert r.status == "no_sources" and r.faithfulness == 0.0


def test_faithfulness_text_only_flow():
    sources = [SourceRef(document="a.pdf", page=1, collection="hr")]
    corpus = Corpus([CorpusPage(collection="hr", document="a.pdf", page=1, text="營收 12,415 千元", text_source="content")])
    judge = ScriptedJudge([
        ClaimExtraction(claims=[
            AlignedClaim(text="營收為 12,415 千元", source_indices=[0]),
            AlignedClaim(text="營收成長 10%", source_indices=[0]),
        ]),
        TextVerdicts(verdicts=[
            VerdictItem(verdict="supported"),
            VerdictItem(verdict="unsupported"),
        ]),
    ])
    r = score_faithfulness(_item(), "營收 12,415 千元,成長 10%", sources, corpus, judge)
    assert r.status == "ok"
    assert r.faithfulness == 0.5 and r.total_claims == 2 and r.evaluable == 2


def test_faithfulness_insufficient_escalates_to_image(tmp_path):
    img = tmp_path / "page_1.png"
    img.write_bytes(b"png")
    sources = [SourceRef(document="a.pdf", page=1, collection="hr", image_path=str(img))]
    corpus = Corpus([CorpusPage(collection="hr", document="a.pdf", page=1, text="摘要", text_source="schema_text")])
    judge = ScriptedJudge([
        ClaimExtraction(claims=[AlignedClaim(text="表格顯示 12,415", source_indices=[0])]),
        TextVerdicts(verdicts=[VerdictItem(verdict="insufficient")]),
        ImageVerdict(verdict="supported"),
    ])
    r = score_faithfulness(_item(), "表格顯示 12,415", sources, corpus, judge)
    assert r.status == "ok" and r.faithfulness == 1.0
    assert judge.calls[-1]["images"] == [img]


def test_faithfulness_multipage_claim_sends_all_images(tmp_path):
    img1 = tmp_path / "page_1.png"
    img1.write_bytes(b"png1")
    img2 = tmp_path / "page_2.png"
    img2.write_bytes(b"png2")
    sources = [
        SourceRef(document="a.pdf", page=1, collection="hr", image_path=str(img1)),
        SourceRef(document="a.pdf", page=2, collection="hr", image_path=str(img2)),
    ]
    corpus = Corpus([
        CorpusPage(collection="hr", document="a.pdf", page=1, text="摘要1", text_source="schema_text"),
        CorpusPage(collection="hr", document="a.pdf", page=2, text="摘要2", text_source="schema_text"),
    ])
    judge = ScriptedJudge([
        ClaimExtraction(claims=[AlignedClaim(text="跨頁表格顯示 12,415", source_indices=[0, 1])]),
        TextVerdicts(verdicts=[VerdictItem(verdict="insufficient")]),
        ImageVerdict(verdict="supported"),
    ])
    r = score_faithfulness(_item(), "跨頁表格顯示 12,415", sources, corpus, judge)
    assert r.status == "ok" and r.faithfulness == 1.0
    assert judge.calls[-1]["images"] == [img1, img2]


def test_faithfulness_textless_claim_without_image_is_unavailable():
    sources = [SourceRef(document="a.pdf", page=1)]  # no content/schema/image, no corpus
    judge = ScriptedJudge([
        ClaimExtraction(claims=[AlignedClaim(text="X", source_indices=[0])]),
    ])
    r = score_faithfulness(_item(), "有一個論斷 42", sources, None, judge)
    assert r.status == "skipped" and r.faithfulness is None
    assert r.evidence_unavailable == 1 and r.evaluable == 0


def test_faithfulness_verdict_length_mismatch_is_judge_error():
    sources = [SourceRef(document="a.pdf", page=1, content="text")]
    judge = ScriptedJudge([
        ClaimExtraction(claims=[AlignedClaim(text="A"), AlignedClaim(text="B")]),
        TextVerdicts(verdicts=[VerdictItem(verdict="supported")]),  # wrong length
    ])
    r = score_faithfulness(_item(), "回答 42 元", sources, None, judge)
    assert r.status == "judge_error" and r.faithfulness is None
