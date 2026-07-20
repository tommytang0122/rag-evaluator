from rag_evaluator.dataset.models import DatasetItem
from rag_evaluator.report import build_report, paired_bootstrap_ci

ITEMS = [
    DatasetItem.model_validate({
        "id": "q-1", "question": "營收?", "answer_type": "answerable",
        "gold_answer": "12,415 千元", "tags": ["numeric"],
        "evidence": [{"document": "a.pdf", "page": 3}],
    }),
    DatasetItem.model_validate({
        "id": "q-2", "question": "??", "answer_type": "refusal",
        "tags": ["unanswerable"],
    }),
]


def _row(qid, run=0, **over):
    base = {
        "qid": qid, "run": run, "system_error": False, "latency_ms": 10,
        "retrieval": {"hit_at_1": True, "hit_at_3": True, "hit_at_5": True,
                      "evidence_recall": 1.0, "all_evidence_hit": True,
                      "mrr": 1.0, "citation_precision": 0.6},
        "gold_type_hits": [{"type": "table_figure", "hit": True}],
        "attribution": None,
        "correctness": 2, "method": "rule_numeric",
        "refusal_state": "substantive_answer",
        "numeric_status": "match", "numeric_canonical": "12415000",
        "unit_mismatch": False, "false_refusal": False,
        "hallucinated_answer": False, "judge_reason": None, "judge_error": False,
        "faithfulness": 1.0, "faithfulness_status": "ok",
        "faithfulness_total_claims": 1, "faithfulness_supported": 1,
        "faithfulness_evaluable": 1, "faithfulness_evidence_unavailable": 0,
        "faithfulness_evaluable_claim_rate": 1.0,
        "judge_prompt_version": "v1", "judge_prompt_hash": "h",
    }
    base.update(over)
    return base


def test_report_contains_core_sections():
    scores = [
        _row("q-1"),
        _row("q-2", retrieval=None, gold_type_hits=None, correctness=2,
             method="refusal_rule", refusal_state="pure_refusal",
             numeric_status=None, numeric_canonical=None,
             faithfulness=None, faithfulness_status="not_applicable"),
    ]
    md = build_report(scores=scores, items=ITEMS)
    assert "## 總覽" in md and "## 特化診斷" in md and "## 分項統計" in md
    assert "mean_correctness" in md and "refusal_accuracy" in md


def test_two_track_counts_system_error_as_zero():
    scores = [
        _row("q-1"),
        _row("q-1x", system_error=True, retrieval=None,
             correctness=None, method=None, faithfulness=None,
             faithfulness_status=None),
    ]
    items = ITEMS + [DatasetItem.model_validate({
        "id": "q-1x", "question": "x?", "answer_type": "answerable",
        "gold_answer": "y", "evidence": [{"document": "a.pdf", "page": 1}],
    })]
    md = build_report(scores=scores, items=items)
    # end-to-end mean correctness (2+0)/2 = 1.0 ; valid-sample = 2.0
    assert "| mean_correctness | 1.000 | 2.000 |" in md


def test_system_error_refusal_row_excluded_from_retrieval_and_faithfulness_e2e():
    # q-1 answerable, normal row: correctness=2, hit_at_1=True (via defaults)
    # q-2 refusal, system_error row: not applicable to hit_at_1/faithfulness,
    # but mean_correctness applies to every item type.
    scores = [
        _row("q-1"),
        _row("q-2", system_error=True, retrieval=None,
             correctness=None, method=None, faithfulness=None,
             faithfulness_status=None),
    ]
    md = build_report(scores=scores, items=ITEMS)
    # end-to-end hit_at_1 must ignore the errored refusal row entirely
    assert "| hit_at_1 | 1.000 | 1.000 |" in md
    # end-to-end faithfulness likewise excludes the errored refusal row
    assert "| faithfulness | 1.000 | 1.000 |" in md
    # mean_correctness still counts the errored refusal row as 0 -> (2+0)/2
    assert "| mean_correctness | 1.000 | 2.000 |" in md


def test_baseline_system_error_refusal_row_excluded_from_retrieval_diff():
    # q-1 answerable: real diff on hit_at_5 (cur True vs base False -> +1.0)
    # q-2 refusal, system_error on BOTH sides: must not contribute a
    # phantom 0.0-vs-0.0 "no difference" diff for a metric that doesn't
    # apply to refusal items, but mean_correctness still pairs 0 vs 0.
    scores = [
        _row("q-1"),
        _row("q-2", system_error=True, retrieval=None,
             correctness=None, method=None, faithfulness=None,
             faithfulness_status=None),
    ]
    baseline = [
        _row("q-1", correctness=1, faithfulness=0.5,
             retrieval={"hit_at_1": True, "hit_at_3": True, "hit_at_5": False,
                        "evidence_recall": 0.5, "all_evidence_hit": False,
                        "mrr": 0.5, "citation_precision": 0.3}),
        _row("q-2", system_error=True, retrieval=None,
             correctness=None, method=None, faithfulness=None,
             faithfulness_status=None),
    ]
    md = build_report(scores=scores, items=ITEMS, baseline=baseline)
    assert "| hit_at_5 | 1.000 |" in md
    assert "| mean_correctness | 0.500 |" in md


def test_multi_run_metrics_present_when_runs_gt_1():
    scores = [
        _row("q-1", run=0),
        _row("q-1", run=1, correctness=0, numeric_canonical="13000000"),
    ]
    md = build_report(scores=scores, items=ITEMS)
    assert "## 多次執行" in md and "success@N" in md
    assert "agreement_rate" in md


def test_multi_run_agreement_rate_catches_different_wrong_numbers():
    # Both runs are wrong (correctness=0, numeric_canonical=None, as rule
    # scoring leaves it on mismatch) but they extracted DIFFERENT numbers
    # from the answer text -> this is real answer instability and must
    # NOT be reported as perfect agreement.
    scores = [
        _row("q-1", run=0, correctness=0, numeric_status="number_mismatch",
             numeric_canonical=None, answer_numeric_signature="12415000"),
        _row("q-1", run=1, correctness=0, numeric_status="number_mismatch",
             numeric_canonical=None, answer_numeric_signature="13000000"),
    ]
    md = build_report(scores=scores, items=ITEMS)
    assert "agreement_rate:0.000" in md


def test_multi_run_agreement_rate_perfect_when_same_signature():
    scores = [
        _row("q-1", run=0, answer_numeric_signature="12415000"),
        _row("q-1", run=1, answer_numeric_signature="12415000"),
    ]
    md = build_report(scores=scores, items=ITEMS)
    assert "agreement_rate:1.000" in md


def test_baseline_paired_comparison():
    scores = [_row("q-1")]
    baseline = [_row("q-1", correctness=0, faithfulness=0.0)]
    md = build_report(scores=scores, items=ITEMS, baseline=baseline)
    assert "## Baseline 比較" in md and "mean_correctness" in md


def test_paired_bootstrap_ci_all_positive_diffs_excludes_zero():
    lo, hi = paired_bootstrap_ci([1.0, 1.0, 2.0, 1.5], iters=500, seed=1)
    assert lo > 0 and hi >= lo


def test_worst_cases_listed():
    scores = [_row("q-1", correctness=0, judge_reason="數字錯", method="judge")]
    md = build_report(scores=scores, items=ITEMS)
    assert "## 最差案例" in md and "q-1" in md


def test_faithfulness_judge_error_counted_separately_from_correctness():
    scores = [
        # rule_numeric correctness (judge_error=False) but faithfulness judge failed
        _row("q-1", method="rule_numeric", judge_error=False,
             faithfulness=None, faithfulness_status="judge_error"),
        # a normal judge-scored row whose correctness judge failed (distinct case)
        _row("q-2", method="judge", judge_error=True,
             correctness=None, judge_reason=None,
             faithfulness=1.0, faithfulness_status="ok"),
    ]
    md = build_report(scores=scores, items=ITEMS)
    assert "faithfulness judge_error 筆數:1" in md
    assert "correctness judge_error 筆數:1" in md
    assert "faithfulness_judge_coverage" in md
