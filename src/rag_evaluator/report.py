from __future__ import annotations

import random
from collections import defaultdict
from statistics import mean

from rag_evaluator.dataset.models import DatasetItem


def _fmt(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.3f}"


def _safe_mean(vals: list[float]) -> float | None:
    return mean(vals) if vals else None


def paired_bootstrap_ci(
    diffs: list[float], iters: int = 2000, seed: int = 0
) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(
        mean(rng.choice(diffs) for _ in range(n)) for _ in range(iters)
    )
    lo = means[int(0.025 * iters)]
    hi = means[min(int(0.975 * iters), iters - 1)]
    return lo, hi


_QUALITY_METRICS = {
    # name → (extractor(row) -> float|None, zero value for system_error rows)
    "mean_correctness": (lambda r: r.get("correctness"), 0.0),
    "hit_at_1": (lambda r: (r.get("retrieval") or {}).get("hit_at_1"), 0.0),
    "hit_at_3": (lambda r: (r.get("retrieval") or {}).get("hit_at_3"), 0.0),
    "hit_at_5": (lambda r: (r.get("retrieval") or {}).get("hit_at_5"), 0.0),
    "evidence_recall": (lambda r: (r.get("retrieval") or {}).get("evidence_recall"), 0.0),
    "all_evidence_hit": (lambda r: (r.get("retrieval") or {}).get("all_evidence_hit"), 0.0),
    "mrr": (lambda r: (r.get("retrieval") or {}).get("mrr"), 0.0),
    "citation_precision": (lambda r: (r.get("retrieval") or {}).get("citation_precision"), 0.0),
    "faithfulness": (lambda r: r.get("faithfulness"), 0.0),
}


def _metric_values(scores: list[dict], name: str, track: str) -> list[float]:
    extract, zero = _QUALITY_METRICS[name]
    vals: list[float] = []
    for r in scores:
        if r.get("system_error"):
            if track == "end_to_end":
                vals.append(zero)
            continue
        v = extract(r)
        if v is None:
            continue  # judge_error / not applicable
        vals.append(float(v))
    return vals


def _overview(scores: list[dict], items: list[DatasetItem]) -> list[str]:
    by_id = {i.id: i for i in items}
    lines = [
        "## 總覽",
        "",
        "| 指標 | 端到端 | 有效樣本 |",
        "|---|---|---|",
    ]
    for name in _QUALITY_METRICS:
        e2e = _safe_mean(_metric_values(scores, name, "end_to_end"))
        valid = _safe_mean(_metric_values(scores, name, "valid"))
        lines.append(f"| {name} | {_fmt(e2e)} | {_fmt(valid)} |")

    ok = [r for r in scores if not r.get("system_error")]
    rule_numeric = [r for r in ok if r.get("method") == "rule_numeric"]
    answerable = [r for r in ok if by_id.get(r["qid"], None) and by_id[r["qid"]].answer_type == "answerable"]
    refusal = [r for r in ok if by_id.get(r["qid"], None) and by_id[r["qid"]].answer_type == "refusal"]
    judged = [r for r in ok if r.get("method") == "judge"]
    judge_ok = [r for r in judged if not r.get("judge_error")]
    faith_judged = [r for r in ok if r.get("faithfulness_status") in ("ok", "skipped", "judge_error")]
    faith_judge_ok = [r for r in faith_judged if r.get("faithfulness_status") != "judge_error"]

    def rate(rows: list[dict], flag: str) -> str:
        return _fmt(mean([1.0 if r.get(flag) else 0.0 for r in rows]) if rows else None)

    lines += [
        "",
        f"- unit_mismatch 率(分母=規則數值題 {len(rule_numeric)}):{rate(rule_numeric, 'unit_mismatch')}",
        f"- false_refusal 率(分母=answerable {len(answerable)}):{rate(answerable, 'false_refusal')}",
        f"- hallucinated_answer 率(分母=refusal {len(refusal)}):{rate(refusal, 'hallucinated_answer')}",
        f"- refusal_accuracy(分母=refusal {len(refusal)}):"
        + _fmt(_safe_mean([1.0 if r.get("correctness") == 2 else 0.0 for r in refusal])),
        f"- correctness_judge_coverage:{_fmt(len(judge_ok) / len(judged) if judged else None)}",
        f"- faithfulness_judge_coverage:{_fmt(len(faith_judge_ok) / len(faith_judged) if faith_judged else None)}",
        f"- faithfulness_evaluable_claim_rate 平均:"
        + _fmt(_safe_mean([r["faithfulness_evaluable_claim_rate"] for r in ok
                           if r.get("faithfulness_evaluable_claim_rate") is not None])),
        f"- 平均延遲 ms:"
        + _fmt(_safe_mean([float(r["latency_ms"]) for r in ok if r.get("latency_ms") is not None])),
        f"- system_error 筆數:{sum(1 for r in scores if r.get('system_error'))}",
        f"- correctness judge_error 筆數:{sum(1 for r in scores if r.get('judge_error'))}",
        f"- faithfulness judge_error 筆數:"
        + f"{sum(1 for r in scores if r.get('faithfulness_status') == 'judge_error')}",
    ]
    return lines


def _diagnostics(scores: list[dict]) -> list[str]:
    lines = ["", "## 特化診斷", ""]
    attributions = [a for r in scores if r.get("attribution") for a in r["attribution"].values()]
    if attributions:
        for label in ("lost_to_cutoff", "ranked_below_top_k", "not_in_probe"):
            n = sum(1 for a in attributions if a == label)
            lines.append(f"- {label}:{n}/{len(attributions)}")
    else:
        lines.append("- 截斷/排名歸因:未啟用或 adapter 不支援(無 probe 資料)")
    buckets: dict[str, list[bool]] = defaultdict(list)
    for r in scores:
        for gh in r.get("gold_type_hits") or []:
            buckets[gh["type"]].append(gh["hit"])
    if buckets:
        lines += ["", "| gold type | evidence recall | n |", "|---|---|---|"]
        for t, hits in sorted(buckets.items()):
            lines.append(f"| {t} | {_fmt(mean([1.0 if h else 0.0 for h in hits]))} | {len(hits)} |")
    else:
        lines.append("- type 分桶:未啟用或 corpus 無 type 資訊")
    return lines


def _multi_run(scores: list[dict]) -> list[str]:
    if not any(r["run"] > 0 for r in scores):
        return []
    by_qid: dict[str, list[dict]] = defaultdict(list)
    for r in scores:
        if not r.get("system_error"):
            by_qid[r["qid"]].append(r)
    success = [1.0 if any(x.get("correctness") == 2 for x in rows) else 0.0
               for rows in by_qid.values()]
    agreement = []
    for rows in by_qid.values():
        if len(rows) < 2:
            continue
        same_corr = len({x.get("correctness") for x in rows}) == 1
        same_sig = len({x.get("answer_numeric_signature") for x in rows}) == 1
        agreement.append(1.0 if same_corr and same_sig else 0.0)
    return [
        "", "## 多次執行", "",
        f"- mean_correctness(逐 run):"
        + _fmt(_safe_mean([float(r['correctness']) for r in scores
                           if not r.get('system_error') and r.get('correctness') is not None])),
        f"- success@N(任一 run 得 2 分):{_fmt(_safe_mean(success))}",
        f"- agreement_rate:{_fmt(_safe_mean(agreement))}",
    ]


def _tags(scores: list[dict], items: list[DatasetItem]) -> list[str]:
    by_id = {i.id: i for i in items}
    per_tag: dict[str, list[dict]] = defaultdict(list)
    for r in scores:
        item = by_id.get(r["qid"])
        if item is None or r.get("system_error"):
            continue
        for t in item.tags:
            per_tag[t].append(r)
    lines = ["", "## 分項統計 (tags)", "", "| tag | mean_correctness | evidence_recall | n |", "|---|---|---|---|"]
    for t, rows in sorted(per_tag.items()):
        corr = _safe_mean([float(r["correctness"]) for r in rows if r.get("correctness") is not None])
        rec = _safe_mean([(r.get("retrieval") or {}).get("evidence_recall")
                          for r in rows if r.get("retrieval")])
        lines.append(f"| {t} | {_fmt(corr)} | {_fmt(rec)} | {len(rows)} |")
    return lines


def _worst_cases(scores: list[dict], items: list[DatasetItem]) -> list[str]:
    by_id = {i.id: i for i in items}
    bad = [
        r for r in scores
        if not r.get("system_error")
        and (
            r.get("correctness") == 0
            or (r.get("retrieval") and not r["retrieval"].get("hit_at_5"))
        )
    ][:10]
    if not bad:
        return []
    lines = ["", "## 最差案例", ""]
    for r in bad:
        item = by_id.get(r["qid"])
        q = item.question if item else "?"
        lines.append(
            f"- `{r['qid']}` run {r['run']}:{q} | correctness={r.get('correctness')} "
            f"| {r.get('judge_reason') or r.get('numeric_status') or ''}"
        )
    return lines


_BASELINE_METRICS = ["mean_correctness", "evidence_recall", "faithfulness", "hit_at_5"]


def _baseline(scores: list[dict], baseline: list[dict], iters: int, seed: int) -> list[str]:
    cur = {(r["qid"], r["run"]): r for r in scores}
    base = {(r["qid"], r["run"]): r for r in baseline}
    shared = sorted(set(cur) & set(base))
    lines = ["", "## Baseline 比較", "", f"配對樣本數:{len(shared)}", "",
             "| 指標 | 差值 | 95% CI | 顯著 |", "|---|---|---|---|"]
    for name in _BASELINE_METRICS:
        extract, zero = _QUALITY_METRICS[name]

        def val(r: dict) -> float | None:
            if r.get("system_error"):
                return zero
            v = extract(r)
            return None if v is None else float(v)

        diffs = []
        for key in shared:
            a, b = val(cur[key]), val(base[key])
            if a is not None and b is not None:
                diffs.append(a - b)
        if not diffs:
            lines.append(f"| {name} | n/a | n/a | |")
            continue
        lo, hi = paired_bootstrap_ci(diffs, iters=iters, seed=seed)
        sig = "✱" if lo > 0 or hi < 0 else ""
        lines.append(f"| {name} | {_fmt(mean(diffs))} | [{_fmt(lo)}, {_fmt(hi)}] | {sig} |")
    return lines


def build_report(
    *,
    scores: list[dict],
    items: list[DatasetItem],
    manifest: dict | None = None,
    baseline: list[dict] | None = None,
    bootstrap_iters: int = 2000,
    seed: int = 0,
) -> str:
    lines = ["# RAG 評測報告", ""]
    if manifest:
        lines += [
            f"- run 開始:{manifest.get('started_at')}",
            f"- dataset sha256:{manifest.get('dataset_sha256')}",
            f"- judge:{manifest.get('judge_model')} "
            f"(prompt {manifest.get('judge_prompt_version')})",
            "",
        ]
    lines += _overview(scores, items)
    lines += _diagnostics(scores)
    lines += _multi_run(scores)
    lines += _tags(scores, items)
    lines += _worst_cases(scores, items)
    if baseline is not None:
        lines += _baseline(scores, baseline, bootstrap_iters, seed)
    return "\n".join(lines) + "\n"
