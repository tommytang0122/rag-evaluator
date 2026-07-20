from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

from rag_evaluator.adapters.base import SourceRef
from rag_evaluator.config import DiagnosticsConfig
from rag_evaluator.dataset.corpus import Corpus
from rag_evaluator.dataset.models import DatasetItem
from rag_evaluator.eval.generation import (
    PROMPT_VERSION,
    JudgeProtocol,
    prompt_hash,
    score_correctness,
    score_faithfulness,
)
from rag_evaluator.eval.numeric import NUMERIC_RULES_VERSION, extract_values
from rag_evaluator.eval.refusal import DEFAULT_REFUSAL_PHRASES, refusal_phrases_hash
from rag_evaluator.eval.retrieval import (
    all_evidence_hit,
    attribute_misses,
    citation_precision,
    evidence_hits,
    evidence_recall,
    hit_at_k,
    mrr,
)
from rag_evaluator.run_manifest import merge_score_fields


def _to_sources(rows: list[dict] | None) -> list[SourceRef]:
    return [SourceRef(**r) for r in (rows or [])]


def _existing(path: Path) -> set[tuple[str, int]]:
    done: set[tuple[str, int]] = set()
    if path.exists():
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    done.add((row["qid"], row["run"]))
    return done


def score_run(
    *,
    run_dir: Path,
    items: list[DatasetItem],
    corpus: Corpus | None,
    judge: JudgeProtocol,
    judge_model: str,
    diagnostics: DiagnosticsConfig,
    top_k: int,
    rescore_tag: str | None = None,
    refusal_phrases=DEFAULT_REFUSAL_PHRASES,
    tolerance: Decimal = Decimal(0),
) -> Path:
    run_dir = Path(run_dir)
    merge_score_fields(
        run_dir,
        {
            "judge_model": judge_model,
            "judge_prompt_version": PROMPT_VERSION,
            "judge_prompt_hash": prompt_hash(),
            "refusal_phrases_hash": refusal_phrases_hash(refusal_phrases),
            "numeric_rules_version": NUMERIC_RULES_VERSION,
        },
        rescore_tag=rescore_tag,
    )
    scores_path = run_dir / (
        f"scores-{rescore_tag}.jsonl" if rescore_tag else "scores.jsonl"
    )
    items_by_id = {i.id: i for i in items}

    answer_rows: list[dict] = []
    probes: dict[str, list[SourceRef]] = {}
    with (run_dir / "raw.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row["kind"] == "probe" and not row.get("error"):
                probes[row["qid"]] = _to_sources(row["sources"])
            elif row["kind"] == "answer":
                answer_rows.append(row)

    done = _existing(scores_path)
    with scores_path.open("a", encoding="utf-8") as out:
        for row in answer_rows:
            qid, run = row["qid"], row["run"]
            if (qid, run) in done or qid not in items_by_id:
                continue
            item = items_by_id[qid]
            score = _score_row(
                item, row, probes.get(qid), corpus, judge, diagnostics, top_k,
                refusal_phrases, tolerance,
            )
            out.write(json.dumps(score, ensure_ascii=False) + "\n")
            out.flush()
    return scores_path


def _score_row(
    item: DatasetItem,
    row: dict,
    probe_sources: list[SourceRef] | None,
    corpus: Corpus | None,
    judge: JudgeProtocol,
    diagnostics: DiagnosticsConfig,
    top_k: int,
    refusal_phrases,
    tolerance: Decimal,
) -> dict:
    base = {
        "qid": item.id,
        "run": row["run"],
        "system_error": bool(row.get("error")),
        "latency_ms": row.get("latency_ms"),
        "retrieval": None,
        "gold_type_hits": None,
        "attribution": None,
        "correctness": None,
        "method": None,
        "refusal_state": None,
        "numeric_status": None,
        "numeric_canonical": None,
        "answer_numeric_signature": None,
        "unit_mismatch": False,
        "false_refusal": False,
        "hallucinated_answer": False,
        "judge_reason": None,
        "judge_error": False,
        "faithfulness": None,
        "faithfulness_status": None,
        "faithfulness_total_claims": 0,
        "faithfulness_supported": 0,
        "faithfulness_evaluable": 0,
        "faithfulness_evidence_unavailable": 0,
        "faithfulness_evaluable_claim_rate": None,
        "judge_prompt_version": PROMPT_VERSION,
        "judge_prompt_hash": prompt_hash(),
    }
    if base["system_error"]:
        for key in (
            "unit_mismatch", "false_refusal", "hallucinated_answer", "judge_error",
            "faithfulness_total_claims", "faithfulness_supported",
            "faithfulness_evaluable", "faithfulness_evidence_unavailable",
            "answer_numeric_signature",
        ):
            base[key] = None
        return base

    answer: str = row["answer"]
    sources = _to_sources(row["sources"])

    if item.answer_type == "answerable":
        base["retrieval"] = {
            "hit_at_1": hit_at_k(item.evidence, sources, 1),
            "hit_at_3": hit_at_k(item.evidence, sources, 3),
            "hit_at_5": hit_at_k(item.evidence, sources, 5),
            "evidence_recall": evidence_recall(item.evidence, sources),
            "all_evidence_hit": all_evidence_hit(item.evidence, sources),
            "mrr": mrr(item.evidence, sources),
            "citation_precision": citation_precision(item.evidence, sources),
        }
        if diagnostics.type_buckets and corpus is not None:
            hits = evidence_hits(item.evidence, sources)
            gold_type_hits = []
            for e, hit in zip(item.evidence, hits):
                page = corpus.lookup(e.document, e.page, e.collection)
                if page is not None and page.type:
                    gold_type_hits.append({"type": page.type, "hit": hit})
            base["gold_type_hits"] = gold_type_hits or None
        if (
            diagnostics.cutoff_probe_top_k
            and probe_sources is not None
            and not base["retrieval"]["all_evidence_hit"]
        ):
            base["attribution"] = attribute_misses(
                item.evidence, sources, probe_sources, top_k
            )

    base["answer_numeric_signature"] = ",".join(
        sorted(str(v.canonical) for v in extract_values(answer))
    )

    corr = score_correctness(
        item, answer, judge, refusal_phrases=refusal_phrases, tolerance=tolerance
    )
    base.update(
        {
            "correctness": corr.correctness,
            "method": corr.method,
            "refusal_state": corr.refusal_state,
            "numeric_status": corr.numeric_status,
            "numeric_canonical": corr.numeric_canonical,
            "unit_mismatch": corr.unit_mismatch,
            "false_refusal": corr.false_refusal,
            "hallucinated_answer": corr.hallucinated_answer,
            "judge_reason": corr.judge_reason,
            "judge_error": corr.judge_error,
        }
    )

    faith = score_faithfulness(
        item, answer, sources, corpus, judge, refusal_phrases=refusal_phrases
    )
    base.update(
        {
            "faithfulness": faith.faithfulness,
            "faithfulness_status": faith.status,
            "faithfulness_total_claims": faith.total_claims,
            "faithfulness_supported": faith.supported,
            "faithfulness_evaluable": faith.evaluable,
            "faithfulness_evidence_unavailable": faith.evidence_unavailable,
            "faithfulness_evaluable_claim_rate": faith.evaluable_claim_rate,
        }
    )
    return base
