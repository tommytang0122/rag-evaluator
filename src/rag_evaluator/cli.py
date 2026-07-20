from __future__ import annotations

import argparse
import importlib.metadata
import os
import sys
import yaml
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from rag_evaluator.adapters.nas_rag import build_adapter
from rag_evaluator.config import load_system_config
from rag_evaluator.dataset.corpus import CorpusError, convert_nas_rag_manifest, load_corpus, write_corpus
from rag_evaluator.dataset.generator import (
    finalize_dataset,
    generate_review_rows,
    write_review_csv,
)
from rag_evaluator.dataset.models import load_dataset
from rag_evaluator.eval.collector import collect
from rag_evaluator.eval.scorer import score_run
from rag_evaluator.judge import GeminiJudge, JudgeError
from rag_evaluator.report import build_report
from rag_evaluator.run_manifest import (
    COLLECT_KEYS,
    RunManifestMismatch,
    build_collect_manifest,
    ensure_consistent,
    load_manifest,
    load_named_manifest,
    save_manifest,
    score_manifest_name,
)

DEFAULT_JUDGE_MODEL = "gemini-2.5-flash"


def _build_judge(args) -> GeminiJudge:
    load_dotenv()
    model = getattr(args, "model", None) or os.environ.get(
        "RAG_EVAL_JUDGE_MODEL", DEFAULT_JUDGE_MODEL
    )
    qps = float(os.environ.get("RAG_EVAL_JUDGE_QPS", "1.0"))
    return GeminiJudge(model, qps=qps)


def _run_dir(args) -> Path:
    return Path(args.runs_dir) / args.run_id


def _cmd_corpus_from_nas_rag(args) -> int:
    pages = convert_nas_rag_manifest(Path(args.manifest))
    write_corpus(pages, Path(args.output))
    print(f"wrote {len(pages)} pages to {args.output}")
    return 0


def _cmd_dataset_generate(args) -> int:
    corpus = load_corpus(Path(args.corpus))
    judge = _build_judge(args)
    rows = generate_review_rows(
        corpus, judge, sample_pages_count=args.sample_pages
    )
    write_review_csv(rows, Path(args.output))
    print(f"wrote {len(rows)} candidate questions to {args.output} (approved 欄請人工填寫;unanswerable 題預設不核可)")
    return 0


def _cmd_dataset_finalize(args) -> int:
    n = finalize_dataset(Path(args.review), Path(args.output))
    print(f"finalized {n} items to {args.output}")
    return 0


def _cmd_collect(args) -> int:
    config = load_system_config(Path(args.system))
    items = load_dataset(Path(args.dataset))
    run_dir = _run_dir(args)
    manifest = build_collect_manifest(
        dataset_path=Path(args.dataset),
        corpus_path=None,
        system_config=config.model_dump(),
        runs=args.runs,
        adapter=config.adapter,
        evaluator_version=importlib.metadata.version("rag-evaluator"),
        started_at=datetime.now(timezone.utc).isoformat(),
    )
    existing = load_manifest(run_dir)
    if existing is not None:
        ensure_consistent(existing, manifest, COLLECT_KEYS)
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
        save_manifest(run_dir, manifest)
    adapter = build_adapter(config)
    stats = collect(
        adapter=adapter,
        items=items,
        run_dir=run_dir,
        runs=args.runs,
        probe_top_k=config.diagnostics.cutoff_probe_top_k,
    )
    print(f"collect done: {stats}")
    return 0


def _cmd_score(args) -> int:
    from rag_evaluator.config import SystemConfig

    run_dir = _run_dir(args)
    manifest = load_manifest(run_dir)
    if manifest is None:
        raise RunManifestMismatch(f"no run_manifest.json in {run_dir} — run collect first")
    config = SystemConfig.model_validate(manifest["system_config"])
    items = load_dataset(Path(args.dataset))
    corpus = load_corpus(Path(args.corpus)) if args.corpus else None
    judge = _build_judge(args)
    model = getattr(args, "model", None) or os.environ.get(
        "RAG_EVAL_JUDGE_MODEL", DEFAULT_JUDGE_MODEL
    )
    out = score_run(
        run_dir=run_dir,
        items=items,
        corpus=corpus,
        judge=judge,
        judge_model=model,
        diagnostics=config.diagnostics,
        top_k=config.top_k,
        rescore_tag=args.rescore_tag,
    )
    print(f"scores written to {out}")
    return 0


def _read_scores(run_dir: Path, rescore_tag: str | None) -> list[dict]:
    import json

    name = f"scores-{rescore_tag}.jsonl" if rescore_tag else "scores.jsonl"
    path = run_dir / name
    return [
        json.loads(l)
        for l in path.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]


def _cmd_report(args) -> int:
    run_dir = _run_dir(args)
    manifest = None
    if args.rescore_tag:
        manifest = load_named_manifest(run_dir, score_manifest_name(args.rescore_tag))
    if manifest is None:
        manifest = load_manifest(run_dir)
    items = load_dataset(Path(args.dataset))
    scores = _read_scores(run_dir, args.rescore_tag)
    baseline_scores = None
    if args.baseline:
        base_dir = Path(args.runs_dir) / args.baseline
        base_manifest = load_manifest(base_dir)
        if manifest and base_manifest:
            ensure_consistent(base_manifest, manifest, ["dataset_sha256"])
        baseline_scores = _read_scores(base_dir, None)
    md = build_report(
        scores=scores, items=items, manifest=manifest, baseline=baseline_scores
    )
    out = run_dir / "report.md"
    out.write_text(md, encoding="utf-8")
    print(str(out))
    return 0


def _cmd_run(args) -> int:
    rc = _cmd_collect(args)
    if rc != 0:
        return rc
    args.rescore_tag = None
    rc = _cmd_score(args)
    if rc != 0:
        return rc
    return _cmd_report(args)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rag-eval")
    sub = parser.add_subparsers(dest="command", required=True)

    corpus = sub.add_parser("corpus").add_subparsers(dest="subcommand", required=True)
    p = corpus.add_parser("from-nas-rag")
    p.add_argument("--manifest", required=True)
    p.add_argument("-o", "--output", required=True)
    p.set_defaults(func=_cmd_corpus_from_nas_rag)

    dataset = sub.add_parser("dataset").add_subparsers(dest="subcommand", required=True)
    p = dataset.add_parser("generate")
    p.add_argument("--corpus", required=True)
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--sample-pages", type=int, default=None)
    p.add_argument("--model", default=None)
    p.set_defaults(func=_cmd_dataset_generate)
    p = dataset.add_parser("finalize")
    p.add_argument("--review", required=True)
    p.add_argument("-o", "--output", required=True)
    p.set_defaults(func=_cmd_dataset_finalize)

    def common_run_args(p):
        p.add_argument("--run-id", required=True)
        p.add_argument("--runs-dir", default="runs")

    p = sub.add_parser("collect")
    p.add_argument("--system", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--runs", type=int, default=1)
    common_run_args(p)
    p.set_defaults(func=_cmd_collect)

    p = sub.add_parser("score")
    p.add_argument("--dataset", required=True)
    p.add_argument("--corpus", default=None)
    p.add_argument("--rescore-tag", default=None)
    p.add_argument("--model", default=None)
    common_run_args(p)
    p.set_defaults(func=_cmd_score)

    p = sub.add_parser("report")
    p.add_argument("--dataset", required=True)
    p.add_argument("--baseline", default=None)
    p.add_argument("--rescore-tag", default=None)
    common_run_args(p)
    p.set_defaults(func=_cmd_report)

    p = sub.add_parser("run")
    p.add_argument("--system", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--corpus", default=None)
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--baseline", default=None)
    p.add_argument("--model", default=None)
    common_run_args(p)
    p.set_defaults(func=_cmd_run)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (RunManifestMismatch, JudgeError, FileNotFoundError, ValueError, CorpusError, yaml.YAMLError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
