from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from rag_evaluator.adapters.base import RAGAnswer, RAGSystem, SupportsTopKOverride
from rag_evaluator.dataset.models import DatasetItem

RAW_NAME = "raw.jsonl"


@dataclass
class CollectStats:
    completed: int = 0
    skipped: int = 0
    errors: int = 0
    probes: int = 0


def _existing_keys(raw_path: Path) -> set[tuple[str, int, str]]:
    keys: set[tuple[str, int, str]] = set()
    if raw_path.exists():
        with raw_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    keys.add((row["qid"], row["run"], row["kind"]))
    return keys


BODY_LIMIT = 200


def describe_error(exc: BaseException) -> str:
    """把例外壓成一行診斷字串。

    第一次接真實環境時 401/404/自簽憑證是常態,只記 "system_error" 會逼人
    回頭手動 curl。下游僅以 bool(error) 判斷有無錯誤(scorer.py),字串內容
    可自由攜帶細節。
    """
    response = getattr(exc, "response", None)
    if response is not None:
        body = " ".join(str(getattr(response, "text", "")).split())
        return f"HTTP {response.status_code} {body[:BODY_LIMIT]}".rstrip()
    detail = " ".join(str(exc).split())
    return f"{type(exc).__name__}: {detail[:BODY_LIMIT]}".rstrip(": ")


def _try_ask(fn, retries: int) -> tuple[RAGAnswer | None, str | None]:
    last: BaseException | None = None
    for _attempt in range(retries + 1):
        try:
            return fn(), None
        except Exception as exc:  # noqa: BLE001 - any transport failure counts
            last = exc
    return None, f"system_error: {describe_error(last)}"


def _row(qid: str, run: int, kind: str, answer: RAGAnswer | None, error: str | None) -> dict:
    return {
        "qid": qid,
        "run": run,
        "kind": kind,
        "answer": answer.answer if answer else None,
        "sources": [asdict(s) for s in answer.sources] if answer else None,
        "latency_ms": answer.latency_ms if answer else None,
        "error": error,
    }


def collect(
    *,
    adapter: RAGSystem,
    items: list[DatasetItem],
    run_dir: Path,
    runs: int = 1,
    probe_top_k: int | None = None,
    retries: int = 2,
) -> CollectStats:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    raw_path = run_dir / RAW_NAME
    done = _existing_keys(raw_path)
    stats = CollectStats()
    can_probe = probe_top_k is not None and isinstance(adapter, SupportsTopKOverride)

    with raw_path.open("a", encoding="utf-8") as fh:

        def write(row: dict) -> None:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()

        for item in items:
            for run in range(runs):
                if (item.id, run, "answer") in done:
                    stats.skipped += 1
                    continue
                answer, error = _try_ask(lambda: adapter.ask(item.question), retries)
                write(_row(item.id, run, "answer", answer, error))
                if error:
                    stats.errors += 1
                else:
                    stats.completed += 1
            if can_probe and item.answer_type == "answerable":
                if (item.id, 0, "probe") in done:
                    stats.skipped += 1
                else:
                    answer, error = _try_ask(
                        lambda: adapter.ask_with_top_k(item.question, probe_top_k), retries
                    )
                    write(_row(item.id, 0, "probe", answer, error))
                    if error:
                        stats.errors += 1
                    else:
                        stats.probes += 1
    return stats
