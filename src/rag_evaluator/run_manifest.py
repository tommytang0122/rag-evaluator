from __future__ import annotations

import hashlib
import json
from pathlib import Path

MANIFEST_NAME = "run_manifest.json"

COLLECT_KEYS = [
    "dataset_sha256",
    "corpus_sha256",
    "system_config",
    "runs",
    "adapter",
    "evaluator_version",
]
SCORE_KEYS = [
    "judge_model",
    "judge_prompt_version",
    "judge_prompt_hash",
    "refusal_phrases_hash",
    "numeric_rules_version",
]


class RunManifestMismatch(Exception):
    pass


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def build_collect_manifest(
    *,
    dataset_path: Path,
    corpus_path: Path | None,
    system_config: dict,
    runs: int,
    adapter: str,
    evaluator_version: str,
    started_at: str,
) -> dict:
    return {
        "dataset_sha256": file_sha256(dataset_path),
        "corpus_sha256": file_sha256(corpus_path) if corpus_path else None,
        "system_config": system_config,
        "runs": runs,
        "adapter": adapter,
        "evaluator_version": evaluator_version,
        "started_at": started_at,
    }


def save_manifest(run_dir: Path, manifest: dict) -> None:
    (Path(run_dir) / MANIFEST_NAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_manifest(run_dir: Path) -> dict | None:
    p = Path(run_dir) / MANIFEST_NAME
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def ensure_consistent(existing: dict, candidate: dict, keys: list[str]) -> None:
    diff = [k for k in keys if existing.get(k) != candidate.get(k)]
    if diff:
        raise RunManifestMismatch(
            "run_manifest mismatch on: " + ", ".join(diff)
            + " — use a new run-id (or --rescore-tag for judge changes)"
        )


def merge_score_fields(
    run_dir: Path, fields: dict, *, allow_mismatch: bool = False
) -> dict:
    manifest = load_manifest(run_dir)
    if manifest is None:
        raise RunManifestMismatch(f"no {MANIFEST_NAME} in {run_dir}")
    if any(k in manifest for k in SCORE_KEYS) and not allow_mismatch:
        ensure_consistent(manifest, {**manifest, **fields}, SCORE_KEYS)
    manifest.update(fields)
    save_manifest(run_dir, manifest)
    return manifest
