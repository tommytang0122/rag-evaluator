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


def score_manifest_name(rescore_tag: str | None) -> str:
    return MANIFEST_NAME if rescore_tag is None else f"run_manifest-{rescore_tag}.json"


def save_named_manifest(run_dir: Path, name: str, manifest: dict) -> None:
    (Path(run_dir) / name).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_named_manifest(run_dir: Path, name: str) -> dict | None:
    p = Path(run_dir) / name
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def save_manifest(run_dir: Path, manifest: dict) -> None:
    save_named_manifest(run_dir, MANIFEST_NAME, manifest)


def load_manifest(run_dir: Path) -> dict | None:
    return load_named_manifest(run_dir, MANIFEST_NAME)


def ensure_consistent(existing: dict, candidate: dict, keys: list[str]) -> None:
    diff = [k for k in keys if existing.get(k) != candidate.get(k)]
    if diff:
        raise RunManifestMismatch(
            "run_manifest mismatch on: " + ", ".join(diff)
            + " — use a new run-id (or --rescore-tag for judge changes)"
        )


def merge_score_fields(
    run_dir: Path, fields: dict, *, rescore_tag: str | None = None
) -> dict:
    """Stamp judge/score provenance fields onto a run manifest.

    rescore_tag=None (canonical): mutates run_manifest.json directly and
    raises RunManifestMismatch if SCORE_KEYS drift from what's already
    stamped there (strict — this is the manifest a resumed plain `score`
    checks against).

    rescore_tag="<tag>": writes to a tag-scoped sidecar
    run_manifest-<tag>.json instead, seeded from the canonical manifest's
    COLLECT_KEYS so it still describes the same collect run. A brand-new
    sidecar accepts any judge fields (no prior tagged run to conflict
    with). If the sidecar already exists (resuming the same rescore tag),
    its own previously-stamped SCORE_KEYS are checked for drift, same as
    the canonical path. The canonical run_manifest.json is never touched.
    """
    run_dir = Path(run_dir)
    canonical = load_manifest(run_dir)
    if canonical is None:
        raise RunManifestMismatch(f"no {MANIFEST_NAME} in {run_dir}")

    if rescore_tag is None:
        manifest = canonical
        if any(k in manifest for k in SCORE_KEYS):
            ensure_consistent(manifest, {**manifest, **fields}, SCORE_KEYS)
        manifest.update(fields)
        save_manifest(run_dir, manifest)
        return manifest

    name = score_manifest_name(rescore_tag)
    existing = load_named_manifest(run_dir, name)
    base = existing if existing is not None else canonical
    manifest = {**base, **fields}
    if existing is not None and any(k in existing for k in SCORE_KEYS):
        ensure_consistent(existing, manifest, SCORE_KEYS)
    save_named_manifest(run_dir, name, manifest)
    return manifest
