import pytest

from rag_evaluator.run_manifest import (
    COLLECT_KEYS,
    RunManifestMismatch,
    build_collect_manifest,
    ensure_consistent,
    file_sha256,
    load_manifest,
    merge_score_fields,
    save_manifest,
)


def _manifest(tmp_path, runs=1):
    d = tmp_path / "dataset.jsonl"
    d.write_text('{"x": 1}\n', encoding="utf-8")
    return build_collect_manifest(
        dataset_path=d,
        corpus_path=None,
        system_config={"adapter": "nas_rag", "top_k": 5},
        runs=runs,
        adapter="nas_rag",
        evaluator_version="0.1.0",
        started_at="2026-07-20T00:00:00+00:00",
    )


def test_file_sha256_stable(tmp_path):
    p = tmp_path / "f"
    p.write_text("abc", encoding="utf-8")
    assert file_sha256(p) == file_sha256(p)
    assert len(file_sha256(p)) == 64


def test_build_and_roundtrip(tmp_path):
    m = _manifest(tmp_path)
    assert m["corpus_sha256"] is None and m["runs"] == 1
    run_dir = tmp_path / "run1"
    run_dir.mkdir()
    save_manifest(run_dir, m)
    assert load_manifest(run_dir) == m
    assert load_manifest(tmp_path / "nope") is None


def test_ensure_consistent_raises_with_diff_keys(tmp_path):
    m1 = _manifest(tmp_path, runs=1)
    m2 = _manifest(tmp_path, runs=3)
    ensure_consistent(m1, dict(m1), COLLECT_KEYS)  # identical → ok
    with pytest.raises(RunManifestMismatch, match="runs"):
        ensure_consistent(m1, m2, COLLECT_KEYS)


def test_merge_score_fields(tmp_path):
    run_dir = tmp_path / "run1"
    run_dir.mkdir()
    save_manifest(run_dir, _manifest(tmp_path))
    fields = {
        "judge_model": "gemini-2.5-flash",
        "judge_prompt_version": "v1",
        "judge_prompt_hash": "h",
        "refusal_phrases_hash": "r",
        "numeric_rules_version": "v1",
    }
    merge_score_fields(run_dir, fields)
    assert load_manifest(run_dir)["judge_model"] == "gemini-2.5-flash"
    # same fields again → ok
    merge_score_fields(run_dir, fields)
    # changed prompt without allow_mismatch → refuse
    with pytest.raises(RunManifestMismatch, match="judge_prompt_hash"):
        merge_score_fields(run_dir, {**fields, "judge_prompt_hash": "h2"})
    # rescore path allows it
    merge_score_fields(run_dir, {**fields, "judge_prompt_hash": "h2"}, allow_mismatch=True)
