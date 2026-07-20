import json

from rag_evaluator.config import DiagnosticsConfig
from rag_evaluator.dataset.corpus import Corpus
from rag_evaluator.dataset.models import CorpusPage, DatasetItem
from rag_evaluator.eval.scorer import score_run
from rag_evaluator.run_manifest import build_collect_manifest, load_manifest, save_manifest

ITEM = DatasetItem.model_validate(
    {
        "id": "q-1",
        "question": "營收?",
        "answer_type": "answerable",
        "gold_answer": "12,415 千元",
        "gold_value": {"number": 12415, "unit": "千元"},
        "tags": ["numeric"],
        "evidence": [
            {"collection": "hr", "document": "營收月報.pdf", "page": 3},
            {"collection": "hr", "document": "營收月報.pdf", "page": 4},
        ],
    }
)

CORPUS = Corpus(
    [
        CorpusPage(collection="hr", document="營收月報.pdf", page=3,
                   text="營收 12,415 千元", text_source="content", type="table_figure"),
        CorpusPage(collection="hr", document="營收月報.pdf", page=4,
                   text="", text_source="none", type="table_figure"),
    ]
)

SRC3 = {"document": "營收月報.pdf", "page": 3, "collection": "hr", "score": 0.9,
        "type": "table_figure", "image_path": None, "content": "營收 12,415 千元",
        "schema_text": None, "file_hash": None}
SRC4 = {**SRC3, "page": 4, "score": 0.5}


from rag_evaluator.eval.generation import ClaimExtraction


class EmptyClaimsJudge:
    """Correctness is rule-scored in these tests; faithfulness's claim
    extraction returns no claims → not_applicable, no further judge calls."""

    def judge(self, prompt, schema, images=()):
        return ClaimExtraction(claims=[])


def _run_dir(tmp_path, raw_rows, dataset_file="d.jsonl"):
    d = tmp_path / dataset_file
    d.write_text("x\n", encoding="utf-8")
    run_dir = tmp_path / "run1"
    run_dir.mkdir()
    save_manifest(
        run_dir,
        build_collect_manifest(
            dataset_path=d, corpus_path=None, system_config={}, runs=1,
            adapter="nas_rag", evaluator_version="0.1.0",
            started_at="2026-07-20T00:00:00+00:00",
        ),
    )
    (run_dir / "raw.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in raw_rows) + "\n",
        encoding="utf-8",
    )
    return run_dir


def _score(run_dir, **over):
    kw = dict(
        run_dir=run_dir, items=[ITEM], corpus=CORPUS, judge=EmptyClaimsJudge(),
        judge_model="gemini-test",
        diagnostics=DiagnosticsConfig(cutoff_probe_top_k=20, type_buckets=True),
        top_k=5,
    )
    kw.update(over)
    return score_run(**kw)


def test_score_rule_numeric_row_with_diagnostics(tmp_path):
    raw = [
        {"qid": "q-1", "run": 0, "kind": "answer", "answer": "營收為 12,415 千元",
         "sources": [SRC3], "latency_ms": 7, "error": None},
        # probe: page 4 at probe rank 2 (≤ top_k) → lost_to_cutoff
        {"qid": "q-1", "run": 0, "kind": "probe", "answer": "",
         "sources": [SRC3, SRC4], "latency_ms": 5, "error": None},
    ]
    run_dir = _run_dir(tmp_path, raw)
    out = _score(run_dir)
    row = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert row["correctness"] == 2 and row["method"] == "rule_numeric"
    assert row["retrieval"]["evidence_recall"] == 0.5
    assert row["retrieval"]["hit_at_1"] is True
    assert row["attribution"] == {"營收月報#p4": "lost_to_cutoff"}
    assert row["gold_type_hits"] == [
        {"type": "table_figure", "hit": True},
        {"type": "table_figure", "hit": False},
    ]
    assert row["judge_prompt_version"] == "v1"
    assert row["faithfulness_status"] == "not_applicable"  # empty claim extraction
    # manifest stamped with judge fields
    assert load_manifest(run_dir)["judge_model"] == "gemini-test"


def test_score_system_error_row(tmp_path):
    raw = [{"qid": "q-1", "run": 0, "kind": "answer", "answer": None,
            "sources": None, "latency_ms": None, "error": "system_error"}]
    run_dir = _run_dir(tmp_path, raw)
    out = _score(run_dir)
    row = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert row["system_error"] is True and row["correctness"] is None
    assert row["retrieval"] is None


def test_score_resume_skips_done(tmp_path):
    raw = [{"qid": "q-1", "run": 0, "kind": "answer", "answer": "營收為 12,415 千元",
            "sources": [SRC3], "latency_ms": 7, "error": None}]
    run_dir = _run_dir(tmp_path, raw)
    out = _score(run_dir)
    before = out.read_text(encoding="utf-8")
    _score(run_dir)  # second pass must not duplicate
    assert out.read_text(encoding="utf-8") == before


def test_rescore_tag_writes_separate_file(tmp_path):
    raw = [{"qid": "q-1", "run": 0, "kind": "answer", "answer": "營收為 12,415 千元",
            "sources": [SRC3], "latency_ms": 7, "error": None}]
    run_dir = _run_dir(tmp_path, raw)
    _score(run_dir)
    out2 = _score(run_dir, rescore_tag="prompt-v2")
    assert out2.name == "scores-prompt-v2.jsonl"
    assert out2.exists() and (run_dir / "scores.jsonl").exists()
