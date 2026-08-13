import json

import pytest

from rag_evaluator import cli
from rag_evaluator.adapters.base import RAGAnswer, SourceRef

MANIFEST_ROW = {
    "collection": "hr", "source": "/nas/a.pdf", "file_hash": "h", "page": 1,
    "image_path": "x.png", "point_id": "p", "flow": "portrait_table",
    "schema_text": "s", "content": "差旅住宿補助每日上限 2,500 元",
}

DATASET_ROW = {
    "id": "q-11111111", "question": "住宿補助上限?", "answer_type": "answerable",
    "gold_answer": "每日 2,500 元", "gold_value": {"number": 2500, "unit": "元"},
    "tags": ["numeric"], "evidence": [{"collection": "hr", "document": "a.pdf", "page": 1}],
}

SYSTEM_YAML = """
adapter: nas_rag
endpoint: http://localhost:8020/v1/query
collection_names: [hr]
top_k: 5
"""


def test_corpus_from_nas_rag(tmp_path):
    m = tmp_path / "manifest.jsonl"
    m.write_text(json.dumps(MANIFEST_ROW, ensure_ascii=False) + "\n", encoding="utf-8")
    out = tmp_path / "corpus.jsonl"
    assert cli.main(["corpus", "from-nas-rag", "--manifest", str(m), "-o", str(out)]) == 0
    row = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert row["text_source"] == "content" and row["type"] == "table_text"


def test_corpus_from_qdrant(tmp_path, monkeypatch):
    from rag_evaluator.dataset.models import CorpusPage

    calls = {}

    def fake_fetch(base_url, collections, client=None):
        calls["args"] = (base_url, list(collections))
        return [CorpusPage(collection="hr", document="/nas/a.pdf", page=1,
                           text="內文", text_source="content", type="table_text")]

    monkeypatch.setattr(cli, "fetch_qdrant_pages", fake_fetch)
    out = tmp_path / "corpus.jsonl"
    assert cli.main([
        "corpus", "from-qdrant", "--url", "http://localhost:6333",
        "--collection", "hr", "-o", str(out),
    ]) == 0
    assert calls["args"] == ("http://localhost:6333", ["hr"])
    row = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert row["type"] == "table_text" and row["text_source"] == "content"


def test_corpus_from_qdrant_url_defaults_from_env(tmp_path, monkeypatch):
    from rag_evaluator.dataset.models import CorpusPage

    calls = {}

    def fake_fetch(base_url, collections, client=None):
        calls["url"] = base_url
        return [CorpusPage(collection="hr", document="a.pdf", page=1)]

    monkeypatch.setattr(cli, "fetch_qdrant_pages", fake_fetch)
    monkeypatch.setenv("RAG_EVAL_QDRANT_URL", "http://qdrant.lan:6333/")
    out = tmp_path / "corpus.jsonl"
    assert cli.main(["corpus", "from-qdrant", "--collection", "hr", "-o", str(out)]) == 0
    assert calls["url"] == "http://qdrant.lan:6333"


def test_dataset_finalize(tmp_path):
    review = tmp_path / "review.csv"
    review.write_text(
        "question,gold_answer,gold_value,answer_type,tags,evidence,"
        "source_excerpt,generation_basis,approved\n"
        '住宿補助上限?,每日 2500 元,,answerable,numeric,hr|a.pdf|1,,,yes\n',
        encoding="utf-8",
    )
    out = tmp_path / "dataset.jsonl"
    assert cli.main(["dataset", "finalize", "--review", str(review), "-o", str(out)]) == 0
    assert len(out.read_text(encoding="utf-8").splitlines()) == 1


class _StubAdapter:
    def ask(self, question):
        return RAGAnswer(
            answer="每日 2,500 元",
            sources=[SourceRef(document="a.pdf", page=1, collection="hr",
                               content="差旅住宿補助每日上限 2,500 元")],
            latency_ms=3,
        )


class _StubJudge:
    def judge(self, prompt, schema, images=()):
        from rag_evaluator.eval.generation import ClaimExtraction
        return ClaimExtraction(claims=[])


def _write_inputs(tmp_path):
    d = tmp_path / "dataset.jsonl"
    d.write_text(json.dumps(DATASET_ROW, ensure_ascii=False) + "\n", encoding="utf-8")
    y = tmp_path / "sys.yaml"
    y.write_text(SYSTEM_YAML, encoding="utf-8")
    return d, y


def test_collect_score_report_pipeline(tmp_path, monkeypatch):
    d, y = _write_inputs(tmp_path)
    monkeypatch.setattr(cli, "build_adapter", lambda cfg: _StubAdapter())
    monkeypatch.setattr(cli, "_build_judge", lambda args: _StubJudge())
    runs_dir = str(tmp_path / "runs")
    assert cli.main(["collect", "--system", str(y), "--dataset", str(d),
                     "--run-id", "r1", "--runs-dir", runs_dir]) == 0
    assert cli.main(["score", "--run-id", "r1", "--dataset", str(d),
                     "--runs-dir", runs_dir]) == 0
    assert cli.main(["report", "--run-id", "r1", "--dataset", str(d),
                     "--runs-dir", runs_dir]) == 0
    report = (tmp_path / "runs" / "r1" / "report.md").read_text(encoding="utf-8")
    assert "mean_correctness" in report


def test_collect_resume_mismatch_exits_2(tmp_path, monkeypatch):
    d, y = _write_inputs(tmp_path)
    monkeypatch.setattr(cli, "build_adapter", lambda cfg: _StubAdapter())
    runs_dir = str(tmp_path / "runs")
    assert cli.main(["collect", "--system", str(y), "--dataset", str(d),
                     "--run-id", "r1", "--runs-dir", runs_dir]) == 0
    # change dataset content → resume must refuse
    d.write_text(json.dumps({**DATASET_ROW, "question": "改了?"},
                            ensure_ascii=False) + "\n", encoding="utf-8")
    assert cli.main(["collect", "--system", str(y), "--dataset", str(d),
                     "--run-id", "r1", "--runs-dir", runs_dir]) == 2


def test_malformed_yaml_system_config_exits_2(tmp_path):
    d = tmp_path / "dataset.jsonl"
    d.write_text(json.dumps(DATASET_ROW, ensure_ascii=False) + "\n", encoding="utf-8")
    bad_yaml = tmp_path / "sys.yaml"
    bad_yaml.write_text("adapter: nas_rag\n  bad: : : indent", encoding="utf-8")
    rc = cli.main(["collect", "--system", str(bad_yaml), "--dataset", str(d),
                   "--run-id", "rbad", "--runs-dir", str(tmp_path / "runs")])
    assert rc == 2


def test_score_with_changed_dataset_exits_2(tmp_path, monkeypatch):
    d, y = _write_inputs(tmp_path)
    monkeypatch.setattr(cli, "build_adapter", lambda cfg: _StubAdapter())
    monkeypatch.setattr(cli, "_build_judge", lambda args: _StubJudge())
    runs_dir = str(tmp_path / "runs")
    assert cli.main(["collect", "--system", str(y), "--dataset", str(d),
                     "--run-id", "r1", "--runs-dir", runs_dir]) == 0
    # modify the dataset (different question/gold) → SHA changes
    d.write_text(json.dumps({**DATASET_ROW, "id": "q-22222222",
                             "question": "不同問題?"}, ensure_ascii=False) + "\n",
                 encoding="utf-8")
    rc = cli.main(["score", "--run-id", "r1", "--dataset", str(d),
                   "--runs-dir", runs_dir])
    assert rc == 2


def test_report_with_changed_dataset_exits_2(tmp_path, monkeypatch):
    d, y = _write_inputs(tmp_path)
    monkeypatch.setattr(cli, "build_adapter", lambda cfg: _StubAdapter())
    monkeypatch.setattr(cli, "_build_judge", lambda args: _StubJudge())
    runs_dir = str(tmp_path / "runs")
    assert cli.main(["collect", "--system", str(y), "--dataset", str(d),
                     "--run-id", "r1", "--runs-dir", runs_dir]) == 0
    assert cli.main(["score", "--run-id", "r1", "--dataset", str(d),
                     "--runs-dir", runs_dir]) == 0
    # modify the dataset after score → report must refuse
    d.write_text(json.dumps({**DATASET_ROW, "id": "q-22222222",
                             "question": "不同問題?"}, ensure_ascii=False) + "\n",
                 encoding="utf-8")
    rc = cli.main(["report", "--run-id", "r1", "--dataset", str(d),
                   "--runs-dir", runs_dir])
    assert rc == 2


def test_score_corpus_mismatch_exits_2(tmp_path, monkeypatch):
    d, y = _write_inputs(tmp_path)
    monkeypatch.setattr(cli, "build_adapter", lambda cfg: _StubAdapter())
    monkeypatch.setattr(cli, "_build_judge", lambda args: _StubJudge())
    runs_dir = str(tmp_path / "runs")
    assert cli.main(["collect", "--system", str(y), "--dataset", str(d),
                     "--run-id", "r1", "--runs-dir", runs_dir]) == 0

    corpus1 = tmp_path / "corpus1.jsonl"
    corpus1.write_text(
        json.dumps({"collection": "hr", "document": "a.pdf", "page": 1,
                    "text": "差旅住宿補助每日上限 2,500 元",
                    "text_source": "content", "type": "table_figure"},
                   ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    rc = cli.main(["score", "--run-id", "r1", "--dataset", str(d),
                   "--corpus", str(corpus1), "--runs-dir", runs_dir])
    assert rc == 0

    corpus2 = tmp_path / "corpus2.jsonl"
    corpus2.write_text(
        json.dumps({"collection": "hr", "document": "a.pdf", "page": 1,
                    "text": "不同的內容",
                    "text_source": "content", "type": "table_figure"},
                   ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    rc = cli.main(["score", "--run-id", "r1", "--dataset", str(d),
                   "--corpus", str(corpus2), "--runs-dir", runs_dir])
    assert rc == 2


def test_corpus_collision_exits_2(tmp_path, monkeypatch):
    d = tmp_path / "dataset.jsonl"
    d.write_text(json.dumps(DATASET_ROW, ensure_ascii=False) + "\n", encoding="utf-8")
    y = tmp_path / "sys.yaml"
    y.write_text(SYSTEM_YAML, encoding="utf-8")
    monkeypatch.setattr(cli, "build_adapter", lambda cfg: _StubAdapter())
    monkeypatch.setattr(cli, "_build_judge", lambda args: _StubJudge())
    runs_dir = str(tmp_path / "runs")
    cli.main(["collect", "--system", str(y), "--dataset", str(d),
              "--run-id", "rc", "--runs-dir", runs_dir])
    # corpus with a normalized-name collision (same collection+doc+page, different file_hash)
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        json.dumps({"collection": "hr", "document": "a.pdf", "page": 1, "file_hash": "h1"}, ensure_ascii=False) + "\n"
        + json.dumps({"collection": "hr", "document": "A.PDF", "page": 1, "file_hash": "h2"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    rc = cli.main(["score", "--run-id", "rc", "--dataset", str(d),
                   "--corpus", str(corpus), "--runs-dir", runs_dir])
    assert rc == 2


def test_bad_corpus_does_not_poison_manifest(tmp_path, monkeypatch):
    from rag_evaluator.run_manifest import load_manifest

    d = tmp_path / "dataset.jsonl"
    d.write_text(json.dumps(DATASET_ROW, ensure_ascii=False) + "\n", encoding="utf-8")
    y = tmp_path / "sys.yaml"
    y.write_text(SYSTEM_YAML, encoding="utf-8")
    monkeypatch.setattr(cli, "build_adapter", lambda cfg: _StubAdapter())
    monkeypatch.setattr(cli, "_build_judge", lambda args: _StubJudge())
    runs_dir = str(tmp_path / "runs")
    run_dir = tmp_path / "runs" / "rp"
    assert cli.main(["collect", "--system", str(y), "--dataset", str(d),
                     "--run-id", "rp", "--runs-dir", runs_dir]) == 0

    # corpus with a normalized-name collision (same collection+doc+page, different file_hash)
    bad_corpus = tmp_path / "bad_corpus.jsonl"
    bad_corpus.write_text(
        json.dumps({"collection": "hr", "document": "a.pdf", "page": 1, "file_hash": "h1"}, ensure_ascii=False) + "\n"
        + json.dumps({"collection": "hr", "document": "A.PDF", "page": 1, "file_hash": "h2"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    rc = cli.main(["score", "--run-id", "rp", "--dataset", str(d),
                   "--corpus", str(bad_corpus), "--runs-dir", runs_dir])
    assert rc == 2

    # the failed load must NOT have stamped a corpus SHA onto the canonical manifest
    manifest = load_manifest(run_dir)
    assert manifest.get("corpus_sha256") is None

    # a subsequent score with a valid corpus must not be rejected for corpus SHA
    good_corpus = tmp_path / "good_corpus.jsonl"
    good_corpus.write_text(
        json.dumps({"collection": "hr", "document": "a.pdf", "page": 1,
                    "text": "差旅住宿補助每日上限 2,500 元",
                    "text_source": "content", "type": "table_figure"},
                   ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    rc = cli.main(["score", "--run-id", "rp", "--dataset", str(d),
                   "--corpus", str(good_corpus), "--runs-dir", runs_dir])
    assert rc == 0
