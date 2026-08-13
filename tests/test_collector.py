import json

from rag_evaluator.adapters.base import RAGAnswer, SourceRef
from rag_evaluator.dataset.models import DatasetItem
from rag_evaluator.eval.collector import CollectStats, collect

ITEM = DatasetItem.model_validate(
    {
        "id": "q-1",
        "question": "營收?",
        "answer_type": "answerable",
        "gold_answer": "12,415 千元",
        "evidence": [{"document": "a.pdf", "page": 3}],
    }
)
REFUSAL_ITEM = DatasetItem.model_validate(
    {"id": "q-2", "question": "??", "answer_type": "refusal", "tags": ["unanswerable"]}
)


class FakeSystem:
    """ask() succeeds; no top-k capability."""

    def __init__(self):
        self.calls = []

    def ask(self, question):
        self.calls.append(("ask", question))
        return RAGAnswer(
            answer="12,415 千元",
            sources=[SourceRef(document="a.pdf", page=3)],
            latency_ms=7,
        )


class FakeSystemWithProbe(FakeSystem):
    def ask_with_top_k(self, question, top_k):
        self.calls.append(("probe", top_k))
        return RAGAnswer(answer="", sources=[SourceRef(document="a.pdf", page=3)])


class FlakySystem(FakeSystem):
    def __init__(self, fail_times):
        super().__init__()
        self.fail_times = fail_times

    def ask(self, question):
        if self.fail_times > 0:
            self.fail_times -= 1
            raise ConnectionError("down")
        return super().ask(question)


def _lines(run_dir):
    return [
        json.loads(l)
        for l in (run_dir / "raw.jsonl").read_text(encoding="utf-8").splitlines()
    ]


def test_collect_writes_answer_rows(tmp_path):
    stats = collect(
        adapter=FakeSystem(), items=[ITEM, REFUSAL_ITEM], run_dir=tmp_path, runs=2
    )
    assert stats == CollectStats(completed=4, skipped=0, errors=0, probes=0)
    rows = _lines(tmp_path)
    assert {(r["qid"], r["run"]) for r in rows} == {
        ("q-1", 0), ("q-1", 1), ("q-2", 0), ("q-2", 1)
    }
    assert rows[0]["sources"][0]["document"] == "a.pdf"


def test_collect_resume_skips_existing(tmp_path):
    collect(adapter=FakeSystem(), items=[ITEM], run_dir=tmp_path, runs=1)
    stats = collect(adapter=FakeSystem(), items=[ITEM], run_dir=tmp_path, runs=1)
    assert stats.skipped == 1 and stats.completed == 0
    assert len(_lines(tmp_path)) == 1


def test_collect_retries_then_succeeds(tmp_path):
    sys_ = FlakySystem(fail_times=2)  # 2 failures, retries=2 → succeeds on 3rd
    stats = collect(adapter=sys_, items=[ITEM], run_dir=tmp_path, runs=1, retries=2)
    assert stats.errors == 0
    assert _lines(tmp_path)[0]["error"] is None


def test_collect_system_error_recorded(tmp_path):
    sys_ = FlakySystem(fail_times=10)
    stats = collect(adapter=sys_, items=[ITEM], run_dir=tmp_path, runs=1, retries=2)
    assert stats.errors == 1
    row = _lines(tmp_path)[0]
    assert row["error"].startswith("system_error: ") and row["answer"] is None


def test_collect_probe_once_for_answerable_only(tmp_path):
    sys_ = FakeSystemWithProbe()
    stats = collect(
        adapter=sys_, items=[ITEM, REFUSAL_ITEM], run_dir=tmp_path, runs=2,
        probe_top_k=20,
    )
    assert stats.probes == 1
    probes = [r for r in _lines(tmp_path) if r["kind"] == "probe"]
    assert len(probes) == 1 and probes[0]["qid"] == "q-1"
    assert ("probe", 20) in sys_.calls


def test_collect_probe_skipped_without_capability(tmp_path):
    stats = collect(
        adapter=FakeSystem(), items=[ITEM], run_dir=tmp_path, runs=1, probe_top_k=20
    )
    assert stats.probes == 0
    assert all(r["kind"] == "answer" for r in _lines(tmp_path))


def test_collect_probe_failure_counts_as_error(tmp_path):
    class FlakyProbe(FakeSystemWithProbe):
        def ask_with_top_k(self, question, top_k):
            raise ConnectionError("probe down")

    stats = collect(
        adapter=FlakyProbe(), items=[ITEM], run_dir=tmp_path, runs=1,
        probe_top_k=20, retries=1,
    )
    assert stats.errors == 1 and stats.probes == 0
    probes = [r for r in _lines(tmp_path) if r["kind"] == "probe"]
    assert probes[0]["error"].startswith("system_error: ")


def test_collect_probe_resume_counts_as_skipped(tmp_path):
    collect(adapter=FakeSystemWithProbe(), items=[ITEM], run_dir=tmp_path,
            runs=1, probe_top_k=20)
    stats = collect(adapter=FakeSystemWithProbe(), items=[ITEM], run_dir=tmp_path,
                    runs=1, probe_top_k=20)
    assert stats.skipped == 2  # 1 answer row + 1 probe row
    assert len([r for r in _lines(tmp_path) if r["kind"] == "probe"]) == 1
