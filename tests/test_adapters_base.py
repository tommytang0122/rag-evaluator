from rag_evaluator.adapters.base import (
    RAGAnswer,
    RAGSystem,
    SourceRef,
    SupportsTopKOverride,
)


def test_source_ref_optional_fields_default_none():
    ref = SourceRef(document="a.pdf", page=3)
    assert ref.collection is None
    assert ref.score is None
    assert ref.type is None
    assert ref.image_path is None
    assert ref.content is None
    assert ref.schema_text is None
    assert ref.file_hash is None


class _Fake:
    def ask(self, question: str) -> RAGAnswer:
        return RAGAnswer(answer="", sources=[], latency_ms=0)


class _FakeWithTopK(_Fake):
    def ask_with_top_k(self, question: str, top_k: int) -> RAGAnswer:
        return self.ask(question)


def test_protocols_runtime_checkable():
    assert isinstance(_Fake(), RAGSystem)
    assert not isinstance(_Fake(), SupportsTopKOverride)
    assert isinstance(_FakeWithTopK(), SupportsTopKOverride)
