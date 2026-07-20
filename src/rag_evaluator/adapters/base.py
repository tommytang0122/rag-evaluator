from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class SourceRef:
    document: str
    page: int
    collection: str | None = None
    score: float | None = None
    type: str | None = None
    image_path: str | None = None
    content: str | None = None
    schema_text: str | None = None
    file_hash: str | None = None


@dataclass
class RAGAnswer:
    answer: str
    sources: list[SourceRef] = field(default_factory=list)
    latency_ms: int = 0


@runtime_checkable
class RAGSystem(Protocol):
    def ask(self, question: str) -> RAGAnswer: ...


@runtime_checkable
class SupportsTopKOverride(Protocol):
    def ask_with_top_k(self, question: str, top_k: int) -> RAGAnswer: ...
