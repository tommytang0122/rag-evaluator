from __future__ import annotations

import time

import httpx

from rag_evaluator.adapters.base import RAGAnswer, RAGSystem, SourceRef
from rag_evaluator.config import SystemConfig


class NasRagAdapter:
    def __init__(self, config: SystemConfig, client: httpx.Client | None = None):
        self._config = config
        self._client = client or httpx.Client(timeout=config.timeout_s)

    def ask(self, question: str) -> RAGAnswer:
        return self.ask_with_top_k(question, self._config.top_k)

    def ask_with_top_k(self, question: str, top_k: int) -> RAGAnswer:
        t0 = time.perf_counter()
        resp = self._client.post(
            self._config.endpoint,
            json={
                "query": question,
                "collection_names": self._config.collection_names,
                "top_k": top_k,
            },
        )
        resp.raise_for_status()
        latency_ms = int((time.perf_counter() - t0) * 1000)
        data = resp.json()
        sources = [
            SourceRef(
                document=d["source"],
                page=int(d["page"]),
                collection=d.get("collection"),
                score=d.get("rerank_score"),
                type=d.get("type"),
                image_path=d.get("image_path"),
                content=d.get("content"),
                schema_text=d.get("schema_text"),
                file_hash=d.get("file_hash"),
            )
            for d in data.get("sources", [])
            if "source" in d and "page" in d
        ]
        return RAGAnswer(answer=data["answer"], sources=sources, latency_ms=latency_ms)


ADAPTERS: dict[str, type] = {"nas_rag": NasRagAdapter}


def build_adapter(config: SystemConfig) -> RAGSystem:
    cls = ADAPTERS.get(config.adapter)
    if cls is None:
        raise ValueError(f"unknown adapter: {config.adapter}")
    return cls(config)
