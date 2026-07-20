import json
from pathlib import Path

import httpx
import pytest

from rag_evaluator.adapters.base import RAGSystem, SupportsTopKOverride
from rag_evaluator.adapters.nas_rag import NasRagAdapter, build_adapter
from rag_evaluator.config import SystemConfig

# Contract fixture: placeholder shaped from the archived nas-rag qa_api.py payload.
# Before the first real evaluation, re-record it against the live service and re-run
# this test file:
#   curl -s -X POST http://<host>:8020/v1/query -H 'Content-Type: application/json' \
#     -d '{"query":"2025年1月營收?","collection_names":["gavin_test"],"top_k":5}' \
#     > tests/fixtures/nas_rag_response.json
# The contract test asserts only fields the evaluator relies on, so a re-recorded
# fixture with extra fields keeps passing.
FIXTURE = Path(__file__).parent / "fixtures" / "nas_rag_response.json"

CFG = SystemConfig(
    adapter="nas_rag",
    endpoint="http://testserver/v1/query",
    collection_names=["gavin_test"],
    top_k=5,
)


def _adapter(handler):
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return NasRagAdapter(CFG, client=client)


def test_contract_fixture_maps_to_raganswer():
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=payload)

    ans = _adapter(handler).ask("2025年1月營收?")
    assert requests[0] == {
        "query": "2025年1月營收?",
        "collection_names": ["gavin_test"],
        "top_k": 5,
    }
    assert "12,415" in ans.answer
    s0, s1 = ans.sources
    assert s0.page == 3 and s0.type == "table_figure" and s0.score == 0.91
    assert s0.collection == "gavin_test" and s0.file_hash == "sha256-aaa"
    assert s1.content.startswith("台塑廠區") and s1.schema_text
    assert ans.latency_ms >= 0


def test_ask_with_top_k_overrides():
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"answer": "x", "sources": []})

    _adapter(handler).ask_with_top_k("q", 20)
    assert requests[0]["top_k"] == 20


def test_source_rows_missing_keys_are_skipped():
    def handler(request):
        return httpx.Response(
            200,
            json={"answer": "x", "sources": [{"source": "a.pdf"}, {"page": 1}]},
        )

    ans = _adapter(handler).ask("q")
    assert ans.sources == []


def test_http_error_propagates():
    def handler(request):
        return httpx.Response(500)

    with pytest.raises(httpx.HTTPStatusError):
        _adapter(handler).ask("q")


def test_build_adapter():
    a = build_adapter(CFG)
    assert isinstance(a, RAGSystem) and isinstance(a, SupportsTopKOverride)
    with pytest.raises(ValueError, match="unknown adapter"):
        build_adapter(CFG.model_copy(update={"adapter": "nope"}))
