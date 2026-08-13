"""system_error 要帶上診斷細節——第一次接真實環境時 401/404 必須看得見。

下游只用 bool(row["error"]) 判斷(scorer.py),所以字串內容可以自由攜帶細節。
"""

import httpx
import pytest

from rag_evaluator.dataset.models import DatasetItem
from rag_evaluator.eval.collector import collect, describe_error

ITEM = DatasetItem.model_validate(
    {"id": "q-1", "question": "營收?", "answer_type": "refusal"}
)


class BoomSystem:
    def __init__(self, exc):
        self.exc = exc

    def ask(self, question):
        raise self.exc


def _error(tmp_path, exc):
    collect(adapter=BoomSystem(exc), items=[ITEM], run_dir=tmp_path, retries=0)
    import json

    row = json.loads((tmp_path / "raw.jsonl").read_text(encoding="utf-8").strip())
    return row["error"]


def _status_error(code: int, body: str) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "http://testserver/v1/query")
    response = httpx.Response(code, text=body, request=request)
    return httpx.HTTPStatusError("boom", request=request, response=response)


def test_http_status_error_keeps_code_and_body(tmp_path):
    err = _error(tmp_path, _status_error(401, '{"detail":"missing token"}'))
    assert err.startswith("system_error: ")
    assert "401" in err and "missing token" in err


def test_body_is_truncated(tmp_path):
    err = _error(tmp_path, _status_error(500, "x" * 5000))
    assert len(err) < 400


def test_transport_error_keeps_exception_type(tmp_path):
    err = _error(tmp_path, httpx.ConnectError("[Errno 111] Connection refused"))
    assert "ConnectError" in err and "Connection refused" in err


def test_still_truthy_so_scorer_flags_system_error(tmp_path):
    assert bool(_error(tmp_path, ConnectionError("down")))


def test_last_attempt_error_is_reported(tmp_path):
    """retries 用完後回報的是最後一次的例外,不是第一次的。"""

    class Escalating:
        def __init__(self):
            self.n = 0

        def ask(self, question):
            self.n += 1
            raise _status_error(500 + self.n, f"attempt-{self.n}")

    collect(adapter=Escalating(), items=[ITEM], run_dir=tmp_path, retries=2)
    import json

    row = json.loads((tmp_path / "raw.jsonl").read_text(encoding="utf-8").strip())
    assert "attempt-3" in row["error"]


@pytest.mark.parametrize(
    "exc, expect",
    [
        (httpx.ReadTimeout("timed out"), "ReadTimeout"),
        (ValueError("bad json"), "ValueError"),
    ],
)
def test_describe_error_shapes(exc, expect):
    assert expect in describe_error(exc)
