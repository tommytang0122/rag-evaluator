"""auth / TLS 設定:秘密以環境變數名稱入 config,值只在建 client 時解析。"""

import httpx
import pytest

from rag_evaluator.adapters import nas_rag
from rag_evaluator.adapters.nas_rag import NasRagAdapter
from rag_evaluator.config import SystemConfig

BASE = {
    "adapter": "nas_rag",
    "endpoint": "http://testserver/v1/query",
    "collection_names": ["gavin_test"],
}


@pytest.fixture
def captured(monkeypatch):
    """攔截 httpx.Client 的建構參數,不實際連線。"""
    seen: dict = {}

    class FakeClient:
        def __init__(self, **kwargs):
            seen.update(kwargs)

    monkeypatch.setattr(nas_rag.httpx, "Client", FakeClient)
    return seen


def test_no_auth_by_default(captured):
    NasRagAdapter(SystemConfig(**BASE))
    assert captured["headers"] == {}
    assert captured["verify"] is True


def test_auth_header_read_from_env(captured, monkeypatch):
    monkeypatch.setenv("MY_TOKEN", "Bearer sk-123")
    NasRagAdapter(SystemConfig(**BASE, auth_env="MY_TOKEN"))
    assert captured["headers"] == {"Authorization": "Bearer sk-123"}


def test_auth_header_name_configurable(captured, monkeypatch):
    monkeypatch.setenv("MY_TOKEN", "sk-123")
    NasRagAdapter(
        SystemConfig(**BASE, auth_env="MY_TOKEN", auth_header="X-API-Key")
    )
    assert captured["headers"] == {"X-API-Key": "sk-123"}


def test_token_is_passed_through_verbatim(captured, monkeypatch):
    """不自動補 Bearer 前綴——Authorization 與 X-API-Key 用同一組欄位表達。"""
    monkeypatch.setenv("MY_TOKEN", "raw-token-no-scheme")
    NasRagAdapter(SystemConfig(**BASE, auth_env="MY_TOKEN"))
    assert captured["headers"]["Authorization"] == "raw-token-no-scheme"


@pytest.mark.parametrize("value", [None, ""])
def test_missing_env_is_a_clear_error(monkeypatch, value):
    monkeypatch.delenv("MY_TOKEN", raising=False)
    if value is not None:
        monkeypatch.setenv("MY_TOKEN", value)
    with pytest.raises(ValueError, match="MY_TOKEN"):
        NasRagAdapter(SystemConfig(**BASE, auth_env="MY_TOKEN"))


def test_verify_false_threaded_to_client(captured):
    NasRagAdapter(SystemConfig(**BASE, verify=False))
    assert captured["verify"] is False


def test_verify_accepts_ca_bundle_path(captured):
    NasRagAdapter(SystemConfig(**BASE, verify="/etc/ssl/corp-ca.pem"))
    assert captured["verify"] == "/etc/ssl/corp-ca.pem"


def test_injected_client_bypasses_auth_construction(monkeypatch):
    """既有測試以 client= 注入 MockTransport,不應被 auth 邏輯影響。"""
    monkeypatch.delenv("MY_TOKEN", raising=False)
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda r: httpx.Response(200, json={"answer": "x", "sources": []})
        )
    )
    adapter = NasRagAdapter(SystemConfig(**BASE, auth_env="MY_TOKEN"), client=client)
    assert adapter.ask("q").answer == "x"


def test_secret_value_never_lands_in_config_dump(monkeypatch):
    """manifest 存的是 model_dump()——只能有環境變數名稱,不能有值。"""
    monkeypatch.setenv("MY_TOKEN", "sk-super-secret")
    dumped = SystemConfig(**BASE, auth_env="MY_TOKEN").model_dump()
    assert dumped["auth_env"] == "MY_TOKEN"
    assert "sk-super-secret" not in str(dumped)
