import pytest
from pydantic import ValidationError

from rag_evaluator.config import SystemConfig, load_system_config

YAML = """
adapter: nas_rag
endpoint: http://localhost:8020/v1/query
collection_names: [hr]
top_k: 5
timeout_s: 90
diagnostics:
  cutoff_probe_top_k: 20
  type_buckets: true
"""


def test_load_full_config(tmp_path):
    p = tmp_path / "sys.yaml"
    p.write_text(YAML, encoding="utf-8")
    cfg = load_system_config(p)
    assert cfg.adapter == "nas_rag"
    assert cfg.diagnostics.cutoff_probe_top_k == 20
    assert cfg.diagnostics.type_buckets is True


def test_diagnostics_default_off(tmp_path):
    p = tmp_path / "sys.yaml"
    p.write_text(
        "adapter: nas_rag\nendpoint: http://x\ncollection_names: [a]\n",
        encoding="utf-8",
    )
    cfg = load_system_config(p)
    assert cfg.top_k == 5
    assert cfg.diagnostics.cutoff_probe_top_k is None
    assert cfg.diagnostics.type_buckets is False


def test_unknown_key_rejected():
    with pytest.raises(ValidationError):
        SystemConfig.model_validate(
            {
                "adapter": "nas_rag",
                "endpoint": "http://x",
                "collection_names": ["a"],
                "typo_key": 1,
            }
        )
