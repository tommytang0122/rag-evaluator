from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


class DiagnosticsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cutoff_probe_top_k: int | None = None
    type_buckets: bool = False


class SystemConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    adapter: str
    endpoint: str
    collection_names: list[str]
    top_k: int = 5
    timeout_s: float = 90
    diagnostics: DiagnosticsConfig = Field(default_factory=DiagnosticsConfig)


def load_system_config(path: Path) -> SystemConfig:
    with Path(path).open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return SystemConfig.model_validate(data)
