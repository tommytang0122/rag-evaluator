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
    # 認證:只存「從哪個環境變數讀」,不存 token 本身——run manifest 會把整份
    # system_config 原樣落盤(run_manifest.py),明文秘密會跟著進版控與報表。
    auth_env: str | None = None
    auth_header: str = "Authorization"
    # True=正常驗證,False=略過(自簽憑證),字串=CA bundle 路徑
    verify: bool | str = True
    diagnostics: DiagnosticsConfig = Field(default_factory=DiagnosticsConfig)


def load_system_config(path: Path) -> SystemConfig:
    with Path(path).open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return SystemConfig.model_validate(data)
