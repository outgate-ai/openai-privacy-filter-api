"""Runtime configuration resolved from CLI args and environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

from . import MODEL_NAME_DEFAULT


def _env_str(key: str, default: str) -> str:
    value = os.environ.get(key)
    return value if value is not None and value != "" else default


def _env_int(key: str, default: int) -> int:
    value = os.environ.get(key)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"env var {key} must be an integer (got {value!r})") from exc


@dataclass(frozen=True)
class Config:
    host: str
    port: int
    device: Literal["cpu", "cuda"]
    model_path: str | None
    model_name: str
    log_level: str

    @classmethod
    def from_env(cls) -> Config:
        return cls(
            host=_env_str("OPF_API_HOST", "127.0.0.1"),
            port=_env_int("OPF_API_PORT", 11435),
            device=_coerce_device(_env_str("OPF_API_DEVICE", "cuda")),
            model_path=os.environ.get("OPF_API_MODEL_PATH") or None,
            model_name=_env_str("OPF_API_MODEL_NAME", MODEL_NAME_DEFAULT),
            log_level=_env_str("OPF_API_LOG_LEVEL", "info"),
        )

    def override(
        self,
        *,
        host: str | None = None,
        port: int | None = None,
        device: str | None = None,
        model_path: str | None = None,
        model_name: str | None = None,
        log_level: str | None = None,
    ) -> Config:
        return Config(
            host=host if host is not None else self.host,
            port=port if port is not None else self.port,
            device=_coerce_device(device) if device is not None else self.device,
            model_path=model_path if model_path is not None else self.model_path,
            model_name=model_name if model_name is not None else self.model_name,
            log_level=log_level if log_level is not None else self.log_level,
        )


def _coerce_device(value: str) -> Literal["cpu", "cuda"]:
    if value not in ("cpu", "cuda"):
        raise ValueError(f"device must be 'cpu' or 'cuda' (got {value!r})")
    return value  # type: ignore[return-value]
