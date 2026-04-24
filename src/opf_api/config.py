"""Runtime configuration resolved from CLI args and environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

from . import MODEL_NAME_DEFAULT

CONTEXT_WINDOW_LENGTH_MAX = 131072
CONTEXT_WINDOW_LENGTH_DEFAULT = CONTEXT_WINDOW_LENGTH_MAX

OUTPUT_MODES = ("typed", "redacted")
DECODE_MODES = ("viterbi", "argmax")


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


def _env_choice(key: str, choices: tuple[str, ...], default: str) -> str:
    value = os.environ.get(key)
    if value is None or value == "":
        return default
    if value not in choices:
        raise ValueError(
            f"env var {key} must be one of {choices} (got {value!r})"
        )
    return value


@dataclass(frozen=True)
class Config:
    host: str
    port: int
    device: Literal["cpu", "cuda"]
    model_path: str | None
    model_name: str
    log_level: str
    context_window_length: int
    output_mode: Literal["typed", "redacted"]
    decode_mode: Literal["viterbi", "argmax"]
    viterbi_calibration_path: str | None
    auth_token: str | None

    @classmethod
    def from_env(cls) -> Config:
        ctx = _env_int(
            "OPF_API_CONTEXT_WINDOW_LENGTH", CONTEXT_WINDOW_LENGTH_DEFAULT
        )
        _validate_context_window(ctx)
        return cls(
            host=_env_str("OPF_API_HOST", "127.0.0.1"),
            port=_env_int("OPF_API_PORT", 11435),
            device=_coerce_device(_env_str("OPF_API_DEVICE", "cuda")),
            model_path=os.environ.get("OPF_API_MODEL_PATH") or None,
            model_name=_env_str("OPF_API_MODEL_NAME", MODEL_NAME_DEFAULT),
            log_level=_env_str("OPF_API_LOG_LEVEL", "info"),
            context_window_length=ctx,
            output_mode=_coerce_output_mode(
                _env_choice("OPF_API_OUTPUT_MODE", OUTPUT_MODES, "typed")
            ),
            decode_mode=_coerce_decode_mode(
                _env_choice("OPF_API_DECODE_MODE", DECODE_MODES, "viterbi")
            ),
            viterbi_calibration_path=os.environ.get("OPF_API_VITERBI_CALIBRATION_PATH")
            or None,
            auth_token=os.environ.get("OPF_API_AUTH_TOKEN") or None,
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
        context_window_length: int | None = None,
        output_mode: str | None = None,
        decode_mode: str | None = None,
        viterbi_calibration_path: str | None = None,
        auth_token: str | None = None,
    ) -> Config:
        if context_window_length is not None:
            _validate_context_window(context_window_length)
        return Config(
            host=host if host is not None else self.host,
            port=port if port is not None else self.port,
            device=_coerce_device(device) if device is not None else self.device,
            model_path=model_path if model_path is not None else self.model_path,
            model_name=model_name if model_name is not None else self.model_name,
            log_level=log_level if log_level is not None else self.log_level,
            context_window_length=(
                context_window_length
                if context_window_length is not None
                else self.context_window_length
            ),
            output_mode=(
                _coerce_output_mode(output_mode)
                if output_mode is not None
                else self.output_mode
            ),
            decode_mode=(
                _coerce_decode_mode(decode_mode)
                if decode_mode is not None
                else self.decode_mode
            ),
            viterbi_calibration_path=(
                viterbi_calibration_path
                if viterbi_calibration_path is not None
                else self.viterbi_calibration_path
            ),
            auth_token=auth_token if auth_token is not None else self.auth_token,
        )


def _validate_context_window(value: int) -> None:
    if value <= 0:
        raise ValueError("context_window_length must be positive")
    if value > CONTEXT_WINDOW_LENGTH_MAX:
        raise ValueError(
            f"context_window_length must be <= {CONTEXT_WINDOW_LENGTH_MAX} (got {value})"
        )


def _coerce_output_mode(value: str) -> Literal["typed", "redacted"]:
    if value not in OUTPUT_MODES:
        raise ValueError(f"output_mode must be one of {OUTPUT_MODES} (got {value!r})")
    return value  # type: ignore[return-value]


def _coerce_decode_mode(value: str) -> Literal["viterbi", "argmax"]:
    if value not in DECODE_MODES:
        raise ValueError(f"decode_mode must be one of {DECODE_MODES} (got {value!r})")
    return value  # type: ignore[return-value]


def _coerce_device(value: str) -> Literal["cpu", "cuda"]:
    if value not in ("cpu", "cuda"):
        raise ValueError(f"device must be 'cpu' or 'cuda' (got {value!r})")
    return value  # type: ignore[return-value]
