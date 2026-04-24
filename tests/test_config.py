"""Config resolution from env vars and CLI overrides."""

from __future__ import annotations

import pytest

from opf_api.config import Config


def test_defaults_when_no_env(monkeypatch):
    for key in (
        "OPF_API_HOST",
        "OPF_API_PORT",
        "OPF_API_DEVICE",
        "OPF_API_MODEL_PATH",
        "OPF_API_MODEL_NAME",
        "OPF_API_LOG_LEVEL",
        "OPF_API_CONTEXT_WINDOW_LENGTH",
        "OPF_API_OUTPUT_MODE",
        "OPF_API_DECODE_MODE",
        "OPF_API_VITERBI_CALIBRATION_PATH",
        "OPF_API_AUTH_TOKEN",
    ):
        monkeypatch.delenv(key, raising=False)

    cfg = Config.from_env()
    assert cfg.host == "127.0.0.1"
    assert cfg.port == 11435
    assert cfg.device == "cuda"
    assert cfg.model_path is None
    assert cfg.model_name == "openai-privacy-filter"
    assert cfg.log_level == "info"
    assert cfg.context_window_length == 131072
    assert cfg.output_mode == "typed"
    assert cfg.decode_mode == "viterbi"
    assert cfg.viterbi_calibration_path is None
    assert cfg.auth_token is None


def test_auth_token_from_env(monkeypatch):
    monkeypatch.setenv("OPF_API_AUTH_TOKEN", "s3cret")
    cfg = Config.from_env()
    assert cfg.auth_token == "s3cret"


def test_context_window_above_max_rejected(monkeypatch):
    monkeypatch.setenv("OPF_API_CONTEXT_WINDOW_LENGTH", "200000")
    with pytest.raises(ValueError):
        Config.from_env()


def test_invalid_output_mode_rejected(monkeypatch):
    monkeypatch.setenv("OPF_API_OUTPUT_MODE", "weird")
    with pytest.raises(ValueError):
        Config.from_env()


def test_override_rejects_invalid_context_window(monkeypatch):
    for key in (
        "OPF_API_CONTEXT_WINDOW_LENGTH",
        "OPF_API_OUTPUT_MODE",
        "OPF_API_DECODE_MODE",
    ):
        monkeypatch.delenv(key, raising=False)
    cfg = Config.from_env()
    with pytest.raises(ValueError):
        cfg.override(context_window_length=0)


def test_env_values_applied(monkeypatch):
    monkeypatch.setenv("OPF_API_HOST", "0.0.0.0")
    monkeypatch.setenv("OPF_API_PORT", "8080")
    monkeypatch.setenv("OPF_API_DEVICE", "cpu")
    monkeypatch.setenv("OPF_API_MODEL_NAME", "my-pf")
    monkeypatch.setenv("OPF_API_LOG_LEVEL", "debug")

    cfg = Config.from_env()
    assert cfg.host == "0.0.0.0"
    assert cfg.port == 8080
    assert cfg.device == "cpu"
    assert cfg.model_name == "my-pf"
    assert cfg.log_level == "debug"


def test_invalid_device_rejected(monkeypatch):
    monkeypatch.setenv("OPF_API_DEVICE", "tpu")
    with pytest.raises(ValueError):
        Config.from_env()


def test_invalid_port_rejected(monkeypatch):
    monkeypatch.setenv("OPF_API_PORT", "abc")
    with pytest.raises(ValueError):
        Config.from_env()


def test_cli_overrides_env(monkeypatch):
    monkeypatch.setenv("OPF_API_PORT", "9000")
    monkeypatch.setenv("OPF_API_DEVICE", "cpu")
    cfg = Config.from_env().override(port=1234, device="cuda")
    assert cfg.port == 1234
    assert cfg.device == "cuda"
