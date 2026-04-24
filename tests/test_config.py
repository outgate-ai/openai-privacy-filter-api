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
    ):
        monkeypatch.delenv(key, raising=False)

    cfg = Config.from_env()
    assert cfg.host == "127.0.0.1"
    assert cfg.port == 11435
    assert cfg.device == "cuda"
    assert cfg.model_path is None
    assert cfg.model_name == "openai-privacy-filter"
    assert cfg.log_level == "info"


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
