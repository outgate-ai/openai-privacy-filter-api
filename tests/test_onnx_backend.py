"""ONNX backend selection: engine loading errors, model-dir detection, fetcher includes/exports."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from opf_api.config import Config
from opf_api.onnx_engine import ONNXEngine, is_onnx_model_dir

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "opf_models.py"


def _fetcher():
    spec = importlib.util.spec_from_file_location("opf_models", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["opf_models"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_backend_comes_from_env(monkeypatch):
    monkeypatch.setenv("OPF_API_BACKEND", "onnx")
    monkeypatch.setenv("OPF_API_ONNX_THREADS", "3")
    cfg = Config.from_env()
    assert (cfg.backend, cfg.onnx_threads) == ("onnx", 3)
    monkeypatch.delenv("OPF_API_BACKEND")
    assert Config.from_env().backend == "torch"


def test_model_dir_detection(tmp_path):
    assert not is_onnx_model_dir(None)
    assert not is_onnx_model_dir(tmp_path)
    (tmp_path / "onnx").mkdir()
    (tmp_path / "onnx" / "model_q4f16.onnx").write_bytes(b"")
    assert is_onnx_model_dir(tmp_path)


def test_missing_card_is_a_load_error(tmp_path):
    pytest.importorskip("onnxruntime")
    (tmp_path / "onnx").mkdir()
    (tmp_path / "onnx" / "model_q4f16.onnx").write_bytes(b"")
    eng = ONNXEngine(model_dir=str(tmp_path))
    with pytest.raises(FileNotFoundError):
        eng.load()
    assert eng.state == "error" and "config.json" in eng.error
    with pytest.raises(RuntimeError):
        eng.redact("x")


def test_fetcher_pulls_onnx_and_card_and_exports_the_model_dir(tmp_path, monkeypatch, capsys):
    m = _fetcher()
    calls = {}

    def fake_resolve(**kw):
        calls.update(kw)
        cache = tmp_path / "models" / "opf-v22-x"
        cache.mkdir(parents=True)
        (cache / "manifest.json").write_text(json.dumps({"runtime": {"dir": "runtime"}, "files": []}))
        return "opf-v22-x", cache

    monkeypatch.setattr(m, "resolve_and_fetch", fake_resolve)
    monkeypatch.setenv("OPF_API_BACKEND", "onnx")
    monkeypatch.setenv("OPF_API_MODEL_ID", "opf-v22-x")
    assert m.main(["fetch", "--exports", "--home", str(tmp_path)]) == 0
    assert calls["include"] == ("onnx/", "card/")
    out = capsys.readouterr().out
    assert f"export OPF_API_MODEL_PATH={tmp_path / 'models' / 'opf-v22-x'}" in out
    assert "OPF_API_VITERBI_CALIBRATION_PATH" not in out
