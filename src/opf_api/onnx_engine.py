"""CPU engine for the q4f16 ONNX export of a released model (no torch at inference).

A released model directory holds ``onnx/model_q4f16.onnx`` (+ ``.onnx_data``) and
``card/`` with the label space (``config.json``), ``tokenizer.json``,
``viterbi_calibration.json`` and the gate options (``gate.json``). Decoding is
the gate's span pipeline: 4096-byte chunks, Viterbi over BIES tags with the
calibrated transition biases, then value propagation; this is what opf-bench
scores, so the numbers on a model card hold for this engine.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from threading import Lock
from typing import Literal

from .engine import DetectedSpan

logger = logging.getLogger(__name__)

ONNX_FILE = "onnx/model_q4f16.onnx"


def _byte_level_alphabet() -> dict[str, int]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1))
    bs += list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs, strict=True)}


class _ByteTokenizer:
    """o200k with BYTE offsets, rebuilt from the byte-level token strings."""

    def __init__(self, path: Path):
        import tokenizers

        self.tk = tokenizers.Tokenizer.from_file(str(path))
        self._alpha = _byte_level_alphabet()

    def encode_bytes(self, data: bytes):
        import numpy as np

        enc = self.tk.encode(data.decode("utf-8"), add_special_tokens=False)
        offsets: list[tuple[int, int]] = []
        pos = 0
        for tok in enc.tokens:
            if not all(ch in self._alpha for ch in tok):
                raise ValueError("token outside the byte-level alphabet")
            offsets.append((pos, pos + len(tok)))
            pos += len(tok)
        if pos != len(data):
            raise ValueError(f"byte offsets cover {pos} of {len(data)} bytes")
        return np.asarray(enc.ids, dtype=np.int64), offsets


def _labels(config: dict) -> list[str]:
    id2label = config["id2label"]
    out = [""] * len(id2label)
    for k, v in id2label.items():
        out[int(k)] = v
    return out


class ONNXEngine:
    """Implements the Engine protocol on onnxruntime's CPU provider."""

    def __init__(self, *, model_dir: str, threads: int = 0) -> None:
        self._model_dir = Path(model_dir)
        self._threads = threads
        self._pipeline = None
        self._state: Literal["unloaded", "loading", "ready", "error"] = "unloaded"
        self._error: str | None = None
        self._lock = Lock()

    @property
    def state(self) -> str:
        return self._state

    @property
    def error(self) -> str | None:
        return self._error

    def load(self) -> None:
        with self._lock:
            if self._state == "ready":
                return
            self._state = "loading"
            try:
                self._pipeline = self._build()
            except Exception as exc:
                self._state = "error"
                self._error = f"{type(exc).__name__}: {exc}"
                logger.error("ONNX engine failed to load from %s: %s", self._model_dir, self._error)
                raise
            self._state = "ready"
            logger.info("ONNX engine ready (%s, threads=%s)", self._model_dir, self._threads or "auto")

    def _build(self):
        import numpy as np
        import onnxruntime as ort

        from .gate.pipeline import Decoder, GatePipeline, TransitionBiases

        card = self._model_dir / "card"
        onnx_path = self._model_dir / ONNX_FILE
        for p in (onnx_path, card / "config.json", card / "tokenizer.json"):
            if not p.is_file():
                raise FileNotFoundError(f"missing {p}")
        labels = _labels(json.loads((card / "config.json").read_text()))
        cal = card / "viterbi_calibration.json"
        biases = TransitionBiases.from_calibration(cal.read_bytes()) if cal.is_file() else TransitionBiases()
        gate_file = card / "gate.json"
        gate = json.loads(gate_file.read_text()).get("gate", {}) if gate_file.is_file() else {}

        so = ort.SessionOptions()
        if self._threads > 0:
            so.intra_op_num_threads = self._threads
        session = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
        inputs = [i.name for i in session.get_inputs() if i.name in ("input_ids", "attention_mask")]
        tokenizer = _ByteTokenizer(card / "tokenizer.json")

        def logits_fn(data: bytes):
            ids, offsets = tokenizer.encode_bytes(data)
            if len(ids) == 0:
                return np.zeros((0, len(labels)), dtype=np.float32), []
            x = ids.reshape(1, -1)
            feeds = {n: (np.ones_like(x) if n == "attention_mask" else x) for n in inputs}
            return session.run(None, feeds)[0][0], offsets

        return GatePipeline(
            logits_fn,
            Decoder(labels, biases),
            protect_placeholders=bool(gate.get("protect_placeholders")),
        )

    def redact(self, text: str) -> list[DetectedSpan]:
        if self._state != "ready" or self._pipeline is None:
            raise RuntimeError(f"engine not ready (state={self._state})")
        res = self._pipeline.run(text)
        seen: set[tuple[int, int, str]] = set()
        out: list[DetectedSpan] = []
        for s in sorted(res.redacted_spans, key=lambda s: (s.start, s.end)):
            key = (s.start, s.end, s.cls)
            if key in seen or s.end <= s.start:
                continue
            seen.add(key)
            out.append(DetectedSpan(text=text[s.start : s.end], category=s.cls))
        return out

    def release_cached_memory(self) -> tuple[int, int] | None:
        return None


def is_onnx_model_dir(path: str | os.PathLike | None) -> bool:
    return bool(path) and (Path(path) / ONNX_FILE).is_file()
