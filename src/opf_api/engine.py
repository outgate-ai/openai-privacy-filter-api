"""OPF engine wrapper: eager startup load with checkpoint-missing detection."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Literal, Protocol

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT_DIR = Path.home() / ".opf" / "privacy_filter"


@dataclass(frozen=True)
class DetectedSpan:
    text: str
    category: str


class Engine(Protocol):
    @property
    def state(self) -> str: ...
    @property
    def error(self) -> str | None: ...
    def load(self) -> None: ...
    def redact(self, text: str) -> list[DetectedSpan]: ...


class OPFEngine:
    """Eagerly loads the OPF model at startup and exposes a simple redact() API."""

    def __init__(
        self,
        *,
        device: Literal["cpu", "cuda"] = "cuda",
        model_path: str | None = None,
        context_window_length: int | None = None,
        output_mode: Literal["typed", "redacted"] = "typed",
        decode_mode: Literal["viterbi", "argmax"] = "viterbi",
        viterbi_calibration_path: str | None = None,
    ) -> None:
        self._device = device
        self._model_path = model_path
        self._context_window_length = context_window_length
        self._output_mode = output_mode
        self._decode_mode = decode_mode
        self._viterbi_calibration_path = viterbi_calibration_path
        self._opf = None
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

            checkpoint_path = self._resolve_checkpoint_path()
            if not self._checkpoint_present(checkpoint_path):
                logger.warning(
                    "Privacy Filter checkpoint not found at %s — downloading from Hugging Face. "
                    "This only happens on first run and may take a few minutes.",
                    checkpoint_path,
                )
            else:
                logger.info("Loading Privacy Filter checkpoint from %s", checkpoint_path)

            try:
                from opf._api import OPF

                self._opf = OPF(
                    model=self._model_path,
                    device=self._device,
                    output_mode=self._output_mode,
                    decode_mode=self._decode_mode,
                    context_window_length=self._context_window_length,
                )
                if self._viterbi_calibration_path and self._decode_mode == "viterbi":
                    self._opf.set_viterbi_decoder(
                        calibration_path=self._viterbi_calibration_path
                    )
                self._opf.get_runtime()
                self._state = "ready"
                logger.info(
                    "Privacy Filter ready (device=%s, n_ctx=%s, output_mode=%s, decode_mode=%s)",
                    self._device,
                    self._context_window_length,
                    self._output_mode,
                    self._decode_mode,
                )
            except Exception as exc:
                self._state = "error"
                self._error = str(exc)
                logger.exception("Failed to load Privacy Filter")
                raise

    def redact(self, text: str) -> list[DetectedSpan]:
        if self._state != "ready" or self._opf is None:
            raise RuntimeError(f"engine not ready (state={self._state})")
        result = self._opf.redact(text)
        return [DetectedSpan(text=s.text, category=s.label) for s in result.detected_spans]

    def _resolve_checkpoint_path(self) -> Path:
        if self._model_path:
            return Path(self._model_path).expanduser()
        env_value = os.environ.get("OPF_CHECKPOINT")
        if env_value:
            return Path(env_value).expanduser()
        return DEFAULT_CHECKPOINT_DIR

    @staticmethod
    def _checkpoint_present(path: Path) -> bool:
        return path.is_dir() and (path / "config.json").exists()
