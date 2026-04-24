"""Wrapper around the OPF Python API with startup-time model load and download checks."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Literal

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT_DIR = Path.home() / ".opf" / "privacy_filter"


@dataclass
class DetectedSpanOut:
    text: str
    category: str


class OPFEngine:
    """Lazy-but-eager wrapper around opf.OPF.

    Loads the model once at startup. If the checkpoint directory is missing,
    logs a clear "downloading" message before the (slow) HuggingFace pull.
    """

    def __init__(self, *, device: Literal["cpu", "cuda"] = "cuda", model_path: str | None = None) -> None:
        self._device = device
        self._model_path = model_path
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
                    output_mode="typed",
                )
                self._opf.get_runtime()
                self._state = "ready"
                logger.info("Privacy Filter ready (device=%s)", self._device)
            except Exception as exc:
                self._state = "error"
                self._error = str(exc)
                logger.exception("Failed to load Privacy Filter")
                raise

    def redact(self, text: str) -> list[DetectedSpanOut]:
        if self._state != "ready" or self._opf is None:
            raise RuntimeError(f"engine not ready (state={self._state})")
        result = self._opf.redact(text)
        spans = []
        for span in result.detected_spans:
            spans.append(DetectedSpanOut(text=span.text, category=span.label))
        return spans

    def _resolve_checkpoint_path(self) -> Path:
        if self._model_path:
            return Path(self._model_path).expanduser()
        env_value = os.environ.get("OPF_CHECKPOINT")
        if env_value:
            return Path(env_value).expanduser()
        return DEFAULT_CHECKPOINT_DIR

    @staticmethod
    def _checkpoint_present(path: Path) -> bool:
        if not path.exists() or not path.is_dir():
            return False
        if not (path / "config.json").exists():
            return False
        return True
