"""Shadow model: score every scanned text with a second checkpoint, off the request path.

The response always comes from the primary model. The shadow sees the same
text on a single worker thread and only per-label span counts are logged for
both models, never text, so a candidate (or the baseline) can be compared on
live traffic before it is promoted. When the queue is full the text is skipped
rather than slowing requests down.
"""

from __future__ import annotations

import logging
import queue
import threading
from collections import Counter

from .engine import DetectedSpan, Engine

logger = logging.getLogger(__name__)

QUEUE_SIZE = 16
SUMMARY_EVERY = 100


def _counts(spans: list[DetectedSpan]) -> Counter[str]:
    return Counter(s.category for s in spans)


def _fmt(c: Counter[str]) -> str:
    return ",".join(f"{k}:{v}" for k, v in sorted(c.items())) or "-"


class Shadow:
    def __init__(self, engine: Engine, *, label: str, primary_label: str) -> None:
        self.engine = engine
        self.label = label
        self.primary_label = primary_label
        self._q: queue.Queue[tuple[str, str, Counter[str]] | None] = queue.Queue(QUEUE_SIZE)
        self._thread: threading.Thread | None = None
        self.scored = 0
        self.skipped = 0
        self.differ = 0
        self.totals: dict[str, Counter[str]] = {"primary": Counter(), "shadow": Counter()}

    def start(self) -> None:
        self.engine.load()
        self._thread = threading.Thread(target=self._run, name="opf-shadow", daemon=True)
        self._thread.start()
        logger.info("shadow model %s enabled next to %s", self.label, self.primary_label)

    def stop(self) -> None:
        if self._thread is not None:
            self._q.put(None)
            self._thread.join(timeout=5)

    def submit(self, request_id: str, text: str, primary: list[DetectedSpan]) -> None:
        if self._thread is None or self.engine.state != "ready":
            return
        try:
            self._q.put_nowait((request_id, text, _counts(primary)))
        except queue.Full:
            self.skipped += 1

    def _run(self) -> None:
        while (job := self._q.get()) is not None:
            rid, text, primary = job
            try:
                shadow = _counts(self.engine.redact(text))
            except Exception as exc:  # noqa: BLE001 - the shadow never affects serving
                logger.warning("shadow redact failed rid=%s error=%s", rid, type(exc).__name__)
                continue
            self.scored += 1
            self.differ += primary != shadow
            self.totals["primary"].update(primary)
            self.totals["shadow"].update(shadow)
            logger.info(
                "shadow rid=%s chars=%d %s=%s %s=%s same=%s",
                rid, len(text), self.primary_label, _fmt(primary), self.label, _fmt(shadow),
                primary == shadow,
            )
            if self.scored % SUMMARY_EVERY == 0:
                logger.info("shadow summary %s", self.summary())

    def summary(self) -> dict:
        return {
            "primary": self.primary_label,
            "shadow": self.label,
            "scored": self.scored,
            "skipped": self.skipped,
            "differ": self.differ,
            "totals": {k: dict(sorted(v.items())) for k, v in self.totals.items()},
        }
