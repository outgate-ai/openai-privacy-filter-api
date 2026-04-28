"""Shared test fixtures with a fake engine so tests don't load real model weights."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from opf_api.engine import DetectedSpan
from opf_api.server import create_app


class FakeEngine:
    """Minimal in-memory engine implementing the Engine Protocol."""

    def __init__(self, spans_by_input: dict[str, list[DetectedSpan]] | None = None) -> None:
        self._spans_by_input = spans_by_input or {}
        self.state: str = "ready"
        self.error: str | None = None
        self.load_calls = 0
        self.redact_calls: list[str] = []

    def load(self) -> None:
        self.load_calls += 1
        self.state = "ready"

    def redact(self, text: str) -> list[DetectedSpan]:
        self.redact_calls.append(text)
        if self.state != "ready":
            raise RuntimeError(f"engine not ready (state={self.state})")
        return list(self._spans_by_input.get(text, []))


@pytest.fixture
def fake_engine() -> FakeEngine:
    return FakeEngine(
        spans_by_input={
            "Contact i@izs.me and Alice.": [
                DetectedSpan(text="i@izs.me", category="private_email"),
                DetectedSpan(text="Alice", category="private_person"),
            ],
            "Token sk-abc and visit https://x.io/secret": [
                DetectedSpan(text="sk-abc", category="secret"),
                DetectedSpan(text="https://x.io/secret", category="private_url"),
            ],
            "Nothing sensitive here.": [],
        }
    )


@pytest.fixture
def client(fake_engine: FakeEngine) -> TestClient:
    app = create_app(engine=fake_engine, model_name="openai-privacy-filter")
    with TestClient(app) as c:
        yield c


@pytest.fixture
def parsed_content():
    """Parse the JSON-encoded assistant message content and return the
    detections list from the wrapped ``{"detections": [...]}`` shape."""

    def _parse(response_json: dict) -> list[dict]:
        body = json.loads(response_json["message"]["content"])
        assert isinstance(body, dict) and "detections" in body, (
            f"expected wrapped {{detections: [...]}} payload, got {body!r}"
        )
        return body["detections"]

    return _parse
