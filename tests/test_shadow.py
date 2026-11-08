"""The shadow model sees every scanned text off the request path and logs counts only."""

from __future__ import annotations

import logging
import time

from fastapi.testclient import TestClient

from opf_api.engine import DetectedSpan
from opf_api.server import create_app

from .conftest import FakeEngine

TEXT = "Contact i@izs.me and Alice."


def _wait(cond, timeout=2.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end and not cond():
        time.sleep(0.01)
    return cond()


def _chat(client: TestClient, text: str):
    return client.post(
        "/api/chat",
        json={"model": "openai-privacy-filter", "messages": [{"role": "user", "content": text}]},
    )


def test_shadow_scores_the_same_text_and_logs_no_text(fake_engine, parsed_content, caplog):
    shadow = FakeEngine({TEXT: [DetectedSpan(text="Alice", category="private_person")]})
    app = create_app(
        engine=fake_engine, model_name="openai-privacy-filter",
        shadow_engine=shadow, shadow_label="opf-v0-baseline", primary_label="opf-v21",
    )
    with caplog.at_level(logging.INFO, logger="opf_api.shadow"), TestClient(app) as client:
        resp = _chat(client, TEXT)
        assert resp.status_code == 200
        assert {d["category"] for d in parsed_content(resp.json())} == {"personal_information"}
        assert len(parsed_content(resp.json())) == 2
        assert _wait(lambda: client.get("/api/shadow").json()["scored"] == 1)
        summary = client.get("/api/shadow").json()

    assert shadow.load_calls == 1 and shadow.redact_calls == [TEXT]
    assert summary["differ"] == 1
    assert summary["totals"] == {
        "primary": {"private_email": 1, "private_person": 1},
        "shadow": {"private_person": 1},
    }
    line = next(r.getMessage() for r in caplog.records if r.getMessage().startswith("shadow rid="))
    assert "opf-v21=private_email:1,private_person:1" in line
    assert "opf-v0-baseline=private_person:1" in line
    assert "Alice" not in caplog.text and "OG_PII" not in caplog.text


def test_shadow_failure_never_affects_the_response(fake_engine, parsed_content):
    class Broken(FakeEngine):
        def redact(self, text):
            raise RuntimeError("boom")

    app = create_app(engine=fake_engine, model_name="openai-privacy-filter", shadow_engine=Broken())
    with TestClient(app) as client:
        resp = _chat(client, TEXT)
        assert resp.status_code == 200
        assert len(parsed_content(resp.json())) == 2


def test_shadow_load_failure_serves_without_it(fake_engine):
    class Unloadable(FakeEngine):
        def load(self):
            raise RuntimeError("no checkpoint")

    app = create_app(engine=fake_engine, model_name="openai-privacy-filter", shadow_engine=Unloadable())
    with TestClient(app) as client:
        assert _chat(client, TEXT).status_code == 200


def test_no_shadow_endpoint_without_a_shadow(client):
    assert client.get("/api/shadow").status_code == 404
