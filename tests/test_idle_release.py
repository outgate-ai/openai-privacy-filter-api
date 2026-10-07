"""Cached GPU memory is handed back once the server has been idle long enough."""

from __future__ import annotations

import time

from fastapi.testclient import TestClient

from opf_api.server import create_app

IDLE = 0.1


def _post(client: TestClient) -> None:
    resp = client.post(
        "/api/chat",
        json={
            "model": "openai-privacy-filter",
            "messages": [{"role": "user", "content": "Nothing sensitive here."}],
        },
    )
    assert resp.status_code == 200


def test_released_once_per_idle_period(fake_engine):
    app = create_app(
        engine=fake_engine, model_name="openai-privacy-filter", idle_release_seconds=IDLE
    )
    with TestClient(app) as client:
        time.sleep(IDLE * 4)
        assert fake_engine.release_calls == 0

        _post(client)
        time.sleep(IDLE * 4)
        assert fake_engine.release_calls == 1

        time.sleep(IDLE * 4)
        assert fake_engine.release_calls == 1

        _post(client)
        time.sleep(IDLE * 4)
        assert fake_engine.release_calls == 2


def test_not_released_while_busy(fake_engine):
    app = create_app(
        engine=fake_engine, model_name="openai-privacy-filter", idle_release_seconds=IDLE * 5
    )
    with TestClient(app) as client:
        for _ in range(8):
            _post(client)
            time.sleep(IDLE)
        assert fake_engine.release_calls == 0


def test_disabled_with_zero(fake_engine):
    app = create_app(
        engine=fake_engine, model_name="openai-privacy-filter", idle_release_seconds=0
    )
    with TestClient(app) as client:
        _post(client)
        time.sleep(IDLE * 4)
        assert fake_engine.release_calls == 0


def test_release_failure_keeps_serving(fake_engine):
    def boom():
        fake_engine.release_calls += 1
        raise RuntimeError("driver hiccup")

    fake_engine.release_cached_memory = boom
    app = create_app(
        engine=fake_engine, model_name="openai-privacy-filter", idle_release_seconds=IDLE
    )
    with TestClient(app) as client:
        _post(client)
        time.sleep(IDLE * 4)
        assert fake_engine.release_calls == 1
        _post(client)
        time.sleep(IDLE * 4)
        assert fake_engine.release_calls == 2
