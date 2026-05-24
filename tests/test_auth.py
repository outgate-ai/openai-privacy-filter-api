"""Authentication tests — Bearer + X-API-Key, off-by-default."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from opf_api.server import create_app

from .conftest import FakeEngine

TOKEN = "s3cret-abc123"


@pytest.fixture
def auth_client() -> TestClient:
    engine = FakeEngine(
        spans_by_input={"Nothing sensitive here.": [], "hi": []}
    )
    app = create_app(
        engine=engine, model_name="openai-privacy-filter", auth_token=TOKEN
    )
    with TestClient(app) as c:
        yield c


def _chat(content: str = "Nothing sensitive here.") -> dict:
    return {
        "model": "openai-privacy-filter",
        "messages": [{"role": "user", "content": content}],
        "stream": False,
    }


def test_missing_token_rejected(auth_client):
    resp = auth_client.post("/api/chat", json=_chat())
    assert resp.status_code == 401
    assert resp.headers["www-authenticate"].lower().startswith("bearer")


def test_wrong_token_rejected(auth_client):
    resp = auth_client.post(
        "/api/chat", json=_chat(), headers={"authorization": "Bearer wrong"}
    )
    assert resp.status_code == 401


def test_valid_bearer_token_accepted(auth_client):
    resp = auth_client.post(
        "/api/chat", json=_chat(), headers={"authorization": f"Bearer {TOKEN}"}
    )
    assert resp.status_code == 200


def test_valid_api_key_header_accepted(auth_client):
    resp = auth_client.post("/api/chat", json=_chat(), headers={"x-api-key": TOKEN})
    assert resp.status_code == 200


def test_tags_requires_auth(auth_client):
    assert auth_client.get("/api/tags").status_code == 401
    assert (
        auth_client.get("/api/tags", headers={"authorization": f"Bearer {TOKEN}"}).status_code
        == 200
    )


def test_version_requires_auth(auth_client):
    assert auth_client.get("/api/version").status_code == 401
    assert (
        auth_client.get("/api/version", headers={"x-api-key": TOKEN}).status_code == 200
    )


def test_health_does_not_require_auth(auth_client):
    resp = auth_client.get("/health")
    assert resp.status_code == 200


def test_bearer_prefix_case_insensitive(auth_client):
    resp = auth_client.post(
        "/api/chat", json=_chat(), headers={"authorization": f"bearer {TOKEN}"}
    )
    assert resp.status_code == 200


def test_no_token_configured_means_open(client):
    resp = client.get("/api/tags")
    assert resp.status_code == 200
