"""/api/chat contract tests — Ollama compatibility."""

from __future__ import annotations


def _chat_request(content: str, *, stream: bool = False, model: str = "openai-privacy-filter"):
    return {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "stream": stream,
    }


def test_chat_returns_detected_spans_as_json_string(client, parsed_content):
    resp = client.post("/api/chat", json=_chat_request("Contact i@izs.me and Alice."))
    assert resp.status_code == 200
    body = resp.json()

    assert body["done"] is True
    assert body["done_reason"] == "stop"
    assert body["message"]["role"] == "assistant"
    assert body["model"] == "openai-privacy-filter"
    assert isinstance(body["created_at"], str)
    assert body["eval_count"] == 0
    assert body["prompt_eval_count"] == len("Contact i@izs.me and Alice.")
    assert body["total_duration"] >= 0

    spans = parsed_content(body)
    assert spans == [
        {"text": "i@izs.me", "category": "private_email"},
        {"text": "Alice", "category": "private_person"},
    ]


def test_chat_empty_results_returns_empty_array(client, parsed_content):
    resp = client.post("/api/chat", json=_chat_request("Nothing sensitive here."))
    assert resp.status_code == 200
    assert parsed_content(resp.json()) == []


def test_chat_rejects_stream_true(client):
    resp = client.post("/api/chat", json=_chat_request("hi", stream=True))
    assert resp.status_code == 400
    assert "stream" in resp.json()["detail"]


def test_chat_rejects_unknown_model_name(client):
    resp = client.post("/api/chat", json=_chat_request("hi", model="glm-5.1:cloud"))
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"]


def test_chat_rejects_empty_model_name(client):
    resp = client.post("/api/chat", json=_chat_request("hi", model=""))
    assert resp.status_code == 404


def test_chat_accepts_configured_model_name(client):
    resp = client.post(
        "/api/chat", json=_chat_request("Nothing sensitive here.", model="openai-privacy-filter")
    )
    assert resp.status_code == 200
    assert resp.json()["model"] == "openai-privacy-filter"


def test_chat_ignores_options(client):
    payload = {
        "model": "openai-privacy-filter",
        "messages": [{"role": "user", "content": "Nothing sensitive here."}],
        "stream": False,
        "options": {"num_predict": 5000, "temperature": 1.0, "weird_flag": True},
    }
    resp = client.post("/api/chat", json=payload)
    assert resp.status_code == 200


def test_chat_uses_last_user_message_only(client, fake_engine, parsed_content):
    payload = {
        "model": "openai-privacy-filter",
        "messages": [
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "Contact i@izs.me and Alice."},
        ],
        "stream": False,
    }
    resp = client.post("/api/chat", json=payload)
    assert resp.status_code == 200
    assert len(parsed_content(resp.json())) == 2
    assert fake_engine.redact_calls == ["Contact i@izs.me and Alice."]


def test_chat_400_when_no_user_message(client):
    payload = {
        "model": "openai-privacy-filter",
        "messages": [{"role": "system", "content": "you are helpful"}],
        "stream": False,
    }
    resp = client.post("/api/chat", json=payload)
    assert resp.status_code == 400


def test_chat_503_when_engine_not_ready(fake_engine, client):
    fake_engine.state = "loading"
    resp = client.post("/api/chat", json=_chat_request("hi"))
    assert resp.status_code == 503


def test_chat_sets_request_id_header(client):
    resp = client.post("/api/chat", json=_chat_request("Nothing sensitive here."))
    assert "x-request-id" in resp.headers
    assert len(resp.headers["x-request-id"]) > 0


def test_chat_echoes_request_id_header(client):
    resp = client.post(
        "/api/chat",
        json=_chat_request("Nothing sensitive here."),
        headers={"x-request-id": "my-trace-id"},
    )
    assert resp.headers["x-request-id"] == "my-trace-id"
