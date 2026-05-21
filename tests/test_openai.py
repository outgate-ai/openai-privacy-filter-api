"""/v1/chat/completions contract tests — OpenAI compatibility."""

from __future__ import annotations

import json


def _openai_request(messages, *, stream: bool = False, model: str = "openai-privacy-filter"):
    return {"model": model, "messages": messages, "stream": stream}


def _detections(body) -> list[dict]:
    payload = json.loads(body["choices"][0]["message"]["content"])
    return payload["detections"]


def test_openai_chat_envelope_shape(client):
    resp = client.post(
        "/v1/chat/completions",
        json=_openai_request([{"role": "user", "content": "Nothing sensitive here."}]),
    )
    assert resp.status_code == 200
    body = resp.json()

    assert body["object"] == "chat.completion"
    assert body["model"] == "openai-privacy-filter"
    assert body["id"].startswith("chatcmpl-")
    assert len(body["choices"]) == 1
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["choices"][0]["finish_reason"] == "stop"
    # Wrapped detections payload, even when empty.
    assert _detections(body) == []


def test_openai_chat_scans_every_message_not_only_last_user(client, fake_engine):
    payload = _openai_request(
        [
            {"role": "system", "content": "credential sk-abc lives in the env"},
            {"role": "user", "content": "Just say hi"},
            {"role": "assistant", "content": "Hello"},
        ]
    )
    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200
    # The redactor saw the system + user + assistant content joined,
    # not just the last user turn.
    seen = fake_engine.redact_calls[-1]
    assert "sk-abc" in seen
    assert "[system]" in seen
    assert "[assistant]" in seen


def test_openai_chat_handles_array_content_parts(client, fake_engine):
    payload = _openai_request(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Contact Alice."},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,zzz"}},
                ],
            }
        ]
    )
    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200
    # Only the text part reached the redactor; the image_url was dropped.
    seen = fake_engine.redact_calls[-1]
    assert "Alice" in seen
    assert "zzz" not in seen


def test_openai_chat_rejects_stream_true(client):
    resp = client.post(
        "/v1/chat/completions",
        json=_openai_request([{"role": "user", "content": "hi"}], stream=True),
    )
    assert resp.status_code == 400


def test_openai_chat_rejects_unknown_model(client):
    resp = client.post(
        "/v1/chat/completions",
        json=_openai_request([{"role": "user", "content": "hi"}], model="gpt-5"),
    )
    assert resp.status_code == 404


def test_openai_chat_400_when_no_text(client):
    resp = client.post(
        "/v1/chat/completions",
        json=_openai_request(
            [
                {
                    "role": "user",
                    "content": [{"type": "image_url", "image_url": {"url": "data:..."}}],
                }
            ]
        ),
    )
    assert resp.status_code == 400


def test_openai_chat_503_when_engine_not_ready(fake_engine, client):
    fake_engine.state = "loading"
    resp = client.post(
        "/v1/chat/completions",
        json=_openai_request([{"role": "user", "content": "hi"}]),
    )
    assert resp.status_code == 503


def test_openai_chat_filters_short_detections(client, fake_engine):
    from opf_api.engine import DetectedSpan

    user_text = "Short detections like A and BC should drop"
    scan_input = f"[user] {user_text}"
    fake_engine._spans_by_input[scan_input] = [
        DetectedSpan(text="A", category="private_person"),
        DetectedSpan(text="BC", category="private_person"),
        DetectedSpan(text="Alice", category="private_person"),
    ]
    resp = client.post(
        "/v1/chat/completions",
        json=_openai_request([{"role": "user", "content": user_text}]),
    )
    assert resp.status_code == 200
    detections = _detections(resp.json())
    assert [d["text"] for d in detections] == ["Alice"]


def test_openai_chat_alias_without_v1_prefix(client):
    # Some clients hit /chat/completions without the v1 prefix; serve
    # both paths from the same handler.
    resp = client.post(
        "/chat/completions",
        json=_openai_request([{"role": "user", "content": "Nothing sensitive here."}]),
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "chat.completion"
    assert _detections(body) == []
