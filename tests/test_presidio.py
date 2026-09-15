"""Presidio pass: client contract, merging, and endpoint behaviour."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from opf_api.engine import DetectedSpan
from opf_api.merge import (
    SOURCE_BOTH,
    SOURCE_OPF,
    SOURCE_PRESIDIO,
    MergedDetection,
    merge_detections,
)
from opf_api.presidio import DEFAULT_ENTITIES, PresidioSpan, map_entity
from opf_api.server import create_app

from .conftest import FakeEngine


class FakePresidio:
    """Stands in for PresidioClient; returns canned spans or raises."""

    def __init__(self, spans_by_input=None, error: Exception | None = None) -> None:
        self._spans = spans_by_input or {}
        self._error = error
        self.calls: list[str] = []
        self.closed = False

    async def analyze(self, text: str) -> list[PresidioSpan]:
        self.calls.append(text)
        if self._error is not None:
            raise self._error
        return list(self._spans.get(text, []))

    async def aclose(self) -> None:
        self.closed = True


def _client(engine, presidio) -> TestClient:
    app = create_app(engine=engine, presidio_client=presidio, model_name="openai-privacy-filter")
    return TestClient(app)


def _detections(response) -> list[dict]:
    return json.loads(response.json()["message"]["content"])["detections"]


def test_longer_opf_span_absorbs_the_presidio_substring():
    host = "db.internal"
    conn = "postgres://svc:" + "hunter2" + "@" + host + ":5432/ledger"
    merged = merge_detections(
        [
            MergedDetection(conn, "credentials", "secret", SOURCE_OPF),
            MergedDetection(host, "personal_information", "URL", SOURCE_PRESIDIO),
        ]
    )
    assert len(merged) == 1
    assert merged[0].text == conn
    assert merged[0].category == "credentials"
    assert merged[0].source == SOURCE_BOTH


def test_presidio_only_value_survives_the_merge():
    name = "Ada " + "Lovelace"
    iban = "DE89" + "37040044" + "0532013000"
    merged = merge_detections(
        [
            MergedDetection(name, "personal_information", "private_person", SOURCE_OPF),
            MergedDetection(iban, "personal_information", "IBAN_CODE", SOURCE_PRESIDIO),
        ]
    )
    assert {d.text for d in merged} == {name, iban}
    assert {d.source for d in merged} == {SOURCE_OPF, SOURCE_PRESIDIO}


def test_same_value_from_both_sources_is_reported_once():
    value = "ada.lovelace" + "@" + "example.org"
    merged = merge_detections(
        [
            MergedDetection(value, "personal_information", "private_email", SOURCE_OPF),
            # Same value, different casing: still one detection.
            MergedDetection(value.upper(), "personal_information", "EMAIL_ADDRESS", SOURCE_PRESIDIO),
        ]
    )
    assert len(merged) == 1
    assert merged[0].text == value
    assert merged[0].source_category == "private_email"
    assert merged[0].source == SOURCE_BOTH


def test_merge_drops_spans_under_the_minimum_length():
    merged = merge_detections(
        [MergedDetection("ab", "personal_information", "PERSON", SOURCE_PRESIDIO)], min_length=3
    )
    assert merged == []


def test_endpoint_merges_both_detectors():
    surname = "Lovelace"
    full_name = "Ada " + surname
    iban = "DE89" + "37040044" + "0532013000"
    text = f"Wire to {iban} for {full_name}"
    engine = FakeEngine(
        spans_by_input={text: [DetectedSpan(text=full_name, category="private_person")]}
    )
    presidio = FakePresidio(
        spans_by_input={
            text: [
                PresidioSpan(text=iban, entity_type="IBAN_CODE", score=1.0),
                # Only the surname: the shorter span of a value OPF got whole.
                PresidioSpan(text=surname, entity_type="PERSON", score=0.85),
            ]
        }
    )
    with _client(engine, presidio) as client:
        response = client.post(
            "/api/chat",
            json={
                "model": "openai-privacy-filter",
                "messages": [{"role": "user", "content": text}],
            },
        )
    assert response.status_code == 200
    by_text = {d["text"]: d for d in _detections(response)}
    assert set(by_text) == {full_name, iban}
    assert by_text[full_name]["source"] == SOURCE_BOTH
    assert by_text[full_name]["source_category"] == "private_person"
    assert by_text[iban]["source"] == SOURCE_PRESIDIO
    assert by_text[iban]["category"] == "personal_information"


def test_analyzer_failure_degrades_to_opf_only_when_failing_open():
    email = "ada" + "@" + "example.org"
    text = f"Contact {email}"
    engine = FakeEngine(spans_by_input={text: [DetectedSpan(text=email, category="private_email")]})
    presidio = FakePresidio(error=RuntimeError("connection refused"))
    with _client(engine, presidio) as client:
        response = client.post(
            "/api/chat",
            json={
                "model": "openai-privacy-filter",
                "messages": [{"role": "user", "content": text}],
            },
        )
    assert response.status_code == 200
    assert [d["text"] for d in _detections(response)] == [email]


def test_analyzer_failure_is_fatal_when_failing_closed():
    text = "Contact " + "ada" + "@" + "example.org"
    engine = FakeEngine(spans_by_input={text: []})
    presidio = FakePresidio(error=RuntimeError("connection refused"))
    app = create_app(
        engine=engine,
        presidio_client=presidio,
        model_name="openai-privacy-filter",
        presidio_fail_open=False,
    )
    with TestClient(app) as client:
        response = client.post(
            "/api/chat",
            json={
                "model": "openai-privacy-filter",
                "messages": [{"role": "user", "content": text}],
            },
        )
    assert response.status_code == 503


def test_openai_endpoint_runs_the_presidio_pass_too():
    card = "4539" + "4512" + "0398" + "7356"
    content = f"card {card}"
    scan_input = f"[user] {content}"
    engine = FakeEngine(spans_by_input={scan_input: []})
    presidio = FakePresidio(
        spans_by_input={
            scan_input: [PresidioSpan(text=card, entity_type="CREDIT_CARD", score=1.0)]
        }
    )
    with _client(engine, presidio) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "openai-privacy-filter",
                "messages": [{"role": "user", "content": content}],
            },
        )
    assert response.status_code == 200
    body = json.loads(response.json()["choices"][0]["message"]["content"])
    assert body["detections"][0]["source_category"] == "CREDIT_CARD"
    assert presidio.calls == [scan_input]


def test_pass_stays_off_by_default_and_output_keeps_its_legacy_shape(client, parsed_content):
    response = client.post(
        "/api/chat",
        json={
            "model": "openai-privacy-filter",
            "messages": [{"role": "user", "content": "Token sk-abc and visit https://x.io/secret"}],
        },
    )
    detections = parsed_content(response.json())
    assert [d["text"] for d in detections] == ["sk-abc", "https://x.io/secret"]
    assert all("source" not in d for d in detections)


@pytest.mark.parametrize(
    ("entity", "expected"),
    [("IBAN_CODE", "personal_information"), ("URL", "credentials"), ("WAT", "personal_information")],
)
def test_entity_category_mapping(entity, expected):
    assert map_entity(entity) == expected


def test_default_entity_allowlist_excludes_the_noisy_recognizers():
    for noisy in ("PERSON", "LOCATION", "DATE_TIME", "URL", "NRP"):
        assert noisy not in DEFAULT_ENTITIES
