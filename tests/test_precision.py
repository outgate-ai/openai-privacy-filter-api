"""Precision filters: low-evidence detections are dropped, real ones are not."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from opf_api.engine import DetectedSpan
from opf_api.merge import SOURCE_BOTH, SOURCE_OPF, SOURCE_PRESIDIO, MergedDetection
from opf_api.precision import (
    DEFAULT_NOISY_CATEGORIES,
    apply_precision_filters,
    is_bare_word_secret,
    is_file_name_secret,
    is_pod_name,
    refine_spans,
)
from opf_api.presidio import DEFAULT_ENTITIES
from opf_api.server import create_app

from .conftest import FakeEngine

NOISY = frozenset(DEFAULT_NOISY_CATEGORIES)


def _filter(dets, corroborated=True, noisy=NOISY):
    return apply_precision_filters(
        dets, corroboration_available=corroborated, noisy_categories=noisy
    )


# ---------------------------------------------------------------- bare words

@pytest.mark.parametrize("word", ["outgate", "Outgate", "claude", "vince", "Deployed"])
def test_bare_word_is_not_a_secret(word):
    assert is_bare_word_secret(word)


@pytest.mark.parametrize(
    "value",
    [
        "hunter2xK9mPass",          # letters + digits
        "sk-abc123def456ghi",       # prefixed key
        "AKIAIOSFODNN7EXAMPLE",     # long, uppercase, digits
        "correcthorsebatterystaple",  # all letters but too long to be a word
    ],
)
def test_real_secret_shapes_survive(value):
    assert not is_bare_word_secret(value)


def test_bare_word_secret_is_dropped_but_a_real_one_is_kept():
    dets = [
        MergedDetection("outgate", "credentials", "secret", SOURCE_OPF),
        MergedDetection("xK9mPass22aa", "credentials", "secret", SOURCE_OPF),
    ]
    kept = _filter(dets)
    assert [d.text for d in kept] == ["xK9mPass22aa"]


def test_bare_word_rule_applies_even_without_corroboration():
    dets = [MergedDetection("outgate", "credentials", "secret", SOURCE_OPF)]
    assert _filter(dets, corroborated=False) == []


# -------------------------------------------------- uncorroborated noise

@pytest.mark.parametrize(
    "cat,value",
    [("private_date", "2026-09-14T08:12:03Z"), ("account_number", "4429183")],
)
def test_uncorroborated_noisy_category_is_dropped(cat, value):
    dets = [MergedDetection(value, "personal_information", cat, SOURCE_OPF)]
    assert _filter(dets) == []


@pytest.mark.parametrize("source", [SOURCE_BOTH, SOURCE_PRESIDIO])
def test_corroborated_noisy_category_survives(source):
    """A card confirmed by Presidio's Luhn recognizer must not be dropped."""
    dets = [MergedDetection("4111111111111111", "personal_information", "account_number", source)]
    assert len(_filter(dets)) == 1


def test_noisy_rule_is_off_when_no_second_detector_ran():
    dets = [MergedDetection("4111111111111111", "personal_information", "account_number", SOURCE_OPF)]
    assert len(_filter(dets, corroborated=False)) == 1


def test_untouched_categories_pass_through():
    dets = [
        MergedDetection("a@b.de", "personal_information", "private_email", SOURCE_OPF),
        MergedDetection("+49 151 000", "personal_information", "private_phone", SOURCE_OPF),
    ]
    assert len(_filter(dets)) == 2


def test_noisy_set_is_configurable():
    dets = [MergedDetection("2026-09-14", "personal_information", "private_date", SOURCE_OPF)]
    assert len(_filter(dets, noisy=frozenset())) == 1


# ------------------------------------------------------------- allowlist

def test_ip_and_medical_license_are_out_of_the_presidio_allowlist():
    assert "IP_ADDRESS" not in DEFAULT_ENTITIES
    assert "MEDICAL_LICENSE" not in DEFAULT_ENTITIES
    assert "CREDIT_CARD" in DEFAULT_ENTITIES and "US_SSN" in DEFAULT_ENTITIES


# -------------------------------------------------------------- endpoint

def _detections(response):
    return json.loads(response.json()["message"]["content"])["detections"]


def _post(client, text):
    return client.post(
        "/api/chat",
        json={
            "model": "openai-privacy-filter",
            "stream": False,
            "messages": [{"role": "user", "content": text}],
        },
    )


def test_endpoint_drops_a_bare_word_secret_with_the_pass_off():
    text = "clone from the outgate repo"
    engine = FakeEngine(spans_by_input={text: [DetectedSpan(text="outgate", category="secret")]})
    client = TestClient(create_app(engine=engine, model_name="openai-privacy-filter"))
    assert _detections(_post(client, text)) == []


def test_endpoint_drops_a_file_name_secret_with_the_pass_off():
    text = "onnx/model_q4f16.onnx + model_q4f16.onnx_data"
    engine = FakeEngine(
        spans_by_input={text: [DetectedSpan(text="model_q4f16.onnx_data", category="secret")]}
    )
    client = TestClient(create_app(engine=engine, model_name="openai-privacy-filter"))
    assert _detections(_post(client, text)) == []


def test_endpoint_keeps_the_bare_word_when_filters_are_disabled():
    text = "clone from the outgate repo"
    engine = FakeEngine(spans_by_input={text: [DetectedSpan(text="outgate", category="secret")]})
    client = TestClient(
        create_app(engine=engine, model_name="openai-privacy-filter", precision_filters=False)
    )
    assert len(_detections(_post(client, text))) == 1


# ------------------------------------- account_number keeps lettered ids

def test_bare_digit_run_under_account_number_is_dropped():
    for value in ["4429183", "2026-0091", "3506155597671759"]:
        dets = [MergedDetection(value, "personal_information", "account_number", SOURCE_OPF)]
        assert _filter(dets) == [], value


def test_lettered_identifier_under_account_number_survives():
    """A passport or licence number is shaped like a real identifier, so an
    uncorroborated hit still carries evidence a bare digit run does not."""
    dets = [MergedDetection("AB1234567", "personal_information", "account_number", SOURCE_OPF)]
    assert len(_filter(dets)) == 1


def test_private_date_is_dropped_regardless_of_letters():
    dets = [MergedDetection("2026-09-14T08:12:03Z", "personal_information", "private_date", SOURCE_OPF)]
    assert _filter(dets) == []


@pytest.mark.parametrize(
    "name",
    ["model_q4f16.onnx_data", "model.safetensors", "pytorch_model-00001-of-00004.bin", "partial/step.json", "tokenizer_config.json"],
)
def test_artifact_file_names_are_not_secrets(name):
    assert is_file_name_secret(name)


@pytest.mark.parametrize(
    "value",
    ["sk_live_4eC39HqLyjWDarjtT1zdp7dc.json", "AKIAIOSFODNN7EXAMPLE", "ghp_16C7e42F292c6912E7710c838347Ae178B4a.txt", "config prod.yaml", "notes"],
)
def test_key_shaped_or_non_file_values_are_not_file_names(value):
    assert not is_file_name_secret(value)


# ------------------------------------- context rules on raw spans

def test_secret_that_stops_at_the_extension_is_a_file_name():
    text = "onnx/model_q4f16.onnx + weights"
    spans = [DetectedSpan(text="onnx/model_q4f16", category="secret")]
    assert refine_spans(text, spans) == []


def test_url_before_an_extension_is_kept():
    text = "get https://dl.example.org/pkg/release_notes.zip now"
    spans = [DetectedSpan(text="https://dl.example.org/pkg/release_notes", category="secret")]
    assert refine_spans(text, spans) == spans


def test_pod_name_is_an_identifier_not_a_secret():
    text = "kubectl logs guardrail-7d9f8b6c5-x2kqp -n outgate"
    out = refine_spans(text, [DetectedSpan(text="guardrail-7d9f8b6c5-x2kqp", category="secret")])
    assert out == [DetectedSpan(text="guardrail-7d9f8b6c5-x2kqp", category="entity_id")]


@pytest.mark.parametrize("value", ["sk-live-abcdefghij-klmno", "api-xq7Zk9", "guardrail-7d9f8b6c5-x2kqpq"])
def test_secret_shapes_are_not_pod_names(value):
    assert not is_pod_name(value)


def test_endpoint_applies_context_rules():
    text = "Copied onnx/model_q4f16.onnx and restarted guardrail-7d9f8b6c5-x2kqp"
    engine = FakeEngine(spans_by_input={text: [
        DetectedSpan(text="onnx/model_q4f16", category="secret"),
        DetectedSpan(text="guardrail-7d9f8b6c5-x2kqp", category="secret"),
    ]})
    client = TestClient(create_app(engine=engine, model_name="openai-privacy-filter"))
    dets = _detections(_post(client, text))
    assert [(d["source_category"], d["category"]) for d in dets] == [("entity_id", "sensitive_data")]


def _long(n, seed="Zq8"):
    return (seed * n)[:n]


def test_long_opaque_value_under_an_id_key_becomes_a_secret():
    value = _long(44)
    text = '{"session' + '_id": "' + value + '", "ok": true}'
    assert refine_spans(text, []) == [DetectedSpan(text=value, category="secret")]


def test_keyed_long_rule_skips_hash_keys_short_values_and_covered_spans():
    sha = "ab12" * 10
    short = _long(30)
    covered = _long(48)
    text = '{"commit_id": "' + sha + '", "digestId": "' + sha + '", "id": "' + short + '", "tokenId": "' + covered + '"}'
    existing = [DetectedSpan(text=covered, category="secret")]
    assert refine_spans(text, existing) == existing


def test_hex_value_under_a_plain_id_key_is_a_secret():
    value = "9f3c" * 16
    text = '{"id": "' + value + '"}'
    assert refine_spans(text, []) == [DetectedSpan(text=value, category="secret")]


@pytest.mark.parametrize("key", ["app" + "Id", "user" + "Id", "pipeline" + "Id", "parentRun" + "Id", "v2" + "Id"])
def test_single_hump_camel_id_keys_are_keyed(key):
    value = _long(44, "Kp7")
    text = '{"' + key + '": "' + value + '"}'
    assert refine_spans(text, []) == [DetectedSpan(text=value, category="secret")]


@pytest.mark.parametrize("key", ["Id", "Valid", "payload" + "ID", "sha" + "Id", "commit" + "Id"])
def test_non_id_or_hash_camel_keys_are_not_keyed(key):
    text = '{"' + key + '": "' + _long(44, "Kp7") + '"}'
    assert refine_spans(text, []) == []


@pytest.mark.parametrize("value", ["run" + "_" + "3f9a" * 3, "app" + "_" + "0c1d2e3f4a5b", "apc" + "_" + "9e" * 10, "ab" + "_" + "deadbeef"])
def test_prefixed_short_hex_id_is_an_identifier_not_a_secret(value):
    text = "run_id: " + value + ", ok"
    assert refine_spans(text, [DetectedSpan(text=value, category="secret")]) == [DetectedSpan(text=value, category="entity_id")]


@pytest.mark.parametrize(
    "value",
    ["og" + "hk_" + "a1b2" * 8, "sk" + "_" + "Live" + "9f3c" * 4, "run" + "_" + "3f9a", "Run" + "_" + "3f9a" * 3, "a" + "_" + "3f9a" * 3],
)
def test_long_or_non_hex_prefixed_tokens_stay_secrets(value):
    span = DetectedSpan(text=value, category="secret")
    assert refine_spans("key " + value, [span]) == [span]


@pytest.mark.parametrize("prefix", ["released ", "due date: ", '{"created": "', "since "])
def test_bare_iso_date_is_a_timestamp(prefix):
    value = "20" + "26-11-08"
    text = prefix + value + " ok"
    assert refine_spans(text, [DetectedSpan(text=value, category="private_date")]) == [DetectedSpan(text=value, category="timestamp")]


@pytest.mark.parametrize("prefix", ["date of birth: ", "DOB ", "born on ", "geboren am "])
def test_iso_date_after_a_birth_cue_stays_private(prefix):
    span = DetectedSpan(text="19" + "84-02-29", category="private_date")
    assert refine_spans(prefix + span.text, [span]) == [span]


@pytest.mark.parametrize("value", ["08.11.2026", "Nov 8, 2026", "2026-13-01", "2026-11-08 trip", "1884-02-03"])
def test_other_date_shapes_stay_private(value):
    span = DetectedSpan(text=value, category="private_date")
    assert refine_spans("on " + value, [span]) == [span]
