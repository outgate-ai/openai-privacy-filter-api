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
