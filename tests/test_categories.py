"""Lock the OPF -> guardrail category mapping."""

from __future__ import annotations

import logging

import pytest

from opf_api.categories import (
    DEFAULT_GUARDRAIL_CATEGORY,
    OPF_TO_GUARDRAIL,
    map_category,
)

ALL_OPF_LABELS = [
    "private_person",
    "private_email",
    "private_phone",
    "private_address",
    "private_date",
    "account_number",
    "private_url",
    "secret",
]


@pytest.mark.parametrize("label", ALL_OPF_LABELS)
def test_every_known_label_is_in_the_map(label):
    assert label in OPF_TO_GUARDRAIL


@pytest.mark.parametrize("label", ALL_OPF_LABELS)
def test_every_known_label_maps_to_a_guardrail_category(label):
    assert map_category(label) in {"personal_information", "credentials"}


def test_secret_and_url_map_to_credentials():
    assert map_category("secret") == "credentials"
    assert map_category("private_url") == "credentials"


@pytest.mark.parametrize(
    "label",
    [
        "private_person",
        "private_email",
        "private_phone",
        "private_address",
        "private_date",
        "account_number",
    ],
)
def test_pii_labels_map_to_personal_information(label):
    assert map_category(label) == "personal_information"


def test_unknown_label_falls_back_safely_with_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="opf_api.categories"):
        result = map_category("future_new_category")
    assert result == DEFAULT_GUARDRAIL_CATEGORY
    assert any(
        "future_new_category" in rec.getMessage() for rec in caplog.records
    )
