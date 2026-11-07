"""Mapping from OPF native categories to guardrail risk categories.

The downstream guardrail service consumes detections under a fixed risk
taxonomy: ``personal_information | credentials | prompt_injection |
malicious_content | sensitive_data``. OPF emits a finer-grained 8-label
taxonomy (``private_email``, ``private_person``, ``secret``, …) which
collapses cleanly onto two of those categories. Fine-tuned 41-label releases
add ``entity_id`` (UUIDs, hashes, opaque record ids) and ``timestamp``; both
are machine values whose exposure is a linkage risk rather than PII, so they
map to ``sensitive_data``.

The original OPF label is preserved on each detection under the
``source_category`` field so callers can still filter or report at OPF's
native granularity.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

# Fail safe: unknown labels must land on a category that still triggers
# anonymization downstream.
DEFAULT_GUARDRAIL_CATEGORY = "personal_information"

OPF_TO_GUARDRAIL: dict[str, str] = {
    "private_person": "personal_information",
    "private_email": "personal_information",
    "private_phone": "personal_information",
    "private_address": "personal_information",
    "private_date": "personal_information",
    "account_number": "personal_information",
    "private_url": "credentials",
    "secret": "credentials",
    "entity_id": "sensitive_data",
    "timestamp": "sensitive_data",
}


def _overrides(raw: str | None) -> dict[str, str]:
    """Parse ``OPF_API_LABEL_CATEGORIES`` (``label=category,label=category``)."""
    out: dict[str, str] = {}
    for item in (raw or "").split(","):
        if not item.strip():
            continue
        label, sep, category = item.partition("=")
        if not sep or not label.strip() or not category.strip():
            raise ValueError(f"OPF_API_LABEL_CATEGORIES entry {item!r} is not label=category")
        out[label.strip()] = category.strip()
    return out


OPF_TO_GUARDRAIL.update(_overrides(os.environ.get("OPF_API_LABEL_CATEGORIES")))


def map_category(opf_category: str) -> str:
    """Return the guardrail risk category for an OPF native label.

    Falls back to ``DEFAULT_GUARDRAIL_CATEGORY`` for unknown labels and
    logs a warning so operators can spot taxonomy drift.
    """
    mapped = OPF_TO_GUARDRAIL.get(opf_category)
    if mapped is None:
        logger.warning(
            "unknown OPF category %r — falling back to %r",
            opf_category,
            DEFAULT_GUARDRAIL_CATEGORY,
        )
        return DEFAULT_GUARDRAIL_CATEGORY
    return mapped
