"""Optional Presidio analyzer pass layered on top of OPF detection.

OPF is the primary detector: it finds whole spans (a full connection string,
an address block) that pattern matchers cut into pieces. Presidio contributes
deterministic, checksum-backed recognizers (IBAN mod-97, card Luhn, SSN) that
do not depend on how the model feels about a given input. Running both and
merging is strictly additive — see :mod:`opf_api.merge` for the merge rule.

Disabled by default; set ``OPF_API_PRESIDIO_ENABLED=true`` and point
``OPF_API_PRESIDIO_URL`` at an analyzer service.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Entity types worth trusting without a human in the loop. NER-driven ones
# (PERSON, LOCATION, NRP) and the catch-alls (DATE_TIME, URL) are left out:
# OPF already covers people and addresses, and on ordinary engineering prose
# those recognizers fire on "Thursday", "Hamburg" and every https:// link.
DEFAULT_ENTITIES: tuple[str, ...] = (
    "CREDIT_CARD",
    "CRYPTO",
    "EMAIL_ADDRESS",
    "IBAN_CODE",
    "MAC_ADDRESS",
    "PHONE_NUMBER",
    "UK_NHS",
    "US_BANK_NUMBER",
    "US_DRIVER_LICENSE",
    "US_ITIN",
    "US_PASSPORT",
    "US_SSN",
)

# Presidio entity -> guardrail risk category. Financial identifiers follow
# OPF's own account_number mapping and land on personal_information.
# IP_ADDRESS and MEDICAL_LICENSE are deliberately out of DEFAULT_ENTITIES:
# the first is a technical identifier the guardrail's own detection criteria
# tell detectors not to flag, and the second is a loose enough pattern to
# match arbitrary hyphenated identifiers such as a UUID fragment.
PRESIDIO_TO_GUARDRAIL: dict[str, str] = {
    "CREDIT_CARD": "personal_information",
    "CRYPTO": "personal_information",
    "DATE_TIME": "personal_information",
    "EMAIL_ADDRESS": "personal_information",
    "IBAN_CODE": "personal_information",
    "IP_ADDRESS": "personal_information",
    "LOCATION": "personal_information",
    "MAC_ADDRESS": "personal_information",
    "MEDICAL_LICENSE": "personal_information",
    "NRP": "personal_information",
    "PERSON": "personal_information",
    "PHONE_NUMBER": "personal_information",
    "UK_NHS": "personal_information",
    "US_BANK_NUMBER": "personal_information",
    "US_DRIVER_LICENSE": "personal_information",
    "US_ITIN": "personal_information",
    "US_PASSPORT": "personal_information",
    "US_SSN": "personal_information",
    "URL": "credentials",
}

DEFAULT_GUARDRAIL_CATEGORY = "personal_information"


def map_entity(entity_type: str) -> str:
    mapped = PRESIDIO_TO_GUARDRAIL.get(entity_type)
    if mapped is None:
        logger.warning(
            "unknown Presidio entity %r — falling back to %r",
            entity_type,
            DEFAULT_GUARDRAIL_CATEGORY,
        )
        return DEFAULT_GUARDRAIL_CATEGORY
    return mapped


@dataclass(frozen=True)
class PresidioSpan:
    text: str
    entity_type: str
    score: float


class PresidioClient:
    """Thin async client for the Presidio analyzer's ``/analyze`` endpoint."""

    def __init__(
        self,
        *,
        url: str,
        language: str = "en",
        score_threshold: float = 0.5,
        entities: tuple[str, ...] | None = DEFAULT_ENTITIES,
        timeout_ms: int = 3000,
    ) -> None:
        self._url = url.rstrip("/")
        self._language = language
        self._score_threshold = score_threshold
        # None means "every entity the analyzer supports" (OPF_API_PRESIDIO_ENTITIES=*).
        self._entities = list(entities) if entities else None
        self._timeout = timeout_ms / 1000
        self._client = None

    async def analyze(self, text: str) -> list[PresidioSpan]:
        """Return spans for ``text``. Raises on transport or protocol errors —
        the caller decides whether that is fatal (see presidio_fail_open)."""
        import httpx

        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)

        payload: dict[str, object] = {
            "text": text,
            "language": self._language,
            "score_threshold": self._score_threshold,
        }
        if self._entities:
            payload["entities"] = self._entities

        response = await self._client.post(f"{self._url}/analyze", json=payload)
        response.raise_for_status()
        results = response.json()
        if not isinstance(results, list):
            raise ValueError(f"unexpected analyzer response type: {type(results).__name__}")

        spans: list[PresidioSpan] = []
        for item in results:
            start, end = item.get("start"), item.get("end")
            if not isinstance(start, int) or not isinstance(end, int):
                continue
            # Offsets index the text we sent, so the span is sliced here rather
            # than trusted from the response.
            spans.append(
                PresidioSpan(
                    text=text[start:end],
                    entity_type=str(item.get("entity_type", "")),
                    score=float(item.get("score", 0.0)),
                )
            )
        return spans

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
