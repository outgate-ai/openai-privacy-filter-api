"""Precision filters for detections whose shape alone cannot carry the claim.

Two failure modes show up repeatedly in evaluation, both from the model
generalizing a pattern past the point where it still means anything:

* ``private_date`` and ``account_number`` fire on machine output — ISO
  timestamps in a log line, an invoice number, a MAC address, any long run of
  digits. The values that genuinely matter under those labels (a card, an
  IBAN, an SSN) are also matched by Presidio's checksum-backed recognizers, so
  when the Presidio pass is running an uncorroborated hit carries no evidence
  beyond "this looked digit-shaped".

* ``secret`` fires on bare dictionary words — a company or product name in a
  file path, a hostname, an image reference. A credential that is nothing but
  letters has close to no entropy, so the shape is not a secret regardless of
  what the label says.

Everything here is opt-out (``OPF_API_PRECISION_FILTERS=false``) and the
category set is configurable, because each rule trades recall for precision
and the right point differs per deployment.
"""

from __future__ import annotations

import logging

from .merge import SOURCE_OPF, MergedDetection

logger = logging.getLogger(__name__)

# Uncorroborated hits under these labels are dropped when a second detector
# was available to corroborate them and did not.
DEFAULT_NOISY_CATEGORIES: tuple[str, ...] = ("private_date", "account_number")

# A `secret` of only letters, no longer than this, is treated as a word.
BARE_WORD_MAX_LEN = 12


def is_bare_word_secret(text: str) -> bool:
    """True for a single all-alphabetic token short enough to be a word."""
    return text.isalpha() and len(text) <= BARE_WORD_MAX_LEN



def _carries_no_structure(source_category: str, text: str) -> bool:
    """Whether an uncorroborated hit under a noisy label has nothing behind it.

    ``account_number`` is the label the model over-generalizes onto any long run
    of digits — an invoice number, a ticket id, a build number. An identifier
    that also contains letters (a passport or licence number) is shaped like a
    real one, so it is kept even without corroboration; a bare digit run is not.
    """
    if source_category == "account_number":
        return not any(ch.isalpha() for ch in text)
    return True


def apply_precision_filters(
    detections: list[MergedDetection],
    *,
    corroboration_available: bool,
    noisy_categories: frozenset[str],
) -> list[MergedDetection]:
    """Drop low-evidence detections, returning the list to emit.

    ``corroboration_available`` says whether a second detector actually ran;
    with only one detector an uncorroborated hit is the only kind there is,
    so the noisy-category rule stays off and recall is unchanged.
    """
    kept: list[MergedDetection] = []
    dropped: list[tuple[str, str]] = []

    for det in detections:
        if (
            corroboration_available
            and det.source == SOURCE_OPF
            and det.source_category in noisy_categories
            and _carries_no_structure(det.source_category, det.text)
        ):
            dropped.append((det.source_category, "uncorroborated"))
            continue
        if det.source_category == "secret" and is_bare_word_secret(det.text):
            dropped.append((det.source_category, "bare-word"))
            continue
        kept.append(det)

    if dropped:
        logger.debug(
            "precision filters dropped %d detection(s): %s",
            len(dropped),
            ", ".join(f"{cat}/{why}" for cat, why in dropped),
        )
    return kept
