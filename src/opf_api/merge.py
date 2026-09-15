"""Merge OPF and Presidio detections into one detection list.

The two detectors disagree about span boundaries far more often than about
whether something is sensitive. OPF returns the whole thing — a connection
string with the password inside it, a signature block with the name and the
phone; Presidio returns the part its recognizer matched. Emitting both would
hand the guardrail two overlapping substitutions of the same value, so the
longer span wins and the shorter one is folded into it, contributing only its
provenance.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

SOURCE_OPF = "opf"
SOURCE_PRESIDIO = "presidio"
SOURCE_BOTH = "opf+presidio"


@dataclass(frozen=True)
class MergedDetection:
    text: str
    category: str
    source_category: str
    source: str

    def as_dict(self) -> dict[str, str]:
        return {
            "text": self.text,
            "category": self.category,
            "source_category": self.source_category,
            "source": self.source,
        }


def _combine_source(existing: str, incoming: str) -> str:
    if existing == incoming:
        return existing
    return SOURCE_BOTH


def merge_detections(
    detections: list[MergedDetection], *, min_length: int = 3
) -> list[MergedDetection]:
    """Collapse overlapping detections, keeping the longest span of each value.

    Input order is irrelevant: candidates are considered longest-first, so a
    shorter span is always folded into the longer one that contains it, never
    the other way round. Comparison is case-insensitive on stripped text;
    the surviving detection keeps its own original casing and category.
    """
    candidates = [d for d in detections if len(d.text.strip()) >= min_length]
    # Longest first so a shorter span never wins; OPF ahead of Presidio on a
    # tie, because its category is the more specific of the two taxonomies.
    candidates.sort(
        key=lambda d: (-len(d.text.strip()), 0 if d.source == SOURCE_OPF else 1, d.text)
    )

    kept: list[MergedDetection] = []
    for candidate in candidates:
        needle = candidate.text.strip().casefold()
        for index, existing in enumerate(kept):
            haystack = existing.text.strip().casefold()
            if needle == haystack or needle in haystack:
                merged_source = _combine_source(existing.source, candidate.source)
                if merged_source != existing.source:
                    kept[index] = replace(existing, source=merged_source)
                break
        else:
            kept.append(candidate)
    return kept
