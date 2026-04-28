"""Input-text preprocessing applied before passing content to the engine.

Empirically, OPF's recall on documents with hard line breaks (emails, OCR'd
PDFs, German addresses, signature blocks) improves substantially when
newlines and other whitespace are flattened to single spaces — multi-line
addresses, names, and contact blocks then look like contiguous spans the
classifier can label end-to-end.

We also defensively decode common literal escape sequences (``\\n``,
``\\r``, ``\\t``) that some clients double-escape when JSON-encoding
content. Those would otherwise survive into the model input as the
two-character strings ``\\`` + ``n`` rather than real whitespace.
"""

from __future__ import annotations

import re

# Match real whitespace runs (≥1 char): regular space, tab, CR, LF, NBSP,
# zero-width space, and the rest of Unicode whitespace.
_WHITESPACE_RUN = re.compile(r"\s+")

# Match literal two-char escape sequences that arrive when content was
# double-encoded as JSON: ``\\n`` is the two characters \\ and n.
_LITERAL_ESCAPE = re.compile(r"\\[nrt]")


def normalize_whitespace(text: str) -> str:
    """Collapse all whitespace (real and literal-escaped) to single spaces.

    - Literal ``\\n``, ``\\r``, ``\\t`` (backslash + letter) → single space.
    - Any run of real whitespace (``\\n``, ``\\t``, NBSP, etc.) → single space.
    - Leading/trailing whitespace is trimmed.

    Idempotent: ``normalize_whitespace(normalize_whitespace(x)) == normalize_whitespace(x)``.
    """
    if not text:
        return text
    text = _LITERAL_ESCAPE.sub(" ", text)
    text = _WHITESPACE_RUN.sub(" ", text)
    return text.strip()
