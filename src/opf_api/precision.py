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
  what the label says. It also fires on artifact file names
  (``model_q4f16.onnx_data``, ``pytorch_model-00001-of-00004.bin``); a name made
  of short word-like parts and ending in a known file extension is a file, not
  a credential.

Everything here is opt-out (``OPF_API_PRECISION_FILTERS=false``) and the
category set is configurable, because each rule trades recall for precision
and the right point differs per deployment.
"""

from __future__ import annotations

import logging
import re

from .engine import DetectedSpan
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


# A word right after a credential key (``password = "sunshine"``) is the
# credential, however low its entropy: on CredData 223 of 240 such bare-word
# secrets are gold. Those are kept.
_CREDENTIAL_KEY = re.compile(
    r"(?:pass(?:word|wd|phrase)?|pwd|secret|token|api[_-]?key|apikey|auth[_-]?key|credentials?|private[_-]?key|key)"
    r"[\w.-]{0,20}[\"']?\s*(?:[:=]|==|=>|:=)\s*[\"'`]?$",
    re.I,
)
_CREDENTIAL_KEY_WINDOW = 40


def is_dropped_bare_word(text: str, value: str, start: int) -> bool:
    """A bare-word ``secret`` that no credential key introduces."""
    if not is_bare_word_secret(value):
        return False
    return start < 0 or not _CREDENTIAL_KEY.search(text[max(0, start - _CREDENTIAL_KEY_WINDOW) : start])



FILE_EXTENSIONS = frozenset(
    ["onnx", "onnx_data", "safetensors", "bin", "pt", "pth", "ckpt", "gguf", "h5", "npz", "npy", "json", "jsonl", "yaml", "yml", "toml", "lock", "log", "txt", "md", "csv", "tsv", "tar", "gz", "tgz", "zip", "whl", "so", "dylib", "dll", "py", "js", "ts", "go", "rs", "java"]
)
# Parts of a file name longer than this, or mixing letters and digits beyond
# it, look like key material rather than words.
FILE_PART_MAX_LEN = 16
_FILE_SPLIT = re.compile(r"[._/\\-]")


def is_file_name_secret(text: str) -> bool:
    """True for a word-like file name ending in a known extension."""
    if any(ch.isspace() for ch in text) or "." not in text:
        return False
    stem, _, ext = text.rpartition(".")
    if ext.lower() not in FILE_EXTENSIONS and not text.lower().endswith(".onnx_data"):
        return False
    parts = [p for p in _FILE_SPLIT.split(stem) if p]
    if not parts:
        return False
    for part in parts:
        if len(part) > FILE_PART_MAX_LEN:
            return False
        if not (part.isalpha() or part.isdigit()) and len(part) > 8:
            return False
    return True


_EXT_AFTER = re.compile(r"\.[A-Za-z0-9_]{1,12}")

# Kubernetes pod names: <workload>-<replicaset hash>-<5-char suffix>, both drawn
# from the vowel-free alphabet the controllers use, so words never match.
_K8S_ALPHABET = "bcdfghjklmnpqrstvwxz2456789"
_POD_NAME = re.compile(
    rf"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*-[{_K8S_ALPHABET}]{{6,10}}-[{_K8S_ALPHABET}]{{5}}$"
)


# Resource ids made of a short lowercase prefix and a short hex body (run_<12 hex>,
# app_<12 hex>, apc_<20 hex>). The model tags them `secret`; on opf-bench dev all
# 75 such spans are outside gold. Tokens with a 32-hex body under the same kind of
# prefix are gold secrets, so the body length is capped (at 16: two review-set
# credentials have a 20-hex body).
_PREFIXED_HEX_ID = re.compile(r"^[a-z]{2,10}_[0-9a-f]{8,16}$")


def is_file_stem_secret(text: str, value: str, end: int) -> bool:
    """True when ``value`` ends right before a file extension in ``text`` and the
    whole thing is a word-like file name (the model often stops at the dot)."""
    m = _EXT_AFTER.match(text, end)
    return bool(m) and "://" not in value and "@" not in value and is_file_name_secret(value + m.group())


def is_pod_name(text: str) -> bool:
    return _POD_NAME.match(text) is not None


def is_prefixed_hex_id(text: str) -> bool:
    return _PREFIXED_HEX_ID.match(text) is not None


# A bare ISO calendar date tagged ``private_date`` is almost always a machine
# date (a release, a due date, a log day): on opf-bench dev 692 of 703 such
# spans are outside gold. It is relabelled ``timestamp`` (still anonymized under
# sensitive_data) unless a birth cue precedes it.
_ISO_DATE = re.compile(r"^(?:19|20)\d{2}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])$")
_BIRTH_CUE = re.compile(r"birth|\bdob\b|\bborn\b|geboren|\bgeb\.", re.I)
_BIRTH_WINDOW = 40


def is_machine_iso_date(text: str, value: str, start: int) -> bool:
    return _ISO_DATE.match(value) is not None and not _BIRTH_CUE.search(text[max(0, start - _BIRTH_WINDOW) : start])


# A bare run of digits tagged ``account_number`` is almost always a machine id
# (a ticket, an epoch, a build or invoice number): on opf-bench dev 860 of 862
# such spans are outside gold. Card numbers pass the Luhn check, so a 13-19
# digit run that does is kept; everything else is relabelled ``entity_id``.
_DIGIT_RUN = re.compile(r"^[0-9]+$")


def _luhn_ok(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        n = int(ch)
        if i % 2:
            n = n * 2 - 9 if n > 4 else n * 2
        total += n
    return total % 10 == 0


# UUIDs and 7-12 char lowercase hex strings (short commit ids) tagged
# ``account_number`` are machine ids too: 58 such spans on dev, none in gold.
# 16-hex values are left alone; gold labels some of those as accounts.
_HEX_ID = re.compile(
    r"^(?:[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}|(?=[0-9a-f]*[a-f])(?=[0-9a-f]*[0-9])[0-9a-f]{7,12})$"
)


def is_digit_run_id(text: str) -> bool:
    """True for a bare digit run that cannot be a payment card number, a UUID,
    or a short hex id."""
    if _HEX_ID.match(text):
        return True
    return _DIGIT_RUN.match(text) is not None and not (13 <= len(text) <= 19 and _luhn_ok(text))


# Assistant and model product names the model reads as a first name ("ask
# Claude", "pull Qwen"). On opf-bench dev 187 single-token person spans are one
# of these, none in gold. Names that are also common first names (Jules, Devin,
# Alexa) are left out.
AI_PRODUCT_NAMES = frozenset(
    ["Claude", "Codex", "Gemini", "Copilot", "ChatGPT", "Grok", "Bard", "Llama", "Ollama", "Mistral", "Qwen", "DeepSeek", "Kimi", "Perplexity", "Cortana"]
)


# The model often stops an email inside the address (before the ``@domain`` or
# the last label) when it sits in a path or next to punctuation. On opf-bench
# dev ~70 gold addresses are cut short that way; the whole address is added as
# a span, and the partial one stays only if its text also occurs elsewhere.
_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")


def widen_email(text: str, start: int, end: int) -> tuple[int, int]:
    """The ``[start, end)`` range widened to every email address it overlaps."""
    lo, hi = start, end
    for m in _EMAIL.finditer(text, max(0, start - 254), min(len(text), end + 254)):
        a, b = m.span()
        if a < end and start < b:
            lo, hi = min(lo, a), max(hi, b)
    return lo, hi


def enclosing_email(text: str, start: int, end: int) -> tuple[int, int]:
    """The email address that strictly contains ``[start, end)``, else the range itself.

    The model sometimes labels part of an address as ``secret`` or a person
    (the local part in a path); such a span becomes the whole address as
    ``private_email``.
    """
    for m in _EMAIL.finditer(text, max(0, start - 254), min(len(text), end + 254)):
        a, b = m.span()
        if a <= start and end <= b and (a, b) != (start, end):
            return a, b
    return start, end


# A long opaque value under a JSON id key ("token_id": "<40+ base64/hex chars>")
# is a credential the model never tags (opf-bench dev: 41 such gold secrets,
# 13 % of v22's leaked characters, 3 look-alikes; hex values included, gold
# labels those credentials too). A key that names a hash (commit_id, sha_id,
# digestId) holds a commit or content hash, not a credential. camelCase keys
# with a single hump (appId, userId) carry 41 more such gold secrets on dev.
_KEYED_LONG = re.compile(r'"([a-z][a-z0-9]*(?:[A-Z][a-z0-9]*)*Id|[a-z_]*_id|id)"\s*:\s*"([A-Za-z0-9_\-+/=]{40,})"')
_HASH_KEY = re.compile(r"commit|sha|hash|digest|checksum|revision|etag", re.I)


def _is_slug_or_path(value: str) -> bool:
    """A lowercase dash-slug (``build-cache-linux-x64-…``) or a value mixing ``-``
    with ``/`` is a name or path, not key material: 28 of CredData's 33
    non-credential keyed values, none of dev's gold ones."""
    if "-" in value and "/" in value:
        return True
    return "-" in value and re.fullmatch(r"[a-z0-9-]+", value) is not None


def keyed_long_secrets(text: str, taken: list[tuple[int, int]]) -> list[DetectedSpan]:
    """Secrets for long opaque values under id keys that no detection touches."""
    out = []
    for m in _KEYED_LONG.finditer(text):
        a, b = m.span(2)
        if _HASH_KEY.search(m.group(1)) or _is_slug_or_path(m.group(2)) or any(s < b and a < e for s, e in taken):
            continue
        out.append(DetectedSpan(text=m.group(2), category="secret"))
    return out


def refine_spans(
    text: str,
    spans: list[DetectedSpan],
    *,
    iso_dates_as_timestamp: bool = False,
    digit_runs_as_entity_id: bool = False,
) -> list[DetectedSpan]:
    """Context rules on raw model spans, before any merging.

    A ``secret`` that is the stem of a word-like file name is dropped, as is a
    bare-word ``secret`` that no credential key introduces, and a
    ``private_person`` that is just an AI product name (Claude, Qwen); a
    ``private_email`` cut short inside an address, or any span the model put
    inside one, is replaced by the whole address as ``private_email`` (kept
    too if its text occurs elsewhere); a ``secret`` shaped like a Kubernetes pod name is an identifier, not a
    credential, and is relabelled ``entity_id`` (still anonymized under
    sensitive_data), as is a prefixed short hex id (``run_<12 hex>``). Long
    opaque values under JSON id keys that nothing covers are added as ``secret``.

    Two opt-in rules move spans between masked categories, which a policy that
    disables sensitive_data would unmask: ``iso_dates_as_timestamp`` (a bare ISO
    ``private_date`` without a birth cue becomes ``timestamp``) and
    ``digit_runs_as_entity_id`` (a bare digit-run ``account_number`` that fails
    the card checksum, a UUID or a short hex id becomes ``entity_id``).
    """
    out: list[DetectedSpan] = []
    taken: list[tuple[int, int]] = []
    widened: dict[str, DetectedSpan] = {}
    cursor = 0
    for span in spans:
        start = text.find(span.text, cursor)
        if start < 0:
            start = text.find(span.text)
        end = start + len(span.text) if start >= 0 else -1
        if start >= 0:
            cursor = start
            taken.append((start, end))
        if span.category == "secret" and end >= 0 and is_file_stem_secret(text, span.text, end):
            continue
        if span.category == "secret" and is_dropped_bare_word(text, span.text, start):
            continue
        if start >= 0:
            widen = widen_email if span.category == "private_email" else enclosing_email
            lo, hi = widen(text, start, end)
            if (lo, hi) != (start, end):
                address = text[lo:hi]
                if address not in widened:
                    widened[address] = DetectedSpan(text=address, category="private_email")
                    taken.append((lo, hi))
                if span.text not in text.replace(address, "\0"):
                    continue
        if span.category == "private_person" and span.text in AI_PRODUCT_NAMES:
            continue
        if span.category == "secret" and (is_pod_name(span.text) or is_prefixed_hex_id(span.text)):
            span = DetectedSpan(text=span.text, category="entity_id")
        if (
            iso_dates_as_timestamp
            and span.category == "private_date"
            and start >= 0
            and is_machine_iso_date(text, span.text, start)
        ):
            span = DetectedSpan(text=span.text, category="timestamp")
        if digit_runs_as_entity_id and span.category == "account_number" and is_digit_run_id(span.text):
            span = DetectedSpan(text=span.text, category="entity_id")
        out.append(span)
    return out + list(widened.values()) + keyed_long_secrets(text, taken)


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
        if det.source_category == "secret" and is_file_name_secret(det.text):
            dropped.append((det.source_category, "file-name"))
            continue
        kept.append(det)

    if dropped:
        logger.debug(
            "precision filters dropped %d detection(s): %s",
            len(dropped),
            ", ".join(f"{cat}/{why}" for cat, why in dropped),
        )
    return kept
