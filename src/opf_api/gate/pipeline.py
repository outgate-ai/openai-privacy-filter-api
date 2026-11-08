# Vendored from outgate-ai/opf-bench src/opfbench/gate.py (0976e55); keep in sync, do not edit here.
"""Bit-faithful port of the og-local gate's span pipeline.

Sources (og-local @ 64abc19):
  internal/pii/decode.go        Viterbi over BIES tags with transition biases
  internal/pii/calibration.go   operating point "default" of viterbi_calibration.json
  internal/pii/redactor.go      MinDetectionLen=3 bytes, value propagation
  internal/pii/substitute.go    longest-first replace, quoted-base64 guard
  internal/redact/chunk.go      4096-byte chunks, 512-byte overlap at hard cuts
  internal/redact/redact.go     per-field detect -> merge -> apply

The gate works on UTF-8 bytes (Rust tokenizers return byte offsets); so does
this module, converting to character offsets only for scoring.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from .gosort import sort_slice

CHUNK_TARGET = 4096
CHUNK_OVERLAP = 512
MIN_DETECTION_LEN = 3
BASE64_MIN_LEN = 40

BIAS_KEYS = (
    "background_stay",
    "background_to_start",
    "end_to_background",
    "end_to_start",
    "inside_to_continue",
    "inside_to_end",
)


@dataclass(frozen=True)
class TransitionBiases:
    background_stay: float = 0.0
    background_to_start: float = 0.0
    end_to_background: float = 0.0
    end_to_start: float = 0.0
    inside_to_continue: float = 0.0
    inside_to_end: float = 0.0

    @classmethod
    def from_calibration(cls, data: bytes | str, operating_point: str = "default") -> "TransitionBiases":
        doc = json.loads(data)
        op = doc.get("operating_points", {}).get(operating_point)
        if op is None:
            raise KeyError(f"operating point {operating_point!r} not found")
        b = op.get("biases", {})
        return cls(**{k: float(b.get(f"transition_bias_{k}", 0.0)) for k in BIAS_KEYS})

    def with_overrides(self, overrides: dict[str, float] | None) -> "TransitionBiases":
        if not overrides:
            return self
        vals = {k: getattr(self, k) for k in BIAS_KEYS}
        for k, v in overrides.items():
            k = k.removeprefix("transition_bias_")
            if k not in vals:
                raise KeyError(f"unknown transition bias {k!r}")
            vals[k] = float(v)
        return TransitionBiases(**vals)


@dataclass
class Span:
    start: int
    end: int
    cls: str
    score: float = 0.0

    def key(self) -> tuple[int, int, str]:
        return (self.start, self.end, self.cls)


def _parse_tag(label: str) -> tuple[str, str]:
    if label in ("O", ""):
        return "O", ""
    dash = label.find("-")
    if dash < 0:
        return "O", ""
    return label[0], label[dash + 1 :]


class Decoder:
    def __init__(self, id2label: Sequence[str], biases: TransitionBiases):
        self.tags = [_parse_tag(lbl) for lbl in id2label]
        self.biases = biases
        k = len(self.tags)
        trans = np.full((k, k), -np.inf, dtype=np.float64)
        for p, (fp, fc) in enumerate(self.tags):
            for j, (tp, tc) in enumerate(self.tags):
                trans[p, j] = self._transition(fp, fc, tp, tc)
        self.trans = trans
        self.no_start = np.array([t[0] in ("I", "E") for t in self.tags])

    def _transition(self, fp: str, fc: str, tp: str, tc: str) -> float:
        b = self.biases
        if fp in ("O", "E", "S"):
            if tp == "O":
                return b.background_stay if fp == "O" else b.end_to_background
            if tp in ("B", "S"):
                return b.background_to_start if fp == "O" else b.end_to_start
            return -math.inf
        if fp in ("B", "I"):
            if tp in ("I", "E") and tc == fc:
                return b.inside_to_continue if tp == "I" else b.inside_to_end
            return -math.inf
        return -math.inf

    def path(self, logits: np.ndarray) -> list[int]:
        n, k = logits.shape
        if n == 0:
            return []
        lg = logits.astype(np.float64)
        score = np.where(self.no_start, -np.inf, lg[0])
        back = np.empty((n, k), dtype=np.int64)
        back[0] = -1
        cols = np.arange(k)
        for i in range(1, n):
            cand = score[:, None] + self.trans
            bp = np.argmax(cand, axis=0)  # first max == Go's strict '>' scan
            score = cand[bp, cols] + lg[i]
            back[i] = bp
        last = int(np.argmax(score))
        out = [0] * n
        for i in range(n - 1, -1, -1):
            out[i] = last
            last = int(back[i][last])
            if last < 0:
                break
        return out

    def decode(self, logits: np.ndarray, offsets: Sequence[tuple[int, int]]) -> list[Span]:
        p = self.path(logits)
        return self._assemble(p, logits, offsets)

    def _assemble(self, path: list[int], logits: np.ndarray, offsets) -> list[Span]:
        spans: list[Span] = []
        i, n = 0, len(path)
        while i < n:
            prefix, cls = self.tags[path[i]]
            if prefix == "S":
                spans.append(Span(offsets[i][0], offsets[i][1], cls, float(_prob(logits[i], path[i]))))
                i += 1
            elif prefix == "B":
                start = offsets[i][0]
                score_sum = _prob(logits[i], path[i])
                count = 1
                j = i + 1
                while j < n:
                    tp, tc = self.tags[path[j]]
                    if tp in ("I", "E") and tc == cls:
                        score_sum = np.float32(score_sum + _prob(logits[j], path[j]))
                        count += 1
                        if tp == "E":
                            spans.append(Span(start, offsets[j][1], cls, float(np.float32(score_sum / np.float32(count)))))
                            break
                        j += 1
                        continue
                    break
                i = j + 1
            else:
                i += 1
        return spans


def _prob(row: np.ndarray, idx: int) -> np.float32:
    row = row.astype(np.float32, copy=False)
    maxv = row.max()
    d = row - maxv
    s = 0.0
    for x in d:
        s += math.exp(float(x))
    return np.float32(math.exp(float(d[idx])) / s)


_SPACE = frozenset(b" \t\n\r")


def trim_spans(data: bytes, spans: list[Span]) -> list[Span]:
    out = []
    for s in spans:
        st, en = s.start, s.end
        while st < en and data[st] in _SPACE:
            st += 1
        while en > st and data[en - 1] in _SPACE:
            en -= 1
        if st < en:
            out.append(Span(st, en, s.cls, s.score))
    return out


# ---------------------------------------------------------------- chunking


@dataclass
class Chunk:
    start: int  # byte offset in the field
    data: bytes
    hard_left: bool = False
    hard_right: bool = False


def _rune_start(b: int) -> bool:
    return (b & 0xC0) != 0x80


def split_chunks(data: bytes, target: int = CHUNK_TARGET, overlap: int = CHUNK_OVERLAP) -> list[Chunk]:
    if overlap > target // 4:
        overlap = target // 4
    if len(data) <= target:
        return [Chunk(0, data)]
    out: list[Chunk] = []
    pos = 0
    hard_left = False
    while len(data) - pos > target:
        window = data[pos : pos + target]
        half = target // 2
        hard = False
        i = window[half:].rfind(b"\n")
        if i >= 0:
            cut = pos + half + i + 1
        else:
            i = window[half:].rfind(b" ")
            if i >= 0:
                cut = pos + half + i + 1
            else:
                b = pos + target
                while not _rune_start(data[b]):
                    b -= 1
                cut, hard = b, True
        out.append(Chunk(pos, data[pos:cut], hard_left, hard))
        if hard:
            pos = cut - overlap
            while not _rune_start(data[pos]):
                pos += 1
        else:
            pos = cut
        hard_left = hard
    out.append(Chunk(pos, data[pos:], hard_left, False))
    return out


def merge_spans(chunks: list[Chunk], per_chunk: list[list[Span]]) -> list[Span]:
    """Byte-offset spans in, byte-offset spans out (chunk.go mergeSpans)."""
    out: list[Span] = []
    for c, spans in zip(chunks, per_chunk):
        for s in spans:
            if c.hard_right and s.end == len(c.data):
                continue
            if c.hard_left and s.start == 0:
                continue
            out.append(Span(s.start + c.start, s.end + c.start, s.cls, s.score))
    sort_slice(out, lambda x, y: x.start < y.start if x.start != y.start else x.end > y.end)
    kept: list[Span] = []
    for s in out:
        if kept and s.start >= kept[-1].start and s.end <= kept[-1].end:
            last = kept[-1]
            if s.start == last.start and s.end == last.end and s.cls == last.cls and s.score > last.score:
                last.score = s.score
            continue
        kept.append(s)
    return kept


# ---------------------------------------------------------------- offsets


class OffsetMap:
    """char <-> UTF-8 byte offsets for one string."""

    def __init__(self, text: str):
        self.text = text
        lens = np.fromiter((len(c.encode("utf-8")) for c in text), dtype=np.int64, count=len(text))
        self.byte_of_char = np.concatenate([[0], np.cumsum(lens)])

    def to_byte(self, c: int) -> int:
        return int(self.byte_of_char[c])

    def to_char(self, b: int, side: str = "left") -> int:
        """Character index for byte offset b. A byte inside a multi-byte
        character snaps outward: starts to the left, ends to the right."""
        i = int(np.searchsorted(self.byte_of_char, b, side="left"))
        if i < len(self.byte_of_char) and self.byte_of_char[i] == b:
            return i
        return i - 1 if side == "left" else i

    def is_boundary(self, b: int) -> bool:
        i = int(np.searchsorted(self.byte_of_char, b))
        return i < len(self.byte_of_char) and self.byte_of_char[i] == b


# ---------------------------------------------------------------- redaction

_QUOTED_BASE64_B = re.compile(rb'"([A-Za-z0-9+/_-]+={0,2})"')


# Candidate gate change (not in og-local yet): region and gate placeholders,
# OG_<LABEL>_<hex>, are never re-detected or replaced.
_PLACEHOLDER_B = re.compile(rb"OG_[A-Z]+(?:_[A-Z]+)*_[0-9a-f]{6,}")


def placeholder_ranges(data: bytes) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in _PLACEHOLDER_B.finditer(data)]


def _cut_placeholders(data: bytes, spans: list[Span]) -> list[Span]:
    """Remove placeholder ranges from spans; keep the remaining pieces that
    still hold something besides whitespace and punctuation."""
    ph = placeholder_ranges(data)
    if not ph:
        return spans
    out = []
    for s in spans:
        pieces = [(s.start, s.end)]
        for a, b in ph:
            nxt = []
            for x, y in pieces:
                if a < y and x < b:
                    if x < a:
                        nxt.append((x, a))
                    if b < y:
                        nxt.append((b, y))
                else:
                    nxt.append((x, y))
            pieces = nxt
        for x, y in pieces:
            seg = data[x:y]
            lead = len(seg) - len(seg.lstrip(b" \t\r\n,;:()[]{}<>\"'"))
            trail = len(seg) - len(seg.rstrip(b" \t\r\n,;:()[]{}<>\"'"))
            x, y = x + lead, y - trail
            if y - x >= MIN_DETECTION_LEN:
                out.append(Span(x, y, s.cls, s.score))
    return out


def protected_ranges(data: bytes, placeholders: bool = False) -> list[tuple[int, int]]:
    out = []
    for m in _QUOTED_BASE64_B.finditer(data):
        if m.end(1) - m.start(1) >= BASE64_MIN_LEN:
            out.append((m.start(1), m.end(1)))
    if placeholders:
        out.extend(placeholder_ranges(data))
        out.sort()
    return out


def _in_ranges(pos: int, ranges: list[tuple[int, int]]) -> bool:
    return any(a <= pos < b for a, b in ranges)


PARITY_NONCE = b"opfbench-parity"


def gate_placeholder(cls: str, value: bytes, nonce: bytes = PARITY_NONCE) -> bytes:
    """Redactor.placeholder. The gate draws a random nonce per process; a fixed
    one keeps the benchmark deterministic and has the same length/alphabet."""
    import hashlib

    return b"OG_" + cls.upper().encode() + b"_" + hashlib.sha256(nonce + value).hexdigest()[:6].encode()


def apply_redaction(data: bytes, spans: list[Span], nonce: bytes = PARITY_NONCE, protect_placeholders: bool = False) -> tuple[bytes, list[Span]]:
    """Redactor.Apply + substitute on UTF-8 bytes. Returns the redacted bytes and
    the byte ranges of the ORIGINAL input that end up replaced. Repeats of a
    detected value elsewhere in the field are replaced too (value propagation),
    so the second list can be longer than the first."""
    seen: dict[bytes, bytes] = {}
    pairs: list[tuple[bytes, bytes, str]] = []
    if protect_placeholders:
        spans = _cut_placeholders(data, spans)
    for s in spans:
        if s.start < 0 or s.end > len(data) or s.end - s.start < MIN_DETECTION_LEN:
            continue
        value = data[s.start : s.end]
        if value in seen:
            continue
        tok = gate_placeholder(s.cls, value, nonce)
        seen[value] = tok
        pairs.append((value, tok, s.cls))

    ordered = sorted([p for p in pairs if p[0]], key=lambda p: -len(p[0]))
    out = data
    orig = list(range(len(data)))
    replaced: list[Span] = []
    for frm, to, cls in ordered:
        protected = protected_ranges(out, protect_placeholders)
        pieces: list[bytes] = []
        new_orig: list[int] = []
        n = len(frm)
        cur = 0
        while True:
            idx = out.find(frm, cur)
            if idx < 0:
                break
            if _in_ranges(idx, protected):
                pieces.append(out[cur : idx + n])
                new_orig.extend(orig[cur : idx + n])
                cur = idx + n
                continue
            pieces.append(out[cur:idx])
            new_orig.extend(orig[cur:idx])
            src = orig[idx : idx + n]
            if src[0] >= 0 and src[-1] - src[0] == n - 1 and min(src) >= 0:
                replaced.append(Span(src[0], src[-1] + 1, cls))
            pieces.append(to)
            new_orig.extend([-1] * len(to))
            cur = idx + n
        pieces.append(out[cur:])
        new_orig.extend(orig[cur:])
        out = b"".join(pieces)
        orig = new_orig
    replaced.sort(key=lambda s: (s.start, s.end))
    return out, replaced


# ---------------------------------------------------------------- pipeline

# (bytes of one chunk) -> (logits [n, labels], byte offsets per token)
LogitsFn = Callable[[bytes], tuple[np.ndarray, list[tuple[int, int]]]]


@dataclass
class FieldResult:
    model_spans: list[Span]  # merged detector spans, char offsets
    redacted_spans: list[Span]  # what the gate replaces incl. propagation, char offsets
    redacted_bytes: bytes
    chunks: int
    snapped: int  # span edges that fell inside a multi-byte character


class GatePipeline:
    """redact.Pipeline.detect + Redactor.Apply for one field, in bytes."""

    def __init__(self, logits_fn: LogitsFn, decoder: Decoder, target: int = CHUNK_TARGET, overlap: int = CHUNK_OVERLAP, protect_placeholders: bool = False):
        self.protect_placeholders = protect_placeholders
        self.post = None  # (chunk bytes, offsets, spans) -> spans, set by the harness
        self.logits_fn = logits_fn
        self.decoder = decoder
        self.target = target
        self.overlap = overlap

    def detect_chunk(self, data: bytes) -> list[Span]:
        if not data:
            return []
        logits, offsets = self.logits_fn(data)
        if len(offsets) == 0:
            return []
        spans = trim_spans(data, self.decoder.decode(logits, offsets))
        if self.post is not None:
            spans = self.post(data, offsets, spans)
        return spans

    def detect_bytes(self, data: bytes) -> tuple[list[Span], int]:
        if len(data) < MIN_DETECTION_LEN:
            return [], 0
        chunks = split_chunks(data, self.target, self.overlap)
        per_chunk = [self.detect_chunk(c.data) if len(c.data) >= MIN_DETECTION_LEN else [] for c in chunks]
        return merge_spans(chunks, per_chunk), len(chunks)

    def run(self, text: str) -> FieldResult:
        data = text.encode("utf-8")
        spans, n = self.detect_bytes(data)
        redacted, replaced = apply_redaction(data, spans, protect_placeholders=self.protect_placeholders)
        om = OffsetMap(text)
        snapped = 0

        def conv(lst: list[Span]) -> list[Span]:
            nonlocal snapped
            out = []
            for s in lst:
                snapped += (not om.is_boundary(s.start)) + (not om.is_boundary(s.end))
                out.append(Span(om.to_char(s.start, "left"), om.to_char(s.end, "right"), s.cls, s.score))
            return out

        return FieldResult(conv(spans), conv(replaced), redacted, n, snapped)
