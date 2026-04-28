"""Whitespace normalization helper."""

from __future__ import annotations

from opf_api.preprocess import normalize_whitespace


def test_collapses_real_newlines():
    assert normalize_whitespace("hello\nworld") == "hello world"


def test_collapses_runs_of_whitespace():
    assert normalize_whitespace("a  b\t\tc\n\nd") == "a b c d"


def test_preserves_single_spaces():
    assert normalize_whitespace("hello world") == "hello world"


def test_handles_literal_escape_sequences():
    # Two-char sequences "\\n", "\\r", "\\t" — caused by double-encoded JSON.
    assert normalize_whitespace("hello\\nworld") == "hello world"
    assert normalize_whitespace("a\\nb\\tc\\rd") == "a b c d"


def test_mixed_real_and_literal():
    assert normalize_whitespace("a\\nb\nc") == "a b c"


def test_strips_leading_trailing_whitespace():
    assert normalize_whitespace("  hello  ") == "hello"
    assert normalize_whitespace("\nhello\n") == "hello"


def test_empty_string():
    assert normalize_whitespace("") == ""


def test_unicode_whitespace_collapses():
    # NBSP (\u00a0) and zero-width space (\u200b) — the regex \s only covers
    # NBSP. We don't aggressively normalize zero-width chars.
    assert normalize_whitespace("foo\u00a0bar") == "foo bar"


def test_idempotent():
    text = "Sehr geehrter\nHerr Khoramshahi,\n\nMercedes-Benz AG\nSalzufer 1\n10587 Berlin"
    once = normalize_whitespace(text)
    twice = normalize_whitespace(once)
    assert once == twice
    assert "\n" not in once


def test_german_address_block_becomes_one_line():
    text = "Andreas Fölsch\nMercedes-Benz AG\nSalzufer 1\n10587 Berlin"
    assert (
        normalize_whitespace(text)
        == "Andreas Fölsch Mercedes-Benz AG Salzufer 1 10587 Berlin"
    )
