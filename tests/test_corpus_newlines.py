"""Corpora are CRLF, and Python text mode silently rewrites them.

The token caches built before 2026-09-29 were encoded through
token_io.encode_file_stream(), which opens the source in text mode with the
default newline=None. That converts every "\r\n" to "\n" before the tokenizer
sees it, so the cache is not reproducible from the source file: same file, same
vocab, different tokens.

These tests pin the behaviour that caused it, so a future refactor of the text
read path cannot quietly reintroduce it.
"""
from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(ROOT, "scripts")
for path in (ROOT, SCRIPTS):
    if path not in sys.path:
        sys.path.insert(0, path)

CRLF = b"line one\r\nline two\r\n"
LF = b"line one\nline two\n"


class _FakeTok:
    """Records exactly what text it was handed, one 'token' per char."""

    def __init__(self):
        self.seen: list[str] = []

    def encode(self, text):
        self.seen.append(text)
        return [ord(c) for c in text]


def test_text_mode_default_strips_cr():
    """The bug, stated plainly: this is why the old caches are not faithful."""
    with open("corpus.bin", "wb") as f:
        f.write(CRLF)
    with open("corpus.bin", encoding="utf-8", errors="replace") as f:
        assert f.read() == "line one\nline two\n"
    with open("corpus.bin", encoding="utf-8", errors="replace", newline="") as f:
        assert f.read() == "line one\r\nline two\r\n"


def test_binary_read_preserves_every_byte(tmp_path):
    src = tmp_path / "c.bin"
    src.write_bytes(CRLF)
    assert src.read_bytes() == CRLF
    assert len(src.read_bytes()) == len(CRLF)


def test_encode_file_stream_reads_bytes_not_text(monkeypatch, tmp_path):
    """encode_file_stream must hand the tokenizer the CR-preserving text.

    If this regresses, every future token cache is subtly wrong in the same
    direction, and nothing downstream can detect it -- the file is valid, the
    model trains, only the corpus is not the one on disk.
    """
    import token_io

    src = tmp_path / "corpus.bin"
    src.write_bytes(CRLF * 4)
    dst = tmp_path / "out.bin"

    tok = _FakeTok()
    token_io.encode_file_stream(str(src), "unused.json", str(dst), tok=tok)

    handed = "".join(tok.seen)
    assert "\r\n" in handed, "CR was stripped: the token stream will not match the source"
    assert "\r\n" not in handed.replace("\r\n", ""), "bare CR in output"


def test_lf_corpus_is_unaffected(tmp_path):
    """An LF-only corpus must tokenize identically either way."""
    import token_io

    src = tmp_path / "lf.bin"
    src.write_bytes(LF * 4)
    dst = tmp_path / "out.bin"

    tok = _FakeTok()
    token_io.encode_file_stream(str(src), "unused.json", str(dst), tok=tok)
    assert "\r" not in "".join(tok.seen)


def test_boundaries_are_newline_aligned_and_contiguous():
    """Parallel splitting relies on both, and neither is obvious later."""
    from tokenize_pile_parallel import boundaries

    src = os.path.join(ROOT, "data", "_par_src.txt")
    if not os.path.exists(src):
        pytest.skip("no sample corpus on this box")

    n = 8
    b = boundaries(src, n)
    total = os.path.getsize(src)
    assert b[0] == 0
    assert b[-1] == total
    assert all(x < y for x, y in zip(b, b[1:])), "ranges must be strictly increasing"

    with open(src, "rb") as f:
        for cut in b[1:-1]:
            f.seek(cut - 1)
            assert f.read(1) == b"\n", f"cut at {cut} is not newline-aligned"
