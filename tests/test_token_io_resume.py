"""encode_file_stream(--resume) must APPEND, never overwrite the prefix.

Regression test for a bug that shipped in f5713c3 and was never caught: the
resume path opened dst with "r+b", which leaves the write pointer at byte 0.
With no seek to the end, the first tofile after resuming landed on top of the
tokens already on disk.

Measured on a 4,399-char source resumed at 2,000: the cache came back with
2,399 ids instead of 4,399, and the first id was the source's 10th character
rather than its 1st. The token count still looked plausible in the sidecar,
which is why it went unnoticed -- the file was neither the right length nor the
right content, and nothing compared it against a one-shot encode.
"""
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(ROOT, "scripts")
for _p in (ROOT, SCRIPTS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import token_io
from token_io import encode_file_stream, memmap_tokens

ALPHABET = "abcdefghijklmnopqrstuvwxyz "


@pytest.fixture()
def tiny_corpus(tmp_path):
    """1 token per character, so expected ids are the source chars themselves."""
    vocab = {"0": "[UNK]"}
    for i, ch in enumerate(ALPHABET, start=1):
        vocab[str(i)] = ch
    vocab_path = tmp_path / "vocab.json"
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump({"vocab": vocab, "merges": [], "eos": "[UNK]"}, f)

    src_path = tmp_path / "src.txt"
    text = ("abcdefghij " * 400).strip()
    with open(src_path, "w", encoding="utf-8") as f:
        f.write(text)
    return str(src_path), str(vocab_path), text


def _tok(vocab_path):
    return token_io.BPETokenizer.load(vocab_path)


def test_resume_appends_instead_of_overwriting(tiny_corpus, tmp_path):
    """The whole point: resumed cache == one-shot cache, byte for byte."""
    src, vocab_path, text = tiny_corpus
    tok = _tok(vocab_path)
    expected = tok.encode(text)
    assert len(expected) == len(text)      # guard the fixture itself

    split = 2000

    # Pass 1 encodes a prefix, then we clear "complete" to imitate a crash
    # landing just before the final meta write.
    bin_path = str(tmp_path / "out.bin")
    encode_file_stream(src, vocab_path, bin_path, chunk_chars=500,
                       limit_chars=split, resume=False, tok=tok, log_every=0)

    meta_path = bin_path + ".meta.json"
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    meta["complete"] = False
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Pass 2 resumes and runs to EOF.
    result = encode_file_stream(src, vocab_path, bin_path, chunk_chars=500,
                                limit_chars=0, resume=True, tok=tok, log_every=0)

    assert result["complete"] is True
    assert result["chars"] == len(text)

    on_disk = memmap_tokens(bin_path).tolist()
    assert len(on_disk) == len(expected), (
        f"resumed cache holds {len(on_disk)} ids, expected {len(expected)}"
    )
    assert on_disk == expected, "resumed cache differs from a one-shot encode"


def test_resume_keeps_one_shot_control_equal(tiny_corpus, tmp_path):
    """Same source, no interruption: the control must not drift either."""
    src, vocab_path, text = tiny_corpus
    tok = _tok(vocab_path)

    bin_path = str(tmp_path / "oneshot.bin")
    encode_file_stream(src, vocab_path, bin_path, chunk_chars=500,
                       limit_chars=0, resume=False, tok=tok, log_every=0)
    on_disk = memmap_tokens(bin_path).tolist()
    assert on_disk == tok.encode(text)


def test_resume_without_sidecar_restarts_cleanly(tiny_corpus, tmp_path, capsys):
    """No usable sidecar: re-encode from zero rather than guess a char offset.

    The .bin alone tells us tokens, not chars, and chars-per-token is corpus
    dependent (4.047 on the Pile), so recovering by multiplying tokens by 4
    silently skips source text. Losing the sidecar must cost a re-encode, not
    a quietly truncated cache.
    """
    src, vocab_path, text = tiny_corpus
    tok = _tok(vocab_path)
    expected = tok.encode(text)

    bin_path = str(tmp_path / "out.bin")
    encode_file_stream(src, vocab_path, bin_path, chunk_chars=500,
                       limit_chars=2000, resume=False, tok=tok, log_every=0)
    os.remove(bin_path + ".meta.json")

    result = encode_file_stream(src, vocab_path, bin_path, chunk_chars=500,
                                limit_chars=0, resume=True, tok=tok, log_every=0)

    assert "no usable sidecar" in capsys.readouterr().out
    assert result["chars"] == len(text)
    assert memmap_tokens(bin_path).tolist() == expected


def test_resume_multiple_times_still_matches(tiny_corpus, tmp_path):
    """Three interruptions in a row -- the shape a long overnight run takes."""
    src, vocab_path, text = tiny_corpus
    tok = _tok(vocab_path)
    expected = tok.encode(text)

    bin_path = str(tmp_path / "out.bin")
    meta_path = bin_path + ".meta.json"

    for stop in (500, 1500, 3000):
        encode_file_stream(src, vocab_path, bin_path, chunk_chars=250,
                           limit_chars=stop, resume=False if stop == 500 else True,
                           tok=tok, log_every=0)
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        meta["complete"] = False
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    encode_file_stream(src, vocab_path, bin_path, chunk_chars=250,
                       limit_chars=0, resume=True, tok=tok, log_every=0)

    on_disk = memmap_tokens(bin_path).tolist()
    assert len(on_disk) == len(expected)
    assert on_disk == expected