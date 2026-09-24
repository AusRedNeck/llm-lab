"""The encoder must not be able to kill a long run, and must not silently drop data.

Two properties, both learned the hard way:
  1. The hardened `_encode_seg` produces byte-identical output to the pre-fix
     implementation (the sentinel change must not move a single token).
  2. If the encoder ever raises mid-run, the chunk is bisected and kept, the
     offending text is filed on disk — and if a piece cannot be encoded at all,
     the exception propagates. A crash is acceptable; missing data is not.

    python -m pytest tests/test_bpe_resilience.py -v
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

import token_io                                              # noqa: E402
from model.bpe import BPETokenizer, _SPLIT                    # noqa: E402

VOCAB = os.path.join(ROOT, "data", "bpe_owt16k.json")
SHARD = os.path.join(ROOT, "data", "openwebtext", "shards",
                     "train-00000-of-00080.txt")


def sample(chars: int = 200_000) -> str:
    with open(SHARD, encoding="utf-8", errors="replace") as f:
        return f.read(chars)


def pre_fix_encode_seg(tok: BPETokenizer, text: str) -> list[int]:
    """The implementation as it was before hardening, inlined for comparison."""
    ids: list[int] = []
    for chunk in _SPLIT.findall(text):
        parts = [bytes([b]) for b in chunk.encode("utf-8")]
        while len(parts) > 1:
            best, best_rank = None, None
            for i in range(len(parts) - 1):
                rank = tok.merges.get((parts[i], parts[i + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best, best_rank = i, rank
            if best is None:
                break
            parts[best:best + 2] = [parts[best] + parts[best + 1]]
        ids.extend(tok.token_to_id[p] for p in parts)
    return ids


def test_output_identical_to_pre_fix_implementation():
    tok = BPETokenizer.load(VOCAB)
    text = sample()
    assert tok.encode(text) == pre_fix_encode_seg(tok, text), \
        "hardening the sentinel must not change a single token"


class SizeLimited(BPETokenizer):
    """Raises on chunks over `limit` chars — models a size/state-dependent fault."""
    limit = 50_000

    def encode(self, text):
        if len(text) > self.limit:
            raise TypeError("slice indices must be integers or None or have an "
                            "__index__ method")
        return super().encode(text)


class AlwaysFails(BPETokenizer):
    def encode(self, text):
        raise TypeError("this tokenizer can never encode anything")


def test_resilient_bisect_keeps_every_byte():
    tok = SizeLimited.load(VOCAB)
    text = sample(150_000)
    failures: list[int] = []
    ids = token_io.encode_resilient(tok, text, lambda piece, exc: failures.append(len(piece)))

    assert failures, "the recovery path must actually fire on this input"
    assert tok.decode(ids) == text, "recovered tokens must decode back to the exact input"

    reference = BPETokenizer.load(VOCAB).encode(text)
    drift = abs(len(ids) - len(reference)) / len(reference)
    assert drift < 0.001, f"seam effects should be negligible, saw {drift:.4%}"


def test_resilient_raises_when_nothing_can_be_encoded():
    tok = AlwaysFails.load(VOCAB)
    with pytest.raises(TypeError):
        token_io.encode_resilient(tok, sample(10_000), lambda *a: None)


def test_error_log_files_the_evidence(tmp_path):
    log = token_io.EncodeErrorLog(str(tmp_path / "out.bin"))
    log("offending text goes here", TypeError("boom"))

    assert log.failures == 1
    assert os.path.exists(log.jsonl), "failures must leave an index behind"
    piece = os.path.join(log.piece_dir, "piece_001.txt")
    assert os.path.exists(piece), "the raw offending bytes must be kept"
    assert open(piece, encoding="utf-8").read() == "offending text goes here"


def test_encode_file_stream_reports_recovered_failures(tmp_path):
    """The pipeline must surface recoveries rather than hiding them."""
    tok = SizeLimited.load(VOCAB)
    src = tmp_path / "in.txt"
    src.write_text(sample(120_000), encoding="utf-8")
    out = str(tmp_path / "out.bin")
    res = token_io.encode_file_stream(str(src), VOCAB, out, chunk_chars=100_000,
                                      write_sidecar=False, log_every=0, tok=tok)
    assert res["encode_failures"] > 0
    assert os.path.exists(out + ".encode_errors.jsonl")
    assert res["tokens"] > 0
