#!/usr/bin/env python3
"""Train BPE via HF tokenizers (fast), export to our BPETokenizer format.

Strategy:
1. Train HF tokenizer (Rust-backed, seconds)
2. Save to disk, read back vocab + merges
3. Convert ByteLevel tokens → raw bytes using GPT-2 byte mapping
4. Write our BPETokenizer JSON

Usage:
    uv run python -m model.train_bpe_hf --data data/incoming/openwebtext_sample_1gb.txt \
        --lines 100000 --merges 4000 --out data/incoming/bpe_owt4k.json
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.bpe import BPETokenizer, EOS


# ── GPT-2 byte-to-unicode mapping (the reverse of HF ByteLevel) ─────
# GPT-2 maps bytes 0-255 to unicode chars. We need the reverse mapping.
# The mapping is: printable ASCII (33-126) stay, space(32)->Ġ(288),
# rest map to 289-510. We build the reverse lookup.
def _build_byte_decoder() -> dict[int, int]:
    """Build reverse mapping: unicode_codepoint -> original_byte."""
    # GPT-2 byte encoding table
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    # bs[i] = original byte, cs[i] = unicode codepoint
    # Reverse: cs[i] -> bs[i]
    return {c: b for b, c in zip(bs, cs)}


BYTE_DECODER = _build_byte_decoder()


def _decode_bytelevel_token(token_str: str) -> bytes:
    """Convert HF ByteLevel token string to raw bytes."""
    result = bytearray()
    for ch in token_str:
        cp = ord(ch)
        if cp in BYTE_DECODER:
            result.append(BYTE_DECODER[cp])
        else:
            # Unknown char — encode as UTF-8 (shouldn't happen for BPE base tokens)
            result.extend(ch.encode("utf-8"))
    return bytes(result)


def _load_sample(path: str, max_lines: int, stride: int) -> list[str]:
    import glob
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "**", "*.txt"), recursive=True))
        if not files:
            raise FileNotFoundError(f"no .txt files under {path}")
        texts: list[str] = []
        per_file_stride = max(1, stride // max(1, len(files) // 10))
        for fp in files:
            if len(texts) >= max_lines:
                break
            with open(fp, encoding="utf-8", errors="replace") as f:
                for i, line in enumerate(f):
                    if len(texts) >= max_lines:
                        break
                    if i % per_file_stride == 0:
                        line = line.strip()
                        if line:
                            texts.append(line)
        return texts
    texts = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if i % stride == 0:
                line = line.strip()
                if line:
                    texts.append(line)
                    if len(texts) >= max_lines:
                        break
    return texts


def train_and_export(texts: list[str], num_merges: int, out_path: str) -> BPETokenizer:
    """Train via HF, export to our format."""
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import ByteLevel

    t0 = time.time()

    # Build HF tokenizer: byte-level BPE
    tok = Tokenizer(BPE(unk_token=None))
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)

    trainer = BpeTrainer(
        vocab_size=256 + num_merges,
        min_frequency=2,
        special_tokens=["<|end_of_text|>"],
        show_progress=True,
    )

    print(f"  training {num_merges} merges on {len(texts)} lines (HF/Rust)...", flush=True)
    tok.train_from_iterator(texts, trainer=trainer)

    hf_time = time.time() - t0
    print(f"  HF training done in {hf_time:.1f}s", flush=True)

    # Save to temp dir to read back merges
    tmpdir = tempfile.mkdtemp()
    tok_path = os.path.join(tmpdir, "tokenizer.json")
    tok.save(tok_path)

    with open(tok_path) as f:
        hf_data = json.load(f)

    hf_vocab = hf_data["model"]["vocab"]  # {str_token: int_id}
    hf_merges = hf_data["model"]["merges"]  # [[str_a, str_b], ...]

    print(f"  HF vocab: {len(hf_vocab)}, merges: {len(hf_merges)}", flush=True)

    # ── Convert to our format ──
    # Vocab: int_id -> bytes
    our_vocab: dict[int, bytes] = {}
    for token_str, token_id in hf_vocab.items():
        if token_str == "<|end_of_text|>":
            our_vocab[token_id] = EOS.encode("utf-8")
        else:
            our_vocab[token_id] = _decode_bytelevel_token(token_str)

    # Merges: (bytes_a, bytes_b) -> rank
    our_merges: dict[tuple[bytes, bytes], int] = {}
    for rank, (a_str, b_str) in enumerate(hf_merges):
        a_bytes = _decode_bytelevel_token(a_str)
        b_bytes = _decode_bytelevel_token(b_str)
        our_merges[(a_bytes, b_bytes)] = rank

    print(f"  converted: vocab {len(our_vocab)}, merges {len(our_merges)}", flush=True)

    # ── Sanity checks ──
    # Build our tokenizer
    our_tok = BPETokenizer(our_vocab, our_merges, eos=False)

    # Test encoding
    probe = "Once upon a time there was a little princess."
    our_ids = our_tok.encode(probe)
    hf_ids = tok.encode(probe).ids
    print(f"  probe: our={len(our_ids)} tokens, HF={len(hf_ids)} tokens, "
          f"bytes={len(probe.encode())}", flush=True)

    # Roundtrip test
    decoded = our_tok.decode(our_ids)
    if decoded == probe:
        print(f"  roundtrip: OK", flush=True)
    else:
        print(f"  roundtrip: MISMATCH!", flush=True)
        print(f"    expected: {probe!r}", flush=True)
        print(f"    got:      {decoded!r}", flush=True)

    # Byte coverage: check that all 256 base bytes are in vocab
    missing = [i for i in range(256) if i not in our_vocab.values()]
    if missing:
        print(f"  WARNING: {len(missing)} missing base bytes", flush=True)
    else:
        print(f"  byte coverage: all 256 base bytes present", flush=True)

    # Cleanup
    os.unlink(tok_path)
    os.rmdir(tmpdir)

    return our_tok


def main():
    ap = argparse.ArgumentParser(description="Train BPE via HF tokenizers")
    ap.add_argument("--data", default="data/incoming/openwebtext_sample_1gb.txt")
    ap.add_argument("--lines", type=int, default=100000)
    ap.add_argument("--stride", type=int, default=88)
    ap.add_argument("--merges", type=int, default=4000)
    ap.add_argument("--out", default="data/incoming/bpe_owt4k.json")
    args = ap.parse_args()

    t0 = time.time()
    texts = _load_sample(args.data, args.lines, args.stride)
    mb = sum(len(t) for t in texts) / 1e6
    print(f"sample: {len(texts)} lines, {mb:.1f}MB ({time.time() - t0:.0f}s)")

    t0 = time.time()
    tok = train_and_export(texts, args.merges, args.out)
    total = time.time() - t0
    print(f"total: vocab {len(tok.vocab)} in {total:.0f}s")

    tok.save(args.out)
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
