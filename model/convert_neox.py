#!/usr/bin/env python3
"""Convert GPT-NeoX HF tokenizer to our BPETokenizer format.

NeoX ships vocab.json (token_str -> id, 50277 entries incl. 25 added)
plus merges in tokenizer.json (50009 pairs, ByteLevel/GPT-2 style).
We decode the ByteLevel unicode back to raw bytes (same GPT-2 byte
mapping as model/train_bpe_hf.py) and write our JSON so train.py can
use --tokenizer data/incoming/bpe_neox.json directly.

Usage:
    uv run python -m model.convert_neox --out data/incoming/bpe_neox.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.bpe import BPETokenizer
from model.train_bpe_hf import _decode_bytelevel_token


def find_snapshot() -> str:
    """Locate the cached gpt-neox-20b snapshot dir."""
    base = os.path.expanduser("~/.cache/huggingface/hub/models--EleutherAI--gpt-neox-20b/snapshots")
    snaps = sorted(os.listdir(base))
    if not snaps:
        raise SystemExit("no cached gpt-neox-20b snapshot — run HF download first")
    return os.path.join(base, snaps[0])


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert NeoX tokenizer to our format")
    ap.add_argument("--snapshot", default=None, help="HF snapshot dir (default: auto-find cache)")
    ap.add_argument("--out", default="data/incoming/bpe_neox.json")
    args = ap.parse_args()

    snap = args.snapshot or find_snapshot()
    print(f"snapshot: {snap}")

    with open(os.path.join(snap, "vocab.json"), encoding="utf-8") as f:
        hf_vocab = json.load(f)  # {str_token: id}
    with open(os.path.join(snap, "tokenizer.json"), encoding="utf-8") as f:
        tj = json.load(f)
    hf_merges = tj["model"]["merges"]  # [[a_str, b_str], ...]

    print(f"  HF vocab: {len(hf_vocab)}, merges: {len(hf_merges)}")

    # id -> raw bytes, ids preserved exactly so HF-encoded .pt files match.
    our_vocab: dict[int, bytes] = {}
    for token_str, tok_id in hf_vocab.items():
        our_vocab[tok_id] = _decode_bytelevel_token(token_str)

    our_merges: dict[tuple[bytes, bytes], int] = {}
    for rank, m in enumerate(hf_merges):
        # tokenizer.json stores merges as single "left right" strings
        # (GPT-2 style — pieces never contain a literal space, Ġ covers it).
        if isinstance(m, str):
            a_str, b_str = m.split(" ", 1)
        else:
            a_str, b_str = m
        our_merges[(_decode_bytelevel_token(a_str),
                    _decode_bytelevel_token(b_str))] = rank
    print(f"  converted: vocab {len(our_vocab)}, merges {len(our_merges)}")

    # Added literals: HF non-special added tokens (NeoX multi-space runs
    # 50254-50276). Longest match wins at encode time, same as HF.
    added: dict[bytes, int] = {}
    for a in tj.get("added_tokens", []):
        if not a.get("special", False):
            content = a["content"]
            tid = a["id"]
            # Already in vocab via vocab.json — register as added literal.
            if tid in our_vocab:
                added[our_vocab[tid]] = tid
    print(f"  added literals: {len(added)} (multi-space runs)")

    tok = BPETokenizer(our_vocab, our_merges, eos=True, added=added)
    print(f"  eos_id={tok.eos_id} (expect 0 = <|endoftext|>)")

    # Byte coverage: every single byte must be encodable.
    missing = [i for i in range(256) if bytes([i]) not in tok.token_to_id]
    if missing:
        print(f"  WARNING: {len(missing)} missing base bytes: {missing[:10]}")
    else:
        print("  byte coverage: all 256 base bytes present")

    # Parity probe vs HF Rust encoder.
    from transformers import AutoTokenizer
    hf = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    for probe in ("Once upon a time there was a little princess.",
                  "Hello world, this is a test.  Double  spaces\n\nnewlines!"):
        our_ids = tok.encode(probe)
        hf_ids = hf.encode(probe, add_special_tokens=False)
        rt = tok.decode(our_ids)
        print(f"  probe {probe[:40]!r}: our={len(our_ids)} hf={len(hf_ids)} "
              f"match={our_ids == hf_ids} roundtrip={rt == probe}")

    tok.save(args.out)
    print(f"saved: {args.out} ({os.path.getsize(args.out) / 1e6:.1f}MB)")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"done in {time.time() - t0:.0f}s")
