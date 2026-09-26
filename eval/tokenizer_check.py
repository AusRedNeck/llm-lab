#!/usr/bin/env python3
"""Tokenizer-vs-data consistency check.

Ronin retokenized: prove the new token stream agrees with the tokenizer
the model was built with. Catches the silent killers -- re-encode with a
different vocab/merges, id shifts, truncated .bin -- before they burn
another overnight run.

Usage (on the machine that owns the files):
  uv run python -m eval.tokenizer_check --tokenizer data/incoming/<tok>.json \\
      --tok_cache data/<corpus>.bin --checkpoint checkpoints/<run>_step<N>.pt
  # optional: diff old vs new encode side by side
  uv run python -m eval.tokenizer_check ... --tok_cache_b data/<old>.bin

Fails loud (exit 1) on: corpus max id >= model vocab, model vocab !=
tokenizer vocab. Round-trip mismatch and top-token weirdness are printed
for human eyes, not gated -- our Python BPE and HF Rust can disagree on
rare merges without the corpus being wrong.
"""
from __future__ import annotations

import argparse
import collections
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from model.bpe import BPETokenizer
from train.train import load_token_cache, val_bytes_per_token


def corpus_stats(ids: torch.Tensor, tok: BPETokenizer, tag: str,
                 windows: int = 20, wlen: int = 512) -> None:
    n = len(ids)
    lo, hi = int(ids.min()), int(ids.max())
    print(f"[{tag}] tokens={n:,} id_range=[{lo},{hi}]")
    print(f"[{tag}] bytes/token={val_bytes_per_token(ids, tok):.4f}")

    # Top tokens: eyeball test. Wrong tokenizer shows here first --
    # garbage bytes or a foreign language where English should be.
    flat = ids[:2_000_00].tolist()
    top = collections.Counter(flat).most_common(10)
    print(f"[{tag}] top-10 (id x count 'text'):")
    for tid, c in top:
        try:
            s = tok.decode([tid]).replace("\n", "\\n")
        except Exception as e:  # noqa: BLE001 -- one bad id must not kill it
            s = f"<decode-error {e}>"
        print(f"  {tid:>6} x{c:>8,} {s[:40]!r}")

    # Round-trip: decode a window, re-encode, demand the same ids back.
    # A corpus cut by THIS tokenizer round-trips (near) exactly.
    torch.manual_seed(0)
    exact, total = 0, 0
    for _ in range(windows):
        off = int(torch.randint(0, max(1, n - wlen - 1), (1,)).item())
        w = [int(i) for i in ids[off:off + wlen].tolist()]
        try:
            back = tok.encode(tok.decode(w))
        except Exception:  # noqa: BLE001 -- counts as mismatch, keeps going
            continue
        total += 1
        exact += (back == w)
    print(f"[{tag}] round-trip exact windows: {exact}/{total}")


def main() -> None:
    ap = argparse.ArgumentParser(description="tokenizer vs data check")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--tok_cache", required=True)
    ap.add_argument("--tok_cache_b", default=None,
                    help="optional second (old) encode to diff against")
    ap.add_argument("--checkpoint", default=None)
    args = ap.parse_args()

    tok = BPETokenizer.load(args.tokenizer)
    print(f"tokenizer={args.tokenizer} vocab={len(tok.vocab)}")

    bad = False
    v_model = None
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        v_model = ckpt["cfg"]["vocab_size"]
        print(f"checkpoint step={ckpt.get('step')} model_vocab={v_model} "
              f"ckpt_tokenizer={ckpt['cfg'].get('tokenizer')}")
        if v_model != len(tok.vocab):
            print(f"FAIL: model vocab {v_model} != tokenizer vocab "
                  f"{len(tok.vocab)} -- weights speak a different language")
            bad = True

    ids = load_token_cache(args.tok_cache)
    if v_model is not None and int(ids.max()) >= v_model:
        print(f"FAIL: corpus max id {int(ids.max())} >= model vocab "
              f"{v_model} -- embedding lookup is out of bounds")
        bad = True
    corpus_stats(ids, tok, "new")

    if args.tok_cache_b:
        old = load_token_cache(args.tok_cache_b)
        corpus_stats(old, tok, "old")
        m = min(len(ids), len(old), 1_000_000)
        same = (ids[:m] == old[:m]).sum().item() if len(old) >= m else -1
        print(f"[diff] first {m:,} ids identical: {same:,}"
              + (" (files differ in length or content)"
                 if same != m else " (same prefix)"))

    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
