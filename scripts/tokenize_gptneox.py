#!/usr/bin/env python3
"""Tokenize OWT shards with the HF GPT-NeoX tokenizer (Rust-fast).

Full:  all 80 shards -> data/openwebtext_combined_bpe_gptneox.bin (flat int32).
Smoke: --only N -> shard N -> --pt (torch int32, train.py-ready).
Resume: --resume skips shards listed in data/.gptneox_tok_progress.txt.
"""
import argparse
import gc
import glob
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

SRC = "EleutherAI/pythia-70m-deduped"
SHARD_DIR = "data/openwebtext/shards"
OUT_BIN = "data/openwebtext_combined_bpe_gptneox.bin"
PROGRESS = "data/.gptneox_tok_progress.txt"
PIECE_CHARS = 50_000_000  # ~50MB text per piece, flat RAM
CHUNK_CHARS = 10_000_000  # Rust batch unit: fast path (3.9M tok/s).
# Boundary note (measured): chunked encode_batch drifts ~2 tokens per
# 200KB vs whole-text encode — ~50 tokens in 113M (~4e-7) per shard.
# Pure noise for training. Speed wins.


def load_ht():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(SRC)  # cached after first pull
    return tok._tokenizer  # Rust core: encode_batch lives here


def encode_text(ht, text):
    # Piece-wise so a 500MB shard never balloons RAM; 10MB chunks
    # into Rust encode_batch (the 3.9M tok/s path).
    ids: list[int] = []
    for s in range(0, len(text), PIECE_CHARS):
        piece = text[s:s + PIECE_CHARS]
        chunks = [piece[i:i + CHUNK_CHARS]
                  for i in range(0, len(piece), CHUNK_CHARS)]
        for enc in ht.encode_batch(chunks, add_special_tokens=False):
            ids.extend(enc.ids)
        del chunks
    return ids


def done_set():
    if os.path.exists(PROGRESS):
        with open(PROGRESS) as f:
            return {ln.strip() for ln in f if ln.strip()}
    return set()


def mark_done(name):
    with open(PROGRESS, "a") as f:
        f.write(name + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=int, default=-1,
                    help="single shard index for smoke (default: all)")
    ap.add_argument("--pt", default="data/shard00_bpe_gptneox.pt",
                    help="torch cache out (only with --only)")
    ap.add_argument("--resume", action="store_true",
                    help="skip shards already in the progress file")
    args = ap.parse_args()

    shards = sorted(glob.glob(os.path.join(SHARD_DIR, "*.txt")))
    assert shards, f"no shards under {SHARD_DIR}"
    ht = load_ht()

    if args.only >= 0:
        # Smoke path: one shard straight to a torch .pt train.py can load.
        import torch
        path = shards[args.only]
        t0 = time.time()
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()
        mb = len(text) / 1e6
        ids = encode_text(ht, text)
        t = torch.tensor(ids, dtype=torch.int32)
        del ids, text
        torch.save(t, args.pt)
        dt = time.time() - t0
        print(f"  {os.path.basename(path)}: {mb:.0f}MB -> "
              f"{len(t) / 1e6:.1f}M toks ({len(t) / mb:.2f} tok/KB, "
              f"{len(t) / dt:,.0f} tok/s) -> {args.pt}", flush=True)
        return

    # Full path: all shards appended to one flat int32 .bin.
    if args.resume and os.path.exists(OUT_BIN) and os.path.exists(PROGRESS):
        done = done_set()  # pick up where we left off
    else:
        if os.path.exists(OUT_BIN):
            os.remove(OUT_BIN)  # stale partial: start clean
        open(OUT_BIN, "wb").close()
        if os.path.exists(PROGRESS):
            os.remove(PROGRESS)
        done = set()
    run_total, t0 = 0, time.time()
    with open(OUT_BIN, "ab") as f:
        for path in shards:
            name = os.path.basename(path)
            if name in done:
                continue
            with open(path, encoding="utf-8", errors="replace") as fh:
                text = fh.read()
            mb = len(text) / 1e6
            ids = encode_text(ht, text)
            arr = np.asarray(ids, dtype=np.int32)
            del ids, text
            f.write(arr.tobytes())
            f.flush()
            n = len(arr)
            del arr
            gc.collect()
            run_total += n
            dt = time.time() - t0
            print(f"  {name}: {n / 1e6:.1f}M toks "
                  f"({n / dt:,.0f} tok/s, run total {run_total / 1e6:.1f}M)",
                  flush=True)
            mark_done(name)
    dt = time.time() - t0
    full = os.path.getsize(OUT_BIN) // 4
    print(f"done: run {run_total / 1e6:.1f}M, file {full / 1e6:.1f}M tokens "
          f"-> {OUT_BIN} ({dt / 60:.0f}min)", flush=True)


if __name__ == "__main__":
    main()
