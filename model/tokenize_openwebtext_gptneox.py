#!/usr/bin/env python3
"""Tokenize all OpenWebText shards with GPT-NeoX tokenizer → combined .bin.

Strategy:
  1. Load BPETokenizer from data/bpe_gptneox.json (already converted by Ronin)
  2. Read each shard sequentially, tokenize, append to data/openwebtext_combined_bpe_gptneox.bin
  3. Insert <|endoftext|> between files so documents don't fuse across shards
  4. Progress tracked in data/.gptneox_tok_progress.txt → resume after crash

Output:
  data/openwebtext_combined_bpe_gptneox.bin   (~64 GB, int32 memmap)
  data/.gptneox_tok_progress.txt              (resume state)

Usage:
    uv run python -m model.tokenize_openwebtext_gptneox
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.bpe import BPETokenizer


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Tokenize OWT shards with GPT-NeoX → single .bin memmap")
    ap.add_argument("--shards-dir", default="data/openwebtext/shards/",
                    help="Directory containing the .txt shard files")
    ap.add_argument("--tokenizer", default="data/bpe_gptneox.json",
                    help="BPETokenizer JSON (from convert_neox.py)")
    ap.add_argument("--out",
                    default="data/openwebtext_combined_bpe_gptneox.bin",
                    help="Output binary file (int32 memmap)")
    args = ap.parse_args()

    # ── Load tokenizer ────────────────────────────────────────────────
    print(f"Loading tokenizer: {args.tokenizer}")
    tok = BPETokenizer.load(args.tokenizer)
    print(f"  Vocab: {len(tok.vocab)} tokens, "
          f"Merges: {len(tok.merges)}, "
          f"EOS id: {tok.eos_id}")

    # ── Find shard files ──────────────────────────────────────────────
    shard_files = sorted(glob.glob(
        os.path.join(args.shards_dir, "**/*.txt"), recursive=True))
    if not shard_files:
        shard_files = sorted(glob.glob(
            os.path.join(args.shards_dir, "*.txt")))
    print(f"\nShards found: {len(shard_files)}")

    # ── Resume logic ──────────────────────────────────────────────────
    progress_file = args.out.replace(".bin", ".progress.txt")
    start_idx = 0
    if os.path.exists(progress_file):
        with open(progress_file) as f:
            start_idx = int(f.read().strip())
        print(f"Resuming from shard {start_idx}/{len(shard_files)}\n")

    # ── Open output binary (append mode) ──────────────────────────────
    total_tokens = 0
    total_chars = 0
    t_start = time.time()

    with open(args.out, "ab") as fout:
        current_pos = fout.tell()
        print(f"Appending to {args.out} (current offset: {current_pos})\n")

        for idx, shard_path in enumerate(shard_files[start_idx:],
                                         start=start_idx):
            fname = os.path.basename(shard_path)
            t0 = time.time()

            # Read & tokenize
            with open(shard_path, "r", encoding="utf-8") as fin:
                text = fin.read()
            n_chars = len(text.encode("utf-8"))
            total_chars += n_chars

            ids = tok.encode(text)
            if tok.eos_id is not None and ids:
                ids.append(tok.eos_id)       # end-of-document marker
            total_tokens += len(ids)

            # Write as int32 little-endian
            arr = np.array(ids, dtype=np.int32)
            arr.tofile(fout)
            fout.flush()

            elapsed = time.time() - t0
            tok_rate = len(ids) / elapsed
            mb_rate = arr.nbytes / elapsed / 1e6
            print(f"[{idx}/{len(shard_files)}] {fname}: "
                  f"{len(ids)/1e6:.2f}M toks  {elapsed:.1f}s  "
                  f"{tok_rate:,.0f} tok/s  {mb_rate:.1f} MB/s",
                  flush=True)

            # Update progress (one file at a time)
            with open(progress_file, "w") as pf:
                pf.write(str(idx + 1))

    wall = time.time() - t_start

    # ── Summary ───────────────────────────────────────────────────────
    n_bins = os.path.getsize(args.out)
    print(f"\n{'─' * 60}")
    print(f"COMPLETE: {args.out}")
    print(f"  Size:     {n_bins / 1e9:.2f} GB")
    print(f"  Tokens:   {total_tokens:,} ({total_tokens/1e9:.2f}B)")
    print(f"  Text:     {total_chars/1e9:.2f} GB (raw bytes)")
    print(f"  Avg bytes/token: {total_chars / total_tokens:.3f}")
    print(f"  Wall:     {wall/3600:.2f}h ({wall/60:.1f}m)")
    print(f"  Rate:     {total_tokens/wall:,.0f} tok/s")
    print(f"{'─' * 60}")

    # Remove progress marker (encoding succeeded)
    if os.path.exists(progress_file):
        os.remove(progress_file)
        print(f"Progress file cleared.")


if __name__ == "__main__":
    main()
