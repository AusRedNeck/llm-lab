#!/usr/bin/env python3
"""Tokenize OpenWebText shards with GPT-NeoX tokenizer → combined .bin memmap.

Streaming version: reads line-by-line, encodes in chunks, writes directly to
output file. Never loads >50MB into memory at once.

Progress: data/.gptneox_tok_progress.txt tracks last completed shard → resume after crash.

Output:
    data/openwebtext_combined_bpe_gptneox.bin   (~64 GB, int32 memmap)

Usage:
    uv run python -m model.tokenize_openwebtext_gptneox_stream
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stream-tokenize OWT shards with Pythia/GPT-NeoX tokenizer")
    ap.add_argument("--shards-dir", default="data/openwebtext/shards/",
                    help="Directory containing the .txt shard files")
    ap.add_argument("--model", default="EleutherAI/pythia-70m-deduped",
                    help="HF model repo with tokenizer (default: pythia-70m-deduped)")
    ap.add_argument("--out",
                    default="data/openwebtext_combined_bpe_gptneox.bin",
                    help="Output binary file (int32 memmap)")
    ap.add_argument("--batch-size", type=int, default=200,
                    help="Encode N lines together before flushing (default 200)")
    args = ap.parse_args()

    # ── Load tokenizer ────────────────────────────────────────────────
    print(f"Loading tokenizer: {args.model}")
    t_load = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    print(f"  Vocab: {tok.vocab_size}, loaded in {time.time()-t_load:.1f}s\n")

    # ── Find shard files ──────────────────────────────────────────────
    shard_files = sorted(glob.glob(
        os.path.join(args.shards_dir, "**/*.txt"), recursive=True))
    if not shard_files:
        shard_files = sorted(glob.glob(
            os.path.join(args.shards_dir, "*.txt")))
    print(f"Shards found: {len(shard_files)}\n")

    # ── Resume logic ──────────────────────────────────────────────────
    progress_file = args.out.replace(".bin", ".progress.txt")
    start_idx = 0
    if os.path.exists(progress_file):
        with open(progress_file) as f:
            start_idx = int(f.read().strip())
        print(f"Resuming from shard {start_idx}/{len(shard_files)}\n")

    # ── Open output binary (append mode) ──────────────────────────────
    total_tokens = 0
    total_lines = 0
    t_start = time.time()
    shard_t0 = time.time()

    with open(args.out, "ab") as fout:
        current_offset = fout.tell()
        print(f"Appending to {args.out} (current offset: {current_offset:,} bytes)\n")

        for idx, shard_path in enumerate(shard_files[start_idx:],
                                         start=start_idx):
            fname = os.path.basename(shard_path)
            shard_t0 = time.time()

            lines_buffer: list[str] = []
            shard_tokens = 0

            with open(shard_path, "r", encoding="utf-8") as fin:
                for line in fin:
                    lines_buffer.append(line)
                    total_lines += 1

                    if len(lines_buffer) >= args.batch_size:
                        # Encode batch, write to file
                        joined = "".join(lines_buffer)
                        ids = tok.encode(joined, add_special_tokens=False)
                        shard_tokens += len(ids)

                        if ids:
                            arr = np.array(ids, dtype=np.int32)
                            arr.tofile(fout)

                        lines_buffer = []

                # Flush remaining lines
                if lines_buffer:
                    joined = "".join(lines_buffer)
                    ids = tok.encode(joined, add_special_tokens=False)
                    shard_tokens += len(ids)
                    if ids:
                        arr = np.array(ids, dtype=np.int32)
                        arr.tofile(fout)
                    lines_buffer = []

            # Per-shard EOS token
            eos_id = tok.eos_token_id if tok.eos_token_id else 0
            np.array([eos_id], dtype=np.int32).tofile(fout)
            shard_tokens += 1
            total_tokens += shard_tokens

            elapsed = time.time() - shard_t0
            tok_rate = shard_tokens / elapsed
            mb_written = fout.tell() - current_offset  # approximate
            print(f"[{idx}/{len(shard_files)}] {fname}: "
                  f"{shard_tokens/1e6:.2f}M toks  {elapsed:.1f}s  "
                  f"{tok_rate:,.0f} tok/s",
                  flush=True)

            # Update progress
            with open(progress_file, "w") as pf:
                pf.write(str(idx + 1))

    wall = time.time() - t_start
    n_bins = os.path.getsize(args.out)

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"COMPLETE: {args.out}")
    print(f"  Size:         {n_bins / 1e9:.2f} GB")
    print(f"  Tokens:       {total_tokens:,} ({total_tokens/1e9:.2f}B)")
    print(f"  Lines:        {total_lines:,}")
    print(f"  Wall:         {wall/3600:.2f}h ({wall/60:.1f}m)")
    print(f"  Rate:         {total_tokens/wall:,.0f} tok/s")
    print(f"{'=' * 60}")

    # Clear progress marker (encoding succeeded)
    if os.path.exists(progress_file):
        os.remove(progress_file)
        print("Progress file cleared.")
