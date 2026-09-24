#!/usr/bin/env python3
"""Tokenize OpenWebText shards with Pythia (GPT-NeoX) tokenizer → combined .bin memmap.

Streaming design: reads each shard line-by-line, buffers paragraphs, flushes
to the tokenizer every ~200 lines. Never holds >50MB of text in memory.

Uses HuggingFace FastTokenizer (Rust backend) — far faster than our Python
BPETokenizer for corpus-scale work. EOS token inserted between shards.

Progress tracking: data/.gptneox_tok_progress.txt records completed shard index
→ resume after crash.

Outputs:
    data/openwebtext_combined_bpe_pythia.bin     (~64 GB, int32 memmap)
    data/openwebtext_combined_bpe_pythia.summary   (stats, for verification)

Usage:
    uv run python -m model.tokenize_openwebtext_pythia
    uv run python -m model.tokenize_openwebtext_pythia --resume     # skip finished shards
    uv run python -m model.tokenize_openwebtext_pythia --batch-size 500  # bigger flushes
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
                    help="HF model repo (tokenizer only)")
    ap.add_argument("--out",
                    default="data/openwebtext_combined_bpe_pythia.bin",
                    help="Output binary file (int32 memmap)")
    ap.add_argument("--batch-size", type=int, default=200,
                    help="Flush every N lines (higher = faster throughput)")
    ap.add_argument("--resume", action="store_true", default=False,
                    help="Skip already-finished shards (default: true)")
    args = ap.parse_args()

    # ── Load tokenizer (FastTokenizer = Rust backend) ────────────────
    print(f"Loading FastTokenizer: {args.model}")
    t_load = time.time()
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    elapsed = time.time() - t_load
    print(f"  Vocab: {tok.vocab_size}, loaded in {elapsed:.1f}s\n")

    # Verify eos id
    eos_id = 0
    if hasattr(tok, 'eos_token_id') and tok.eos_token_id is not None:
        eos_id = tok.eos_token_id
    elif hasattr(tok, 'eos_token') and tok.eos_token is not None:
        eos_id = tok.convert_tokens_to_ids(tok.eos_token)
    else:
        # GPT-NeoX uses <|endoftext|>
        try:
            eos_id = tok.convert_tokens_to_ids("<|endoftext|>")
        except Exception:
            eos_id = 0
    print(f"  EOS token id: {eos_id}\n")

    # ── Find shard files ──────────────────────────────────────────────
    shard_files = sorted(glob.glob(
        os.path.join(args.shards_dir, "**/*.txt"), recursive=True))
    if not shard_files:
        shard_files = sorted(glob.glob(
            os.path.join(args.shards_dir, "*.txt")))
    total_shards = len(shard_files)
    print(f"Shards found: {total_shards}\n")

    # ── Resume logic ──────────────────────────────────────────────────
    progress_file = args.out.replace(".bin", ".progress.txt")
    start_idx = 0
    if os.path.exists(progress_file) and args.resume:
        with open(progress_file) as f:
            start_idx = int(f.read().strip())
        print(f"Resume: starting shard {start_idx}/{total_shards}\n")
    elif not args.resume:
        # Clear any stale progress
        if os.path.exists(progress_file):
            os.remove(progress_file)

    # ── Stream through shards ─────────────────────────────────────────
    total_tokens = 0
    total_chars = 0
    t_start = time.time()

    with open(args.out, "ab") as fout:
        current_offset = fout.tell()
        print(f"Appending to {args.out} (current offset: {current_offset:,} bytes)\n")

        for idx, shard_path in enumerate(shard_files[start_idx:],
                                         start=start_idx):
            fname = os.path.basename(shard_path)
            shard_t0 = time.time()
            shard_chars = 0
            shard_tokens = 0

            # Buffer lines before flushing to tokenizer
            lines_buffer = []
            buf_size_estimate = 0
            MAX_BUF_SIZE = 50 * 1024 * 1024  # 50MB max buffer

            with open(shard_path, "r", encoding="utf-8") as fin:
                for line in fin:
                    line_len = len(line.encode("utf-8"))
                    lines_buffer.append(line)
                    buf_size_estimate += line_len
                    total_chars += line_len

                    # Flush when buffer hits size limit OR batch-size reached
                    if (buf_size_estimate >= MAX_BUF_SIZE or
                            len(lines_buffer) >= args.batch_size):
                        joined = "".join(lines_buffer)
                        # Fast tokenizer handles long inputs natively, but let's
                        # keep it reasonable — chunk further if needed
                        ids = tok(joined, add_special_tokens=False)
                        shard_tokens += len(ids["input_ids"])

                        if ids["input_ids"]:
                            arr = np.array(ids["input_ids"], dtype=np.int32)
                            arr.tofile(fout)

                        lines_buffer = []
                        buf_size_estimate = 0

                # Final flush for remaining lines
                if lines_buffer:
                    joined = "".join(lines_buffer)
                    ids = tok(joined, add_special_tokens=False)
                    shard_tokens += len(ids["input_ids"])
                    if ids["input_ids"]:
                        arr = np.array(ids["input_ids"], dtype=np.int32)
                        arr.tofile(fout)

            # Insert EOS token between documents (never after last shard)
            is_last = (idx == total_shards - 1)
            if not is_last:
                np.array([eos_id], dtype=np.int32).tofile(fout)
                shard_tokens += 1

            total_tokens += shard_tokens
            elapsed = time.time() - shard_t0
            tokens_per_mb = shard_chars / elapsed / 1e6

            print(f"[{idx+1}/{total_shards}] {fname}: "
                  f"{shard_chars/1e6:.1f}MB → {shard_tokens/1e6:.2f}M toks  "
                  f"{elapsed:.1f}s  {tokens_per_mb:,.0f} tok/s/MB",
                  flush=True)

            # Progress checkpoint (one shard at a time)
            with open(progress_file, "w") as pf:
                pf.write(str(idx + 1))

    wall = time.time() - t_start
    n_bins = os.path.getsize(args.out)

    # ── Summary ───────────────────────────────────────────────────────
    summary = (
        f"Pythia/NeoX Tokenization Summary\n"
        f"{'=' * 50}\n"
        f"Input:      {args.shards_dir}\n"
        f"Model:      {args.model}\n"
        f"Vocab:      {tok.vocab_size} tokens\n"
        f"Shards:     {total_shards}\n"
        f"Output:     {args.out}\n"
        f"Size:       {n_bins / 1e9:.2f} GB ({n_bins:,} bytes)\n"
        f"Tokens:     {total_tokens:,} ({total_tokens/1e9:.2f}B)\n"
        f"Text:       {total_chars:,} bytes ({total_chars/1e9:.2f}GB raw)\n"
        f"bytes/token:{total_chars/total_tokens:.3f}\n"
        f"Wall:       {wall/3600:.2f}h ({wall/60:.1f}min)\n"
        f"Rate:       {total_tokens/wall:,.0f} tok/s\n"
        f"EOS tokens: added between each shard pair\n"
    )

    print(summary)
    with open(args.out.replace(".bin", ".summary"), "w") as sf:
        sf.write(summary)

    # Clear progress marker (encoding succeeded)
    if os.path.exists(progress_file):
        os.remove(progress_file)
        print("Progress cleared.")


if __name__ == "__main__":
    main()
