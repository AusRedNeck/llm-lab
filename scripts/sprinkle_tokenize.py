#!/usr/bin/env python3
"""Parallel BPE tokenization — sprinkler pattern.
Each worker accumulates tokens in memory (numpy), writes one .npy at end.
Usage: python sprinkle_tokenize.py <src.txt> <vocab.json> <output.npy> [--workers 8]
"""
import argparse
import os
import sys
import time
import multiprocessing as mp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model.bpe import BPETokenizer

WORKERS_DEFAULT = 8
# Flush every N lines to keep RAM bounded (~100MB per flush)
FLUSH_LINES = 2_000_000


def count_lines(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        return sum(1 for _ in f)


def tokenize_range(args):
    """Tokenize a range of lines. Accumulates in numpy, writes one .npy."""
    src, vocab_path, out_npy, start_line, n_lines, worker_id = args
    tok = BPETokenizer.load(vocab_path)

    parts = []
    total = 0
    t0 = time.time()

    with open(src, encoding="utf-8", errors="replace") as f:
        for _ in range(start_line):
            f.readline()
        buf = []
        for i in range(n_lines):
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if line:
                ids = tok.encode(line)
                buf.extend(ids)
                total += len(ids)
            # Periodically flush buffer to numpy array
            if len(buf) >= FLUSH_LINES * 30:  # ~30 tokens/line avg
                parts.append(np.array(buf, dtype=np.int32))
                buf = []
                dt = time.time() - t0
                rate = total / dt if dt > 0 else 0
                # Print to stderr so it shows in parent log
                print(f"  w{worker_id}: {total:,} toks ({rate:,.0f} tok/s)", flush=True)
        if buf:
            parts.append(np.array(buf, dtype=np.int32))

    # Concatenate all parts
    if parts:
        out = np.concatenate(parts)
    else:
        out = np.array([], dtype=np.int32)
    np.save(out_npy, out)

    dt = time.time() - t0
    print(f"  w{worker_id} DONE: {len(out):,} toks in {dt:.0f}s ({len(out)/dt:,.0f} tok/s)", flush=True)
    return worker_id, len(out), dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("vocab")
    ap.add_argument("output")
    ap.add_argument("--workers", type=int, default=WORKERS_DEFAULT)
    args = ap.parse_args()

    print(f"Counting lines in {args.src}...", flush=True)
    total_lines = count_lines(args.src)
    print(f"{total_lines:,} lines", flush=True)

    per = total_lines // args.workers
    outdir = os.path.join(os.path.dirname(args.output), "owt4k_tmp")
    os.makedirs(outdir, exist_ok=True)

    tasks = []
    for i in range(args.workers):
        start = i * per
        n = per if i < args.workers - 1 else total_lines - start
        out_npy = os.path.join(outdir, f"chunk_{i:02d}.npy")
        tasks.append((args.src, args.vocab, out_npy, start, n, i))

    print(f"Launching {args.workers} parallel tokenizers...", flush=True)
    t0 = time.time()

    with mp.Pool(args.workers) as pool:
        results = pool.map(tokenize_range, tasks)

    # Merge results
    print("\nMerging...", flush=True)
    all_npy = sorted(results, key=lambda x: x[0])
    sizes = []
    for wid, count, dt in all_npy:
        npy_path = os.path.join(outdir, f"chunk_{wid:02d}.npy")
        sizes.append((npy_path, count))
        print(f"  worker {wid}: {count:,} toks in {dt:.0f}s", flush=True)

    total_tokens = sum(c for _, c in sizes)
    print(f"Total: {total_tokens:,} tokens", flush=True)

    out = np.empty(total_tokens, dtype=np.int32)
    off = 0
    for npy_path, sz in sizes:
        arr = np.load(npy_path, mmap_mode="r")
        out[off:off+sz] = arr
        off += sz

    import torch
    torch.save(torch.from_numpy(out), args.output)

    dt = time.time() - t0
    print(f"\nDone: {total_tokens:,} tokens in {dt:.0f}s ({total_tokens/dt:,.0f} tok/s) -> {args.output}", flush=True)
    print(f"Speedup vs single-thread (~550k tok/s): ~{total_tokens/dt/550000:.1f}x", flush=True)

    # Cleanup temp dir
    import shutil
    shutil.rmtree(outdir, ignore_errors=True)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
