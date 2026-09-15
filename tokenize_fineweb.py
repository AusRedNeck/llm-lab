#!/usr/bin/env python3
"""tokenize_fineweb.py — Chunked tokenization of FineWeb to numpy shards.

Writes tokens to disk in fixed-size numpy int16 shards instead of
accumulating a giant Python list in memory. Handles 10B+ tokens without
blowing RAM.

Usage:
    python tokenize_fineweb.py
"""
import numpy as np
import os
import sys
import time
from pathlib import Path

# Add model to path
sys.path.insert(0, str(Path(__file__).parent))
from model.bpe import BPETokenizer

SHARD_SIZE = 10_000_000  # 10M tokens per shard
INPUT = "data/fineweb_combined.txt"
OUTPUT_DIR = "data/fineweb_shards"
MANIFEST = "data/fineweb_shards/manifest.txt"


def main():
    tok = BPETokenizer.load("checkpoints/bpe8k.json")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    shard_idx = 0
    buf = []
    total_tokens = 0
    t0 = time.time()
    chunk_size = 500_000  # 500K chars per read (balance speed vs memory)

    manifest_lines = []

    with open(INPUT, "r", encoding="utf-8", errors="replace") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break

            # Tokenize this chunk
            ids = tok.encode(chunk)
            buf.extend(ids)

            # Flush completed shards
            while len(buf) >= SHARD_SIZE:
                shard = np.array(buf[:SHARD_SIZE], dtype=np.int16)
                path = os.path.join(OUTPUT_DIR, f"shard_{shard_idx:04d}.npy")
                np.save(path, shard)
                manifest_lines.append(f"shard_{shard_idx:04d}.npy {SHARD_SIZE}")
                shard_idx += 1
                buf = buf[SHARD_SIZE:]
                total_tokens += SHARD_SIZE

                elapsed = time.time() - t0
                rate = total_tokens / elapsed
                print(
                    f"  shard {shard_idx}: {total_tokens:,} tokens "
                    f"({rate:,.0f} tok/s, {elapsed:.0f}s elapsed)",
                    flush=True,
                )

    # Flush remaining
    if buf:
        shard = np.array(buf, dtype=np.int16)
        path = os.path.join(OUTPUT_DIR, f"shard_{shard_idx:04d}.npy")
        np.save(path, shard)
        manifest_lines.append(f"shard_{shard_idx:04d}.npy {len(buf)}")
        total_tokens += len(buf)

    # Write manifest
    with open(MANIFEST, "w") as f:
        f.write(f"total_tokens {total_tokens}\n")
        f.write(f"shard_size {SHARD_SIZE}\n")
        for line in manifest_lines:
            f.write(line + "\n")

    elapsed = time.time() - t0
    print(f"\nDone: {total_tokens:,} tokens in {elapsed:.0f}s")
    print(f"Shards: {shard_idx + 1} files in {OUTPUT_DIR}/")
    print(f"Manifest: {MANIFEST}")


if __name__ == "__main__":
    main()
