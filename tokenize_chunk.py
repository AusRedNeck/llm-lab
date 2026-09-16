#!/usr/bin/env python3
"""tokenize_chunk.py — Tokenize a single chunk file to numpy shards.

Usage: python tokenize_chunk.py <chunk_file> <output_dir> [--resume] [--tokenizer PATH]
Each chunk writes to its own subdirectory to avoid index collisions.
Tokenizer defaults to the fresh OWT vocab (Gen1+Gen2); point --tokenizer
anywhere else for older vocabs.
"""
import numpy as np
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from model.bpe import BPETokenizer

SHARD_SIZE = 10_000_000

def count_existing_shards(output_dir):
    """Count existing shard files and total tokens already written."""
    existing = [f for f in os.listdir(output_dir) if f.startswith("shard_") and f.endswith(".npy")]
    if not existing:
        return 0, 0
    # Count full shards (10M tokens each)
    full = 0
    partial = 0
    for f in existing:
        arr = np.load(os.path.join(output_dir, f))
        if len(arr) >= SHARD_SIZE:
            full += 1
        else:
            partial += len(arr)
    return full, partial

def main():
    if len(sys.argv) < 3:
        print("Usage: python tokenize_chunk.py <chunk_file> <output_dir> [--resume] [--tokenizer PATH]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_dir = sys.argv[2]
    resume = "--resume" in sys.argv
    os.makedirs(output_dir, exist_ok=True)

    # Tokenizer: CLI wins, fresh OWT vocab is the default (old
    # checkpoints/bpe8k.json path carried pre-Gen1 flattened whitespace).
    tok_path = "data/incoming/bpe_owt8k.json"
    if "--tokenizer" in sys.argv:
        tok_path = sys.argv[sys.argv.index("--tokenizer") + 1]
    print(f"  tokenizer={tok_path}")
    tok = BPETokenizer.load(tok_path)
    
    # Resume: count existing shards, skip them
    skip_tokens = 0
    shard_idx = 0
    if resume:
        full_shards, partial_tokens = count_existing_shards(output_dir)
        shard_idx = full_shards
        skip_tokens = full_shards * SHARD_SIZE + partial_tokens
        if skip_tokens > 0:
            print(f"  [{os.path.basename(input_file)}] Resuming: skipping {skip_tokens:,} tokens ({full_shards} full shards + {partial_tokens} partial)", flush=True)
    
    buf = []
    total_tokens = 0
    t0 = time.time()
    chunk_size = 500_000
    chars_read = 0
    
    with open(input_file, "r", encoding="utf-8", errors="replace") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            
            chars_read += len(chunk)
            ids = tok.encode(chunk)
            buf.extend(ids)
            
            # Skip tokens we've already processed on resume
            if skip_tokens > 0:
                skip_now = min(skip_tokens, len(buf))
                buf = buf[skip_now:]
                skip_tokens -= skip_now
                continue
            
            while len(buf) >= SHARD_SIZE:
                shard = np.array(buf[:SHARD_SIZE], dtype=np.int16)
                path = os.path.join(output_dir, f"shard_{shard_idx:04d}.npy")
                np.save(path, shard)
                shard_idx += 1
                buf = buf[SHARD_SIZE:]
                total_tokens += SHARD_SIZE
                
                elapsed = time.time() - t0
                rate = total_tokens / elapsed
                print(f"  [{os.path.basename(input_file)}] shard {shard_idx}: {total_tokens:,} tokens ({rate:,.0f} tok/s)", flush=True)
    
    if buf and not skip_tokens:
        shard = np.array(buf, dtype=np.int16)
        path = os.path.join(output_dir, f"shard_{shard_idx:04d}.npy")
        np.save(path, shard)
        total_tokens += len(buf)
    
    elapsed = time.time() - t0
    print(f"  [{os.path.basename(input_file)}] Done: {total_tokens:,} new tokens in {elapsed:.0f}s, {shard_idx} total shards")

if __name__ == "__main__":
    main()
