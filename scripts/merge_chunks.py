#!/usr/bin/env python3
"""Merge N tokenized chunk .pt files into one.
Usage: python merge_chunks.py <chunk_dir> <output.pt>
Reads chunk_00.npy ... chunk_07.npy, concatenates, saves as torch int32 tensor.
"""
import os
import sys
import time
import numpy as np
import torch

def main():
    chunk_dir = sys.argv[1]
    dst = sys.argv[2]

    files = sorted(f for f in os.listdir(chunk_dir) if f.endswith(".npy"))
    print(f"Merging {len(files)} chunks from {chunk_dir}...", flush=True)

    t0 = time.time()
    sizes = []
    for f in files:
        arr = np.load(os.path.join(chunk_dir, f), mmap_mode="r")
        sizes.append(len(arr))
    total = sum(sizes)
    print(f"Total: {total:,} tokens", flush=True)

    out = np.empty(total, dtype=np.int32)
    offset = 0
    for f, sz in zip(files, sizes):
        arr = np.load(os.path.join(chunk_dir, f), mmap_mode="r")
        out[offset:offset + sz] = arr
        offset += sz
        print(f"  merged {f} ({sz:,})", flush=True)

    torch.save(torch.from_numpy(out), dst)
    dt = time.time() - t0
    print(f"Saved {total:,} tokens -> {dst} ({dt:.0f}s)", flush=True)

if __name__ == "__main__":
    main()
