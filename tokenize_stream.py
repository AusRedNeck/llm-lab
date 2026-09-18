#!/usr/bin/env python3
"""Stream-tokenize a large text file with BPE, writing chunks to disk.

Avoids RAM blowup: encodes in 500K-char chunks, saves each as a temp
numpy file, then concatenates at the end via memory-mapped reads.

Usage:
    python -u tokenize_stream.py <src.txt> <vocab.json> <dst.pt>
"""
import os
import sys
import time
import tempfile
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model.bpe import BPETokenizer

CHUNK_CHARS = 500_000  # 500K chars per encode call


def main():
    if len(sys.argv) != 4:
        print("Usage: python -u tokenize_stream.py <src.txt> <vocab.json> <dst.pt>")
        sys.exit(1)

    src, vocab_path, dst = sys.argv[1], sys.argv[2], sys.argv[3]
    tok = BPETokenizer.load(vocab_path)
    print(f"Vocab: {len(tok.vocab)}", flush=True)

    tmpdir = tempfile.mkdtemp(prefix="tok_")
    chunk_files = []
    total = 0
    t0 = time.time()

    with open(src, encoding="utf-8", errors="replace") as f:
        while True:
            text = f.read(CHUNK_CHARS)
            if not text:
                break
            ids = tok.encode(text)
            # Save chunk to disk immediately — never accumulate in RAM
            cf = os.path.join(tmpdir, f"chunk_{len(chunk_files):04d}.npy")
            np.save(cf, np.array(ids, dtype=np.int32))
            chunk_files.append(cf)
            total += len(ids)
            dt = time.time() - t0
            rate = total / dt if dt > 0 else 0
            print(f"  chunk {len(chunk_files):4d} | {total:>12,} toks | {rate:,.0f} tok/s", flush=True)

    # Concatenate via memory-mapped reads — peak RAM = one chunk at a time
    print(f"\nConcatenating {len(chunk_files)} chunks...", flush=True)
    t1 = time.time()

    # First pass: count total tokens
    sizes = []
    for cf in chunk_files:
        arr = np.load(cf, mmap_mode="r")
        sizes.append(len(arr))
    total = sum(sizes)

    # Allocate output and fill chunk by chunk
    out = np.empty(total, dtype=np.int32)
    offset = 0
    for cf, sz in zip(chunk_files, sizes):
        arr = np.load(cf, mmap_mode="r")
        out[offset:offset + sz] = arr
        offset += sz

    torch.save(torch.from_numpy(out), dst)
    dt = time.time() - t0
    print(f"Done: {total:,} tokens in {dt:.0f}s ({total/dt:,.0f} tok/s) -> {dst}", flush=True)

    # Cleanup temp chunks
    for cf in chunk_files:
        os.unlink(cf)
    os.rmdir(tmpdir)


if __name__ == "__main__":
    main()
