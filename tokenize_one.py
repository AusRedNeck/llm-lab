#!/usr/bin/env python3
"""Tokenize a text file with BPE using streaming (low RAM).
Writes temp .npy chunks, concatenates at end.
Usage: python tokenize_one.py <src.txt> <vocab.json> <output.npy>
"""
import os
import sys
import time
import tempfile
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model.bpe import BPETokenizer

CHUNK_CHARS = 500_000

def main():
    src, vocab_path, outpath = sys.argv[1], sys.argv[2], sys.argv[3]
    tok = BPETokenizer.load(vocab_path)
    print(f"Vocab: {len(tok.vocab)}", flush=True)

    tmpdir = tempfile.mkdtemp(prefix="tok_")
    parts = []
    total = 0
    t0 = time.time()

    with open(src, encoding="utf-8", errors="replace") as f:
        while True:
            text = f.read(CHUNK_CHARS)
            if not text:
                break
            ids = tok.encode(text)
            cf = os.path.join(tmpdir, f"p_{len(parts):05d}.npy")
            np.save(cf, np.array(ids, dtype=np.int32))
            parts.append(cf)
            total += len(ids)
            dt = time.time() - t0
            if len(parts) % 20 == 0:
                print(f"  {len(parts)} sub-chunks | {total:,} toks | {total/dt:,.0f} tok/s", flush=True)

    # Concatenate
    sizes = [len(np.load(p, mmap_mode="r")) for p in parts]
    total = sum(sizes)
    out = np.empty(total, dtype=np.int32)
    off = 0
    for p, sz in zip(parts, sizes):
        arr = np.load(p, mmap_mode="r")
        out[off:off+sz] = arr
        off += sz
    np.save(outpath, out)

    for p in parts:
        os.unlink(p)
    os.rmdir(tmpdir)

    dt = time.time() - t0
    print(f"Done: {total:,} toks in {dt:.0f}s ({total/dt:,.0f} tok/s) -> {outpath}", flush=True)

if __name__ == "__main__":
    main()
