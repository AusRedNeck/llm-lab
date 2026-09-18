#!/usr/bin/env python3
"""Tokenize a single text file with BPE, write as .npy int32 array.
Usage: python tokenize_chunk.py <src.txt> <vocab.json> <output.npy>
"""
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model.bpe import BPETokenizer

CHUNK_CHARS = 500_000

def main():
    src, vocab_path, outpath = sys.argv[1], sys.argv[2], sys.argv[3]
    tok = BPETokenizer.load(vocab_path)
    print(f"Vocab: {len(tok.vocab)}", flush=True)

    total = 0
    t0 = time.time()
    with open(src, encoding="utf-8", errors="replace") as f:
        while True:
            text = f.read(CHUNK_CHARS)
            if not text:
                break
            ids = tok.encode(text)
            total += len(ids)
            dt = time.time() - t0
            if total % 5000000 < 100000:
                print(f"  {total:,} toks ({total/dt:,.0f} tok/s)", flush=True)

    np.save(outpath, np.array(ids, dtype=np.int32))
    dt = time.time() - t0
    print(f"Done: {total:,} toks in {dt:.0f}s -> {outpath}", flush=True)

if __name__ == "__main__":
    main()
