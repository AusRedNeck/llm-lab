#!/usr/bin/env python3
"""Corpus probe: random-window dup hash + head/mid/tail char entropy + HTML sniff."""
import hashlib, math, pathlib, random
from collections import Counter

P = pathlib.Path("D:/Projects/llm-lab/data/openwebtext_combined.txt")
size = P.stat().st_size
print(f"combined: {size/1e9:.2f} GB")

# random-window duplicate hash: 400 windows of 2KB at random offsets
random.seed(1234)
hashes = []
with open(P, "rb") as f:
    for _ in range(400):
        off = random.randint(0, size - 4096)
        f.seek(off)
        chunk = f.read(2048)
        hashes.append(hashlib.blake2b(chunk, digest_size=8).hexdigest())
u = len(set(hashes))
print(f"windows: 400 sampled, {u} unique ({u/400*100:.1f}%)")

# head/mid/tail byte entropy (1MB each)
def entropy(data: bytes) -> float:
    c = Counter(data)
    n = len(data)
    return -sum(v/n * math.log2(v/n) for v in c.values())

with open(P, "rb") as f:
    for name, off in [("head", 0), ("mid", size//2), ("tail", size-1_000_000)]:
        f.seek(off)
        d = f.read(1_000_000)
        print(f"{name}: byte-entropy={entropy(d):.3f} bits/byte, bytes={len(d)}")

# token-level entropy via 4k tokenizer on 3 slices (10k chars each)
import sys
sys.path.insert(0, "D:/Projects/llm-lab")
from model.bpe import BPETokenizer
tok = BPETokenizer.load("D:/Projects/llm-lab/data/bpe_owt4k.json")
with open(P, "rb") as f:
    for name, off in [("head", 0), ("mid", size//2), ("tail", size-30_000)]:
        f.seek(off)
        s = f.read(20_000).decode("utf-8", errors="replace")
        ids = tok.encode(s)
        c = Counter(ids)
        n = len(ids)
        ent = -sum(v/n * math.log2(v/n) for v in c.values())
        print(f"{name}: tokens={n} tok-entropy={ent:.3f} bits/tok bytes/tok={len(s)/max(1,n):.2f}")
print("done")
