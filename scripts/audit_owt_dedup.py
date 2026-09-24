#!/usr/bin/env python3
"""Full OWT dedup scan: exact + normalized hash over all 8M docs."""
import hashlib, re, pathlib
import numpy as np
import pyarrow.parquet as pq

SRC = pathlib.Path("D:/Projects/llm-lab/data/openwebtext/plain_text")
files = sorted(SRC.glob("*.parquet"))
TOTAL = 8013769  # measured row count

ws_re = re.compile(r"\s+")

def h64(b: bytes) -> int:
    return int.from_bytes(hashlib.blake2b(b, digest_size=8).digest(), "little")

exact = np.empty(TOTAL, dtype=np.uint64)
norm = np.empty(TOTAL, dtype=np.uint64)
pos = 0
short = 0
empty = 0
for f in files:
    tbl = pq.read_table(str(f), columns=["text"])
    texts = tbl.column("text").to_pylist()
    del tbl
    for t in texts:
        if not t:
            empty += 1
            s = ""
        else:
            s = t.strip()
        if len(s) < 100:
            short += 1
        exact[pos] = h64(s.encode("utf-8", errors="ignore"))
        n = ws_re.sub(" ", s.lower()).strip()
        norm[pos] = h64(n.encode("utf-8", errors="ignore"))
        pos += 1
    print(f"  {f.name}: pos={pos:,}", flush=True)

print(f"docs={pos:,} empty={empty} short<100={short}")
for name, arr in [("exact", exact), ("norm", norm)]:
    u, c = np.unique(arr, return_counts=True)
    dup_docs = int((c - 1).sum())          # docs that are copies beyond first
    dup_groups = int((c > 1).sum())        # distinct texts with >1 copy
    print(f"{name}: unique={len(u):,} dup_groups={dup_groups:,} dup_docs={dup_docs:,} "
          f"dup_rate={dup_docs/pos*100:.3f}%")
    # top 5 most-copied
    idx = np.argsort(-c)[:5]
    print(f"  top multiplicities: {c[idx].tolist()}")
    # save duplicated hashes for example fetch
    np.save(f"D:/Projects/llm-lab/data/dup_{name}_hashes.npy",
            u[c > 1][:10000])
print("saved dup hash lists")
