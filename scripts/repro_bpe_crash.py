"""Reproduce the encoder crash on shard 33 and pinpoint the exact input.

The traceback said model/bpe.py line 69 (the merge slice) got a non-integer
index, which shouldn't be possible from the code as written. Find the real cause
by re-running the same loop and dumping the offending chunk.
"""
import sys

sys.path.insert(0, ".")
from model.bpe import BPETokenizer, _SPLIT

SRC = "data/openwebtext/shards/train-00033-of-00080.txt"
CHUNK = 500_000

tok = BPETokenizer.load("data/bpe_owt16k.json")
print(f"vocab {len(tok.vocab)}, merges {len(tok.merges)}")

with open(SRC, encoding="utf-8", errors="replace") as f:
    n = 0
    while True:
        text = f.read(CHUNK)
        if not text:
            print("reached EOF without crashing")
            break
        n += 1
        try:
            tok.encode(text)
            print(f"chunk {n:4d} ok ({len(text):,} chars)", flush=True)
        except Exception as exc:
            print(f"\nFAILED at chunk {n} ({len(text):,} chars): {type(exc).__name__}: {exc}")
            # Walk the same regex chunks to find the exact one that blows up.
            for i, chunk in enumerate(_SPLIT.findall(text)):
                parts = [bytes([b]) for b in chunk.encode("utf-8")]
                try:
                    while len(parts) > 1:
                        best, best_rank = None, None
                        for j in range(len(parts) - 1):
                            rank = tok.merges.get((parts[j], parts[j + 1]))
                            if rank is not None and (best_rank is None or rank < best_rank):
                                best, best_rank = j, rank
                        if best is None:
                            break
                        parts[best:best + 2] = [parts[best] + parts[best + 1]]
                except Exception as exc2:
                    print(f"  regex chunk #{i}: {type(exc2).__name__}: {exc2}")
                    print(f"  chunk repr : {chunk!r}"[:400])
                    print(f"  best       : {best!r} ({type(best).__name__})")
                    print(f"  best_rank  : {best_rank!r} ({type(best_rank).__name__})")
                    print(f"  parts[:6]  : {[p for p in parts[:6]]}")
                    bad = [k for k, v in tok.merges.items() if not isinstance(v, int)]
                    print(f"  non-int merge values in vocab: {len(bad)} e.g. {bad[:3]}")
                    dupes = [k for k in tok.merges if not isinstance(k, tuple) or len(k) != 2]
                    print(f"  malformed merge keys: {len(dupes)} e.g. {dupes[:3]}")
                    break
            break
