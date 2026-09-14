# Train BPE on real TinyStories, save the vocab.
# Full file is 2.2GB: sample strided lines (diversity without the wait),
# learn merges, write bpe.json. Run: uv run python -m model.train_bpe
"""Train a BPE tokenizer on a sample of TinyStories.

Usage:
    uv run python -m model.train_bpe --lines 30000 --merges 2000
    uv run python -m model.train_bpe --lines 100000 --merges 8000 --out checkpoints/bpe8k.json
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.bpe import train_bpe


def load_sample(path: str, max_lines: int, stride: int) -> list[str]:
    # Every Nth line: spans the whole file instead of just the head.
    # Dir support: round-robin across .txt files (LibriSpeech-style trees)
    # so the sample covers many books, not just the first one.
    import glob
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "**", "*.txt"),
                                 recursive=True))
        if not files:
            raise FileNotFoundError(f"no .txt files under {path}")
        texts: list[str] = []
        # Stride per file, cycle files until quota met (diversity first).
        per_file_stride = max(1, stride // max(1, len(files) // 10))
        for fp in files:
            if len(texts) >= max_lines:
                break
            with open(fp, encoding="utf-8", errors="replace") as f:
                for i, line in enumerate(f):
                    if len(texts) >= max_lines:
                        break
                    if i % per_file_stride == 0:
                        line = line.strip()
                        if line:
                            texts.append(line)
        return texts
    texts = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if i % stride == 0:
                line = line.strip()
                if line:
                    texts.append(line)
                    if len(texts) >= max_lines:
                        break
    return texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/TinyStories.txt")
    ap.add_argument("--lines", type=int, default=30000)
    ap.add_argument("--stride", type=int, default=100)
    ap.add_argument("--merges", type=int, default=2000)
    ap.add_argument("--out", default="checkpoints/bpe2k.json")
    args = ap.parse_args()

    t0 = time.time()
    texts = load_sample(args.data, args.lines, args.stride)
    mb = sum(len(t) for t in texts) / 1e6
    print(f"sample: {len(texts)} lines, {mb:.1f}MB ({time.time() - t0:.0f}s)")

    t0 = time.time()
    tok = train_bpe(texts, args.merges)
    print(f"trained: vocab {len(tok.vocab)} in {time.time() - t0:.0f}s")

    # Sanity: a story-ish line should compress hard vs bytes.
    probe = "Once upon a time there was a little princess."
    n = len(tok.encode(probe))
    print(f"probe: {n} tokens vs {len(probe.encode())} bytes: {probe!r}")

    tok.save(args.out)
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
