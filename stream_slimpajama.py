#!/usr/bin/env python3
"""Stream ~100M tokens from SlimPajama-627B to a text file.

Uses HuggingFace datasets streaming mode — no full download needed.
Resumable: skips tokens already written (checks file size).

Run: python stream_slimpajama.py [--tokens 100000000]
"""
import argparse
import os
import sys
import time
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
OUT_FILE = DATA_DIR / "slimpajama_100m.txt"
CHARS_PER_TOKEN = 4  # rough estimate
DATASET = "DKYoon/SlimPajama-6B"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=100_000_000,
                        help="Target tokens (default 100M)")
    parser.add_argument("--split", default="train", help="Dataset split")
    args = parser.parse_args()

    target_chars = args.tokens * CHARS_PER_TOKEN
    already_done = OUT_FILE.stat().st_size if OUT_FILE.exists() else 0
    if already_done >= target_chars:
        print(f"Already have {already_done/1e6:.0f}M chars ({already_done // CHARS_PER_TOKEN:,} tokens). Done.")
        return

    print(f"Streaming {DATASET} → {OUT_FILE.name}")
    print(f"Target: {args.tokens:,} tokens (~{target_chars/1e6:.0f}M chars)")
    print(f"Already on disk: {already_done/1e6:.0f}M chars")

    from datasets import load_dataset

    ds = load_dataset(DATASET, split=args.split, streaming=True)

    t0 = time.time()
    total_chars = already_done
    total_docs = 0

    with open(OUT_FILE, "a", encoding="utf-8") as f:
        for example in ds:
            text = example.get("text", "").strip()
            if len(text) < 100:
                continue

            f.write(text + "\n\n")
            total_chars += len(text) + 2
            total_docs += 1

            if total_docs % 1000 == 0:
                elapsed = time.time() - t0
                pct = total_chars / target_chars * 100
                rate = (total_chars - already_done) / elapsed / 1e6 if elapsed > 0 else 0
                print(f"  {total_docs:,} docs | {total_chars/1e6:.0f}M chars | "
                      f"{pct:.1f}% | {rate:.1f}M chars/s | {elapsed:.0f}s")

            if total_chars >= target_chars:
                break

    elapsed = time.time() - t0
    final_size = OUT_FILE.stat().st_size
    print(f"\nDone: {total_docs:,} docs, {final_size/1e6:.0f}M chars in {elapsed:.0f}s")
    print(f"Estimated tokens: ~{final_size // CHARS_PER_TOKEN:,}")


if __name__ == "__main__":
    main()
