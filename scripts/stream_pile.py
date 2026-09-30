#!/usr/bin/env python3
"""Stream The Pile deduplicated -> plain text (.txt) with HuggingFace datasets.

Uses `datasets.load_dataset(..., streaming=True)` -- no full parquet download needed.
Resumable: checks if target size already met. Skips short documents (<50 chars).

Run:
    cd /d/Projects/llm-lab && .venv/Scripts/python.exe scripts/stream_pile.py [--tokens 2000000000]
    
Target: ~2B tokens (~8-10 GB of raw text), enough for meaningful pretraining.
"""
import argparse
import os
import sys
import time
from pathlib import Path

DATA_DIR = Path("data")
OUT_FILE = DATA_DIR / "pile_train_full.txt"
CHARS_PER_TOKEN = 4  # rough estimate for English text
DATASET = "EleutherAI/the_pile_deduplicated"


def main() -> None:
    ap = argparse.ArgumentParser(description="Stream The Pile -> plain text")
    ap.add_argument("--tokens", type=int, default=2_000_000_000,
                    help="Target tokens (default 2B)")
    ap.add_argument("--min-doc-chars", type=int, default=50,
                    help="Skip documents shorter than this")
    args = ap.parse_args()

    target_chars = args.tokens * CHARS_PER_TOKEN
    already_done = OUT_FILE.stat().st_size if OUT_FILE.exists() else 0

    if already_done >= target_chars:
        print(f"Already have {already_done/1e6:.0f}M chars ({already_done // CHARS_PER_TOKEN:,} estimated tokens). Done.")
        return

    print(f"Streaming {DATASET} -> {OUT_FILE.name}")
    print(f"Target: {args.tokens:,} tokens (~{target_chars/1e9:.1f}B chars)")
    print(f"Already on disk: {already_done/1e9:.1f}B chars\n")

    from datasets import load_dataset

    ds = load_dataset(DATASET, split="train", streaming=True)

    t0 = time.time()
    total_chars = already_done
    total_docs = 0

    with open(OUT_FILE, "a", encoding="utf-8") as f:
        for example in ds:
            text = example.get("text", "").strip()
            if len(text) < args.min_doc_chars:
                continue

            f.write(text + "\n\n")
            total_chars += len(text) + 2
            total_docs += 1

            if total_docs % 1000 == 0:
                elapsed = time.time() - t0
                pct = total_chars / target_chars * 100
                rate = (total_chars - already_done) / elapsed / 1e6 if elapsed > 0 else 0
                print(f"  {total_docs:>8,} docs | {total_chars/1e6:>8.0f}M chars | {pct:5.1f}% | "
                      f"{rate:6.1f}M chars/s | {elapsed/60:.0f}min", flush=True)

            if total_chars >= target_chars:
                break

    elapsed = time.time() - t0
    final_size = OUT_FILE.stat().st_size
    est_tokens = final_size // CHARS_PER_TOKEN

    print(f"\nDone: {total_docs:,} docs, {final_size/1e6:.0f}M chars in {elapsed/60:.1f}min")
    print(f"Estimated tokens: ~{est_tokens:,}")
    print(f"Wrote: {OUT_FILE}")


if __name__ == "__main__":
    main()
