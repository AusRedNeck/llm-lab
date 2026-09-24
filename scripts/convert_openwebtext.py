#!/usr/bin/env python3
"""Convert OpenWebText parquets → plain text.

Design: low memory, crash-safe, resumable.
  - Each worker writes one shard file (no shared state)
  - 2 workers max (each parquet is ~300MB, don't eat the machine)
  - Final concat step after all shards complete
  - Resumable: skip shards that already exist

Run: python convert_openwebtext.py
"""
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "openwebtext"
SRC_DIR = DATA_DIR / "plain_text"
SHARD_DIR = DATA_DIR / "shards"
OUT_FILE = DATA_DIR.parent / "openwebtext_combined.txt"
MIN_TEXT_LEN = 100
MAX_WORKERS = 2  # each worker loads ~300MB parquet — don't be greedy


def process_one(args):
    """Convert one parquet to a .txt shard. Returns (name, docs, chars, skipped|None)."""
    src_path, shard_path, shard_name = args
    shard_path = Path(shard_path)
    src_path = Path(src_path)

    # Skip if already done
    if shard_path.exists() and shard_path.stat().st_size > 0:
        return (shard_name, 0, 0, "skipped")

    try:
        table = pq.read_table(str(src_path), columns=["text"])
    except Exception as e:
        return (shard_name, 0, 0, f"ERROR: {e}")

    texts = []
    for val in table["text"]:
        text = str(val).strip()
        if len(text) > MIN_TEXT_LEN:
            texts.append(text)

    # Free the table before writing
    del table

    # Write shard — stream, don't hold full string in memory
    tmp_path = shard_path.with_suffix(".tmp")
    n_docs = len(texts)
    n_chars = 0
    with open(tmp_path, "w", encoding="utf-8") as f:
        for i, text in enumerate(texts):
            if i > 0:
                f.write("\n\n")
            f.write(text)
            n_chars += len(text)
    tmp_path.rename(shard_path)

    return (shard_name, n_docs, n_chars, None)


def main():
    SHARD_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(f for f in SRC_DIR.iterdir() if f.suffix == ".parquet")
    print(f"Found {len(files)} parquet files in {SRC_DIR}")

    tasks = []
    for src_path in files:
        shard_name = src_path.stem + ".txt"
        shard_path = SHARD_DIR / shard_name
        tasks.append((str(src_path), str(shard_path), shard_name))

    print(f"Converting with {MAX_WORKERS} workers (gentle on RAM)...")

    t0 = time.time()
    total_docs = 0
    total_chars = 0
    skipped = 0
    errors = 0

    with Pool(processes=MAX_WORKERS) as pool:
        for i, result in enumerate(pool.imap_unordered(process_one, tasks), 1):
            name, n_docs, n_chars, status = result
            if status and "ERROR" in status:
                errors += 1
                print(f"  ERROR {name}: {status}")
            elif status == "skipped":
                skipped += 1
            else:
                total_docs += n_docs
                total_chars += n_chars

            if i % 10 == 0 or i == len(files):
                elapsed = time.time() - t0
                done = i - skipped - errors
                rate = total_chars / elapsed / 1e6 if elapsed > 0 else 0
                print(f"  {i}/{len(files)}: {done} converted, {skipped} skip, {errors} err | "
                      f"{total_docs:,} docs, {total_chars/1e6:.0f}M chars | "
                      f"{rate:.1f}M chars/s ({elapsed:.0f}s)")

    # Concat shards
    shard_files = sorted(SHARD_DIR.glob("*.txt"))
    print(f"\nConcatenating {len(shard_files)} shards → {OUT_FILE.name}")

    with open(OUT_FILE, "w", encoding="utf-8") as out:
        for shard in shard_files:
            size = shard.stat().st_size
            if size > 0:
                with open(shard, "r", encoding="utf-8") as inp:
                    while True:
                        chunk = inp.read(8 * 1024 * 1024)  # 8MB chunks
                        if not chunk:
                            break
                        out.write(chunk)
                out.write("\n\n")

    elapsed = time.time() - t0
    final_size = OUT_FILE.stat().st_size
    print(f"\nDone: {total_chars:,} chars, {total_docs:,} docs in {elapsed:.0f}s")
    print(f"File: {final_size/1024/1024:.0f} MB")
    print(f"Estimated tokens: ~{final_size // 4:,}")


if __name__ == "__main__":
    main()
