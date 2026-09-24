#!/usr/bin/env python3
"""Split a large text file into N roughly equal chunks by line count.
Usage: python split_chunks.py <input.txt> <n_chunks> <output_dir>
"""
import os
import sys

def main():
    src = sys.argv[1]
    n = int(sys.argv[2])
    outdir = sys.argv[3]
    os.makedirs(outdir, exist_ok=True)

    # Count lines first
    print(f"Counting lines in {src}...", flush=True)
    with open(src, encoding="utf-8", errors="replace") as f:
        total = sum(1 for _ in f)
    per = total // n
    print(f"{total:,} lines -> {n} chunks of ~{per:,}", flush=True)

    # Split
    with open(src, encoding="utf-8", errors="replace") as f:
        for i in range(n):
            outpath = os.path.join(outdir, f"chunk_{i:02d}.txt")
            count = per if i < n - 1 else total - per * (n - 1)
            print(f"  writing chunk_{i:02d}.txt ({count:,} lines)...", flush=True)
            with open(outpath, "w", encoding="utf-8") as out:
                for _ in range(count):
                    line = f.readline()
                    if not line:
                        break
                    out.write(line)
    print("Done splitting.", flush=True)

if __name__ == "__main__":
    main()
