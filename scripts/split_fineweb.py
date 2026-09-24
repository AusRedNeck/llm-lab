#!/usr/bin/env python3
"""split_fineweb.py — Split FineWeb text into N chunks for parallel tokenization."""
import os
import sys

INPUT = "data/fineweb_combined.txt"
CHUNKS_DIR = "data/fineweb_chunks"
N_CHUNKS = 4  # number of parallel workers

def main():
    os.makedirs(CHUNKS_DIR, exist_ok=True)
    
    # Get total size
    total = os.path.getsize(INPUT)
    chunk_size = total // N_CHUNKS
    
    print(f"Splitting {total:,} bytes into {N_CHUNKS} chunks of ~{chunk_size:,} bytes each")
    
    with open(INPUT, 'r', encoding='utf-8', errors='replace') as f:
        for i in range(N_CHUNKS):
            out_path = os.path.join(CHUNKS_DIR, f"chunk_{i:02d}.txt")
            with open(out_path, 'w', encoding='utf-8') as out:
                bytes_written = 0
                while bytes_written < chunk_size:
                    line = f.readline()
                    if not line:
                        break
                    out.write(line)
                    bytes_written += len(line.encode('utf-8', errors='replace'))
            print(f"  chunk_{i:02d}.txt: {bytes_written:,} bytes")
    
    print("Done splitting")

if __name__ == "__main__":
    main()
