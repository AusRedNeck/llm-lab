#!/usr/bin/env python3
"""Tokenize OWT shards with 16k vocab, serialize (one file at a time), concatenate result.

Usage:
    python d:/Projects/llm-lab/tokenize_owt_16k.py

Output: openwebtext_combined_bpe_owt16k.pt (~45-50GB)
Total estimated runtime: ~6 hours on 4070 Ti CPU.
"""
import os, sys, glob, shutil
import numpy as np
import torch

SCRIPT_DIR = r"D:\Projects\llm-lab"
SHARD_DIR = os.path.join(SCRIPT_DIR, "data", "openwebtext", "shards")
VOCAB_JSON = os.path.join(SCRIPT_DIR, "data", "bpe_owt16k.json")
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "data", "openwebtext_combined_bpe_owt16k.pt")
TOKENIZE_SCRIPT = os.path.join(SCRIPT_DIR, "tokenize_stream.py")
TEMP_DIR = os.path.join(SCRIPT_DIR, ".tmp_shard_tokens")


def main():
    # Find shards sorted by number
    shards = sorted(glob.glob(os.path.join(SHARD_DIR, "*.txt")))
    print(f"Found {len(shards)} shards ({sum(os.path.getsize(s) for s in shards)/(1024**3):.1f} GB total)", flush=True)
    if len(shards) != 80:
        print(f"WARNING: expected 80 shards, found {len(shards)}", flush=True)

    # Resume check: find already-encoded shards in TEMP_DIR
    existing_shards = set()
    if os.path.isdir(TEMP_DIR):
        for f in os.listdir(TEMP_DIR):
            if f.endswith("_encoded.pt"):
                existing_shards.add(f.replace("_encoded.pt", ""))
    done = len(existing_shards)
    remaining = len(shards) - done
    print(f"Resumed state: {done} done, {remaining} to go", flush=True)

    # Ensure temp dir exists (resume-safe)
    os.makedirs(TEMP_DIR, exist_ok=True)

    # Phase 1: encode each shard individually via tokenize_stream.py
    shard_token_files = []
    for i, shard_path in enumerate(shards):
        basename = os.path.basename(shard_path).replace(".txt", "")
        out_name = f"{basename}_encoded.pt"
        out_path = os.path.join(TEMP_DIR, out_name)

        cmd = [sys.executable, "-u", TOKENIZE_SCRIPT, shard_path, VOCAB_JSON, out_path]
        
        # Skip if already encoded (resume support)
        cached = os.path.join(TEMP_DIR, f"{basename}_encoded.pt")
        if basename in existing_shards and os.path.exists(cached):
            sz = os.path.getsize(cached) / (1024**2)
            print(f"[{i+1}/{len(shards)}] {basename} [SKIP - already encoded {sz:.0f}MB]", flush=True)
            shard_token_files.append(cached)
            continue
        
        print(f"[{i+1}/{len(shards)}] {basename} [{os.path.getsize(shard_path)//1024//1024}MB]", flush=True)
        
        import subprocess
        proc = subprocess.run(cmd, capture_output=False)
        if proc.returncode != 0:
            print(f"  FAILED exit={proc.returncode}", flush=True)
            continue
        
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            shard_token_files.append(out_path)
            sz = os.path.getsize(out_path) / (1024**2)
            print(f"  -> {sz:.0f} MB", flush=True)
        else:
            print(f"  WARNING: no output!", flush=True)

    if not shard_token_files:
        print("ERROR: No shards encoded!")
        sys.exit(1)

    # Phase 2: concatenate all shard tensors
    print(f"\nConcatenating {len(shard_token_files)} shard encodings...", flush=True)
    
    offset = 0
    total_tokens = 0
    for fp in shard_token_files:
        t = torch.load(fp, weights_only=True)
        n = len(t)
        total_tokens += n
        offset += n
        print(f"  + {n:,} tokens ({total_tokens:,} total)", flush=True)

    # Re-build from individual tensors to avoid OOM
    combined_parts = []
    for fp in shard_token_files:
        t = torch.load(fp, weights_only=True)
        combined_parts.append(t)
    combined = torch.cat(combined_parts)
    
    del combined_parts
    torch.cuda.empty_cache() if hasattr(torch, 'cuda') else None
    
    print(f"Writing {OUTPUT_FILE}...", flush=True)
    torch.save(combined, OUTPUT_FILE)
    
    import time
    out_size_mb = os.path.getsize(OUTPUT_FILE) / (1024**2)
    print(f"\nDONE: {total_tokens:,} total tokens -> {out_size_mb:.0f} MB ({out_size_mb/1024:.1f} GB)")

    # Cleanup temp files
    for fp in shard_token_files:
        try:
            os.unlink(fp)
        except:
            pass
    try:
        shutil.rmtree(TEMP_DIR, ignore_errors=True)
    except:
        pass
    print("Temp cleaned up.")


if __name__ == "__main__":
    main()
