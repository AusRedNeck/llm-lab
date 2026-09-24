#!/usr/bin/env python3
"""Encode the full OpenWebText corpus with the 50k BPE tokenizer.

Uses token_io.encode_file_stream() + HfBpeShim to bridge HF format -> int list.
Writes raw int32 tokens (memmap-friendly, like the 16k corpus).

Usage:
    python encode_owt_50k.py [--chunk-chars 1000000] [--resume]
    
Output:
    data/openwebtext_combined_bpe_owt50k.bin   (~26 GB estimated)
"""
from __future__ import annotations

import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import token_io
from hf_bpe_shim import HfBpeShim

DATA = os.path.join(ROOT, "data")
SRC = os.path.join(DATA, "openwebtext_combined.txt")
VOCAB = os.path.join(ROOT, "references/pythia-50k-owt/tokenizer.json")
DEFAULT_OUT = os.path.join(DATA, "openwebtext_combined_bpe_owt50k.bin")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Encode full OWT with 50k BPE tokenizer")
    ap.add_argument("--src", default=SRC, help="Source text file")
    ap.add_argument("--vocab", default=VOCAB, help="Tokenizer JSON")
    ap.add_argument("--out", default=DEFAULT_OUT, help="Output .bin file")
    ap.add_argument("--chunk-chars", type=int, default=2_000_000,
                    help="Chunk size in chars (default 2M)")
    ap.add_argument("--resume", action="store_true", help="Resume interrupted run")
    args = ap.parse_args(argv)

    if not os.path.exists(args.src):
        raise SystemExit(f"source not found: {args.src}")
    if not os.path.exists(args.vocab):
        raise SystemExit(f"vocab not found: {args.vocab}")

    src_size = os.path.getsize(args.src)
    print(f"src   {os.path.basename(args.src)} ({token_io.human(src_size)})")
    print(f"vocab {os.path.basename(args.vocab)}")
    print(f"out   {os.path.basename(args.out)}")
    print(f"chunk {args.chunk_chars:,} chars\n", flush=True)

    # Load the HF tokenizer wrapped to return list[int]
    tok = HfBpeShim.from_file(args.vocab)
    actual_vocab = len(tok._hf.get_vocab())
    print(f"tokenizer vocab size: {actual_vocab:,}", flush=True)

    # Estimate output size
    ratio_est = 4.71  # from benchmark
    est_tokens = src_size / ratio_est
    est_bytes = int(est_tokens) * token_io.DTYPE.itemsize
    free = shutil_disk_usage(os.path.dirname(os.path.abspath(args.out))).free
    print(f"estimate {est_tokens/1e9:.2f}B tokens -> {token_io.human(est_bytes)} "
          f"(free: {token_io.human(free)} {'OK' if est_bytes < free*0.9 else 'WARNING'})\n",
          flush=True)

    t0 = time.time()
    result = token_io.encode_file_stream(
        args.src, args.vocab, args.out,
        chunk_chars=args.chunk_chars,
        resume=args.resume,
        write_sidecar=True,
        label="50k  ",
        tok=tok,
        log_every=10,
    )
    elapsed = time.time() - t0

    print(f"\n{'=' * 68}")
    print(f"result : {result['tokens']:,} tokens in {elapsed/60:.1f}m ({result['tok_per_sec']:,.0f} tok/s)")
    print(f"output : {args.out} ({token_io.human(result['bytes'])})")
    print(f"complete: {result['complete']}")
    print(f"failures: {result['encode_failures']}")

    return 0


def shutil_disk_usage(path):
    """Compatibility shim."""
    import shutil
    return shutil.disk_usage(path)


if __name__ == "__main__":
    import shutil
    sys.exit(main())
