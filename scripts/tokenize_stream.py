#!/usr/bin/env python3
"""Stream-tokenize one text file with BPE → raw int32 token file.

Replaces the old version that parked intermediate chunks in the system temp
directory (%TEMP%, i.e. C:) and concatenated via a full in-RAM numpy buffer.
Now it appends straight to the destination on disk, so peak RAM is one chunk.

Usage:
    python -u tokenize_stream.py <src.txt> <vocab.json> <dst.bin>
    python -u tokenize_stream.py <src.txt> <vocab.json> <dst.pt>    # + .pt conversion
    python -u tokenize_stream.py <src.txt> <vocab.json> <dst.bin> --resume
    python -u tokenize_stream.py <src.txt> <vocab.json> <dst.bin> --chunk-chars 2000000

Notes:
  * dst.bin is the real artifact: raw little-endian int32, memmap-friendly.
  * A .pt destination keeps the old command shape working — the .bin is written
    first, then converted without ever loading the whole thing into RAM.
  * --resume continues an interrupted run from <dst>.meta.json.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.dirname(os.path.abspath(__file__))
for path in (ROOT, SCRIPTS):
    if path not in sys.path:
        sys.path.insert(0, path)

import token_io
from token_io import encode_file_stream, bins_to_pt, human


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("vocab")
    ap.add_argument("dst")
    ap.add_argument("--chunk-chars", type=int, default=500_000)
    ap.add_argument("--limit-chars", type=int, default=0,
                    help="stop after N chars (smoke slices)")
    ap.add_argument("--resume", action="store_true",
                    help="continue from <dst>.meta.json instead of restarting")
    args = ap.parse_args()

    if not os.path.exists(args.src):
        raise SystemExit(f"source not found: {args.src}")
    if not os.path.exists(args.vocab):
        raise SystemExit(f"vocab not found: {args.vocab}")

    # A .pt destination is a convenience wrapper: the token stream itself is
    # always the .bin, and the .pt is a thin conversion at the end.
    wants_pt = args.dst.endswith(".pt")
    bin_path = args.dst[:-3] + ".bin" if wants_pt else args.dst
    os.makedirs(os.path.dirname(os.path.abspath(bin_path)) or ".", exist_ok=True)

    print(f"src   {args.src} ({human(os.path.getsize(args.src))})")
    print(f"vocab {args.vocab}")
    print(f"bin   {bin_path}")
    print(flush=True)

    t0 = time.time()
    result = encode_file_stream(args.src, args.vocab, bin_path,
                               chunk_chars=args.chunk_chars,
                               limit_chars=args.limit_chars,
                               resume=args.resume,
                               log_every=max(1, int(20 * 500_000 / args.chunk_chars)))
    print(f"\n{result['tokens']:,} tokens in {time.time() - t0:,.0f}s "
          f"({result['tok_per_sec']:,.0f} tok/s) -> {bin_path} "
          f"({human(result['bytes'])})", flush=True)

    if wants_pt:
        print(f"\nconverting to {args.dst} ...", flush=True)
        bins_to_pt(bin_path, args.dst)
    return 0


if __name__ == "__main__":
    sys.exit(main())
