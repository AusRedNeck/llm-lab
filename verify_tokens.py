#!/usr/bin/env python3
"""Acceptance checks for a token file — run this before believing a long encode.

Checks (all six must pass; exit 0 = PASS):

  1. shape     — size is a whole number of int32 tokens, file is non-empty
  2. manifest  — per-shard token counts sum to the file size (no gaps/overlap)
  3. range     — every sampled id is inside [0, vocab_size)
  4. roundtrip — decode(tokens[:N]) reproduces the source text
  5. boundary  — the tokens sitting at each shard's manifest offset really are
                 that shard's text (catches append/offset bugs)
  6. loadable  — the file loads the way train.py loads it (memmap path)

Usage:
    python verify_tokens.py --bin data/tok16k_smoke.bin --vocab data/bpe_owt16k.json --src data/openwebtext/shards/train-00000-of-00080.txt
    python verify_tokens.py --bin data/openwebtext_combined_bpe_owt16k.bin --vocab data/bpe_owt16k.json --manifest
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

import token_io
from token_io import DTYPE, count_tokens, human, memmap_tokens, read_meta
from model.bpe import BPETokenizer

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name: str, ok: bool, detail: str) -> bool:
    results.append((name, PASS if ok else FAIL, detail))
    print(f"  [{PASS if ok else FAIL}] {name}: {detail}", flush=True)
    return ok


def prefix_tokens(tok: BPETokenizer, text: str, drop_tail: int = 8) -> list[int]:
    """Encode a text slice and drop the tail — the last regex chunk of a slice
    can differ from the same text inside a bigger chunk."""
    ids = tok.encode(text)
    return ids[:-drop_tail] if len(ids) > drop_tail else ids


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True, dest="bin_path")
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--src", default="", help="source text for the round-trip check")
    ap.add_argument("--manifest", action="store_true",
                    help="also verify against <bin>.manifest.json")
    ap.add_argument("--probe-chars", type=int, default=20_000)
    ap.add_argument("--shards-dir", default=os.path.join(token_io.data_dir(), "openwebtext", "shards"))
    ap.add_argument("--boundary-checks", type=int, default=3,
                    help="how many shard offsets to verify (first/middle/last)")
    args = ap.parse_args()

    path = args.bin_path
    print(f"verifying {path}")
    if not os.path.exists(path):
        print(f"  [{FAIL}] missing file")
        return 1

    size = os.path.getsize(path)
    n = count_tokens(path)
    check("shape", size % DTYPE.itemsize == 0 and n > 0,
          f"{human(size)} = {n:,} int32 tokens"
          + (f" (+{size % 4} stray bytes!)" if size % 4 else ""))

    tok = BPETokenizer.load(args.vocab)
    vocab_size = len(tok.vocab)
    arr = memmap_tokens(path)

    manifest = read_meta(path + ".manifest.json") if args.manifest else {}
    if args.manifest:
        recs = manifest.get("shards", {})
        claimed = sum(int(r["tokens"]) for r in recs.values())
        gaps = n - claimed
        check("manifest", claimed == n,
              f"{len(recs)} shard record(s) claim {claimed:,} tokens "
              f"(file has {n:,}; delta {gaps:+,})" + ("" if claimed == n else "  <-- MISMATCH"))
    else:
        recs = {}

    # Range: sample the ends plus a handful of random windows — a full pass over
    # 10B tokens is I/O we don't need to spend.
    samples = [arr[:500_000], arr[-500_000:]]
    rng = np.random.default_rng(0)
    for _ in range(4):
        start = int(rng.integers(0, max(1, n - 100_000)))
        samples.append(arr[start:start + 100_000])
    lo = min(int(s.min()) for s in samples)
    hi = max(int(s.max()) for s in samples)
    check("range", 0 <= lo and hi < vocab_size,
          f"sampled ids in [{lo}, {hi}], vocab {vocab_size}")

    if args.src:
        with open(args.src, encoding="utf-8", errors="replace") as f:
            text = f.read(args.probe_chars)
        # Encode the slice standalone, then drop the last few tokens: the file
        # kept encoding past the cut, so its version of the final partial word
        # is legitimately longer. Everything before that must match exactly.
        want = prefix_tokens(tok, text)
        got = list(arr[: len(want)])
        check("roundtrip", want == got,
              f"{len(want):,} tokens are byte-for-byte the start of "
              f"{os.path.basename(args.src)}")
        # Belt and braces: the decoded text must literally be source text.
        decoded = tok.decode(got)
        check("roundtrip-text", text.startswith(decoded),
              f"decode()s back to {len(decoded):,} chars, a verbatim prefix of the source")
    else:
        check("roundtrip", True, "skipped (no --src given)")
        check("roundtrip-text", True, "skipped (no --src given)")

    # Boundary check: does the token at each shard offset match that shard's text?
    if recs:
        names = list(recs)
        picks = sorted({0, len(names) // 2, len(names) - 1})
        picks = picks[: max(1, args.boundary_checks)] if args.boundary_checks else []
        offset = 0
        offsets = []
        for nm in names:
            offsets.append(offset)
            offset += int(recs[nm]["tokens"])
        ok_all, details = True, []
        for idx in picks:
            nm = names[idx]
            src_path = os.path.join(args.shards_dir, nm)
            if not os.path.exists(src_path):
                details.append(f"{nm}: source missing, skipped")
                continue
            with open(src_path, encoding="utf-8", errors="replace") as f:
                text = f.read(args.probe_chars)
            want = prefix_tokens(tok, text)
            got = list(arr[offsets[idx]: offsets[idx] + len(want)])
            match = want == got
            ok_all &= match
            details.append(f"{nm}@{offsets[idx]:,} {'ok' if match else 'MISMATCH'}")
        check("boundary", ok_all, "; ".join(details))
    else:
        check("boundary", True, "skipped (use --manifest to check shard offsets)")

    try:
        t = token_io.memmap_tensor(path)
        import torch
        check("loadable", t.dtype == torch.int32 and len(t) == n and t.is_contiguous(),
              f"torch {str(t.dtype)} tensor, {len(t):,} elements, contiguous, mapped not loaded")
    except Exception as exc:                      # pragma: no cover - diagnostic path
        check("loadable", False, f"{type(exc).__name__}: {exc}")

    failed = [r for r in results if r[1] == FAIL]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed"
          f"{' — ALL GREEN' if not failed else ' — ' + ', '.join(r[0] for r in failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
