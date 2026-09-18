#!/usr/bin/env python3
"""Shared token I/O for the llm-lab corpus pipeline.

Three rules, each one exists because an earlier run blew up:

  1. Tokens live on disk as raw little-endian int32 and are appended chunk by
     chunk. Nothing accumulates in RAM, and nothing goes near the system temp
     dir (%TEMP% is on C:). Corpus work stays under llm-lab/data on D:.
  2. Every output gets a sidecar (.meta.json / .manifest.json) recording what
     was written, so a killed run resumes instead of starting over.
  3. Big token files are read with np.memmap. A 38GB corpus gets trained on
     without ever being loaded into the 34GB the machine actually has.

Write with encode_file_stream(), read with memmap_tokens(). That's the contract.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from model.bpe import BPETokenizer

# Little-endian int32: fits any vocab past 4B tokens-of-range, matches the
# int32 caches the trainer already loads, and memmaps cleanly.
DTYPE = np.dtype("<i4")


def data_dir() -> str:
    # Everything processed lands together, on the same drive as the sources.
    return os.path.join(ROOT, "data")


def human(n: int) -> str:
    for unit, div in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(n) >= div:
            return f"{n / div:.2f}{unit}"
    return str(n)


def fmt_eta(seconds: float) -> str:
    seconds = max(0.0, seconds)
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m:02d}m" if h else f"{m}m {s:02d}s"


def write_meta(path: str, meta: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    os.replace(tmp, path)          # atomic: a crash mid-write can't corrupt it


def read_meta(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def count_tokens(path: str) -> int:
    # The file *is* the token count. No sidecar needed for this part.
    return os.path.getsize(path) // DTYPE.itemsize


def memmap_tokens(path: str) -> np.ndarray:
    n = count_tokens(path)
    return np.memmap(path, dtype=DTYPE, mode="r", shape=(n,))


def encode_file_stream(src: str, vocab_path: str, dst_bin: str, *,
                       chunk_chars: int = 500_000, limit_chars: int = 0,
                       resume: bool = False, append: bool = False,
                       write_sidecar: bool = True,
                       log_every: int = 20,
                       label: str = "", tok: BPETokenizer | None = None) -> dict:
    """Encode src -> dst_bin as raw int32, one chunk at a time.

    append=True  : add these tokens at the end of dst_bin (next corpus shard).
    resume=True  : pick up mid-file where the sidecar left off.
    limit_chars  : stop after N chars (smoke tests, partial corpora).
    write_sidecar: record progress in <dst_bin>.meta.json. Off when a caller
                   (the shard orchestrator) keeps its own manifest instead.
    """
    if tok is None:
        tok = BPETokenizer.load(vocab_path)

    meta_path = dst_bin + ".meta.json"
    src_abs = os.path.abspath(src)
    chars = 0
    tokens = 0

    if append:
        out_mode = "ab"
    else:
        if resume and os.path.exists(meta_path) and os.path.exists(dst_bin):
            m = read_meta(meta_path)
            reusable = (m.get("src") == src_abs and m.get("chunk_chars") == chunk_chars
                        and not m.get("complete") and m.get("chars"))
            if reusable:
                chars, tokens = int(m["chars"]), int(m["tokens"])
                # A crash can leave bytes past the last recorded chunk — those
                # tokens would shift every later token if we kept them.
                with open(dst_bin, "r+b") as f:
                    f.truncate(tokens * DTYPE.itemsize)
                print(f"  resume: {chars:,} chars / {tokens:,} tokens already on disk",
                      flush=True)
        if not chars:
            open(dst_bin, "wb").close()      # fresh: never inherit old bytes
        out_mode = "r+b" if chars else "wb"

    t0 = time.time()
    n_chunks = 0

    with open(src, encoding="utf-8", errors="replace") as f, open(dst_bin, out_mode) as out:
        if chars:
            f.read(chars)               # text mode: same unit we recorded
        while True:
            if limit_chars and chars >= limit_chars:
                break
            n = chunk_chars if not limit_chars else min(chunk_chars, limit_chars - chars)
            text = f.read(n)
            if not text:
                break
            ids = tok.encode(text)
            np.asarray(ids, dtype=DTYPE).tofile(out)
            chars += len(text)
            tokens += len(ids)
            n_chunks += 1
            if log_every and n_chunks % log_every == 0:
                dt = time.time() - t0
                rate = tokens / dt if dt else 0
                tag = f"{label} " if label else ""
                print(f"  {tag}chunk {n_chunks:5d} | {tokens:>12,} toks | "
                      f"{rate:,.0f} tok/s", flush=True)
            if write_sidecar and (n_chunks % 25 == 0):
                write_meta(meta_path, {
                    "src": src_abs, "vocab": os.path.abspath(vocab_path),
                    "chunk_chars": chunk_chars, "chars": chars, "tokens": tokens,
                    "complete": False,
                })

    # "complete" = we reached EOF. A limit_chars run is a deliberate slice,
    # so it is never a complete transcript of the source.
    complete = (limit_chars == 0) or (chars < limit_chars)
    out_size = os.path.getsize(dst_bin)
    if write_sidecar:
        write_meta(meta_path, {
            "src": src_abs, "vocab": os.path.abspath(vocab_path),
            "chunk_chars": chunk_chars, "chars": chars, "tokens": tokens,
            "bytes": out_size, "complete": complete,
        })
    dt = time.time() - t0
    return {"chars": chars, "tokens": tokens, "bytes": out_size,
            "seconds": dt, "complete": complete,
            "tok_per_sec": tokens / dt if dt else 0.0}


def memmap_tensor(path: str):
    """np.memmap-backed torch tensor — training reads without loading RAM."""
    import warnings
    import torch
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")       # torch nags about read-only arrays
        t = torch.from_numpy(memmap_tokens(path))
    print(f"  memmap {os.path.basename(path)}: {len(t):,} tokens "
          f"({human(len(t) * 4)} on disk, 0 RAM)", flush=True)
    return t


def bins_to_pt(src_bin: str, dst_pt: str) -> str:
    """Convert a raw int32 token file to a torch .pt (memmapped, not loaded)."""
    import torch
    t = memmap_tensor(src_bin)
    torch.save(t, dst_pt)
    print(f"  wrote {dst_pt} ({human(os.path.getsize(dst_pt))})", flush=True)
    return dst_pt


def rss_mb() -> float:
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
    except Exception:
        return 0.0
