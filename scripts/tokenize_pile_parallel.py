#!/usr/bin/env python3
"""Tokenize one large text file in parallel -> raw int32 token file.

Why this exists: the Pile pull is 8.1GB / ~2.05B tokens. Serial encoding runs
at ~680k tok/s on this box (measured, HF encoder), so ~50 minutes. This splits
the source into `--workers` contiguous byte ranges, aligns every boundary to a
newline, encodes each range in its own process, and concatenates in order.

Correctness rules, each from a way this has bitten before:

  1. Boundaries are NEWLINE-ALIGNED and verified. BPE is applied per regex
     chunk, so a split inside a word can merge differently across the seam and
     every token after it shifts. `tests/test_tokenize_parallel.py` asserts
     split == whole on a real slice; `--selftest` re-checks on live data.
  2. Parts are concatenated in INDEX order, and the parent verifies every
     part's byte count against its meta before merging. A missing part must
     fail loudly, never silently produce a short token stream.
  3. Merge streams bytes (never `torch.cat` / numpy concatenate into RAM).
     Peak RAM is one 500K-char chunk per worker.
  4. Separate processes, not mp.Pool -- Pool crashed silently on Windows.
  5. Resume is per-part: a killed run restarts only the parts still marked
     incomplete in <out>.parts.json.
  6. BYTES IN, TOKENS OUT. The source is read in binary and decoded with an
     incremental UTF-8 decoder, so every byte survives. This is not pedantry:
     the Pile (and the OWT pulls) are CRLF corpora -- 379,519 CRLF endings in
     30MB. Opening the source in Python text mode with the default
     `newline=None` silently rewrites every \\r\\n to \\n, which drops 0.77% of
     the characters before the tokenizer ever sees them (measured: 7,454,059
     tokens text-mode vs 7,511,674 byte-faithful on the same 30MB). That is a
     different corpus, so the cache would not be reproducible from the source.
     See README "CRLF: read corpora as bytes".

Known and accepted: a boundary-aligned split is not byte-identical to encoding
the whole file as one string, because BPE can merge across a chunk seam. On
30MB / 8 workers that is +-8 tokens per range (total -2 of 7.5M, i.e. 3e-7) --
the same order as the serial path's own chunk-boundary noise (+-60 on the same
file). Exactness would need the whole 8GB in RAM, which is the thing the
memmap pipeline exists to avoid.

Usage:
    python -u scripts/tokenize_pile_parallel.py SRC VOCAB OUT --workers 8
    python -u scripts/tokenize_pile_parallel.py SRC VOCAB OUT --selftest
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.dirname(os.path.abspath(__file__))
for path in (ROOT, SCRIPTS):
    if path not in sys.path:
        sys.path.insert(0, path)

import numpy as np

import token_io
from token_io import DTYPE, encode_resilient, human, read_meta, write_meta

DEFAULT_CHUNK_CHARS = 500_000     # ~1s encode work, ~4MB peak RAM per worker


def load_tokenizer(vocab_path: str):
    """HF `tokenizer.json` -> HfBpeShim, legacy {vocab,merges} -> BPETokenizer.

    A corpus must be encoded by the SAME encoder that built its vocab. For a
    comparison against EleutherAI open weights, "same encoder" means the real HF
    Pythia tokenizer -- the legacy re-derived pythia vocab measures ~1.1% fewer
    tokens and first diverges around curly quotes/accented chars.
    """
    with open(vocab_path, encoding="utf-8") as f:
        head = json.load(f)
    if "model" in head and "pre_tokenizer" in head:
        from hf_bpe_shim import HfBpeShim
        return HfBpeShim.from_file(vocab_path), "hf"
    from model.bpe import BPETokenizer
    return BPETokenizer.load(vocab_path), "legacy"


def boundaries(src: str, n_parts: int) -> list[int]:
    """n_parts+1 byte offsets covering the file, each interior one newline-aligned.

    Reads nothing but a few KB: seeks to the approximate cut and scans forward
    to the next newline. Ranges are contiguous by construction -- range i is
    [b[i], b[i+1]) -- so concatenation in order reproduces the whole stream.
    """
    total = os.path.getsize(src)
    if n_parts < 1:
        raise SystemExit("--workers must be >= 1")
    if total < n_parts:
        raise SystemExit(f"file is {total}B, too small to split into {n_parts}")
    b = [0]
    with open(src, "rb") as f:
        for i in range(1, n_parts):
            approx = (total * i) // n_parts
            if approx <= b[-1]:
                b.append(min(b[-1] + 1, total))
                continue
            f.seek(approx)
            buf = f.readline(65536)          # lands us on a line boundary
            b.append(min(approx + len(buf), total))
    b.append(total)
    # Defensive: strictly increasing, or a part would be empty/negative.
    for i in range(1, len(b)):
        if b[i] <= b[i - 1]:
            raise SystemExit(f"degenerate boundary at index {i}: {b[i-3:i+3]}")
    return b


def encode_range(src: str, vocab: str, part_bin: str, start: int, end: int,
                 chunk_chars: int, worker_id: int) -> int:
    """Encode [start,end) of src -> part_bin. Returns exit code."""
    import codecs

    tok, _kind = load_tokenizer(vocab)
    meta_path = part_bin + ".meta.json"
    done_chars, done_tokens = 0, 0

    # Resume: trust the sidecar only if it describes this exact range.
    m = read_meta(meta_path)
    if (m.get("start") == start and m.get("end") == end
            and m.get("complete") and os.path.exists(part_bin)):
        print(f"  w{worker_id}: already complete ({m['tokens']:,} toks)", flush=True)
        return 0
    if os.path.exists(part_bin):
        m = read_meta(meta_path)
        if m.get("start") == start and m.get("end") == end and not m.get("complete"):
            done_chars, done_tokens = int(m["chars"]), int(m["tokens"])
            # Bytes past the last recorded chunk would shift every later token.
            with open(part_bin, "r+b") as f:
                f.truncate(done_tokens * DTYPE.itemsize)
            print(f"  w{worker_id}: resume at {done_chars:,} chars", flush=True)

    mode = "r+b" if done_chars else "wb"
    t0 = time.time()
    n_chunks = 0
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

    with open(src, "rb") as fin, open(part_bin, mode) as fout:
        fin.seek(start + done_chars)
        if done_chars:
            pass
        else:
            fout.seek(0)
        # Byte->str incrementally so a multi-byte char straddling a chunk
        # boundary is not corrupted into two replacement chars.
        pending = ""
        while True:
            pos = fin.tell()
            if pos >= end:
                break
            raw = fin.read(min(chunk_chars, end - pos))
            if not raw:
                break
            text = pending + decoder.decode(raw, final=(pos + len(raw) >= end))
            pending = ""
            if not text:
                continue
            ids = encode_resilient(tok, text)
            np.asarray(ids, dtype=DTYPE).tofile(fout)
            done_chars += len(raw)
            done_tokens += len(ids)
            n_chunks += 1
            if n_chunks % 20 == 0:
                dt = time.time() - t0
                print(f"  w{worker_id}: chunk {n_chunks:5d} | {done_tokens:>12,} toks"
                      f" | {(done_tokens / dt if dt else 0):,.0f} tok/s", flush=True)
            if n_chunks % 25 == 0:
                write_meta(meta_path, {"start": start, "end": end,
                                       "chars": done_chars, "tokens": done_tokens,
                                       "complete": False})

    write_meta(meta_path, {"start": start, "end": end, "chars": done_chars,
                           "tokens": done_tokens, "bytes": os.path.getsize(part_bin),
                           "complete": True})
    print(f"  w{worker_id}: DONE {done_tokens:,} tokens in "
          f"{time.time() - t0:,.0f}s", flush=True)
    return 0


def selftest(src: str, vocab: str) -> int:
    """Prove a line-aligned split reproduces the unsplit token stream."""
    tok, kind = load_tokenizer(vocab)
    with open(src, encoding="utf-8", errors="replace") as f:
        text = f.read(600_000)
    if not text:
        raise SystemExit("source too small for selftest")
    mid = len(text) // 2
    nl = text.find("\n", mid)
    if nl < 0:
        raise SystemExit("no newline to split on")
    whole = tok.encode(text)
    split = tok.encode(text[:nl + 1]) + tok.encode(text[nl + 1:])
    ok = whole == split
    print(f"selftest ({kind} encoder): split==whole -> {ok} "
          f"({len(whole):,} vs {len(split):,} tokens)", flush=True)
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src")
    ap.add_argument("vocab")
    ap.add_argument("dst")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--chunk-chars", type=int, default=DEFAULT_CHUNK_CHARS)
    ap.add_argument("--selftest", action="store_true",
                    help="verify split==whole and exit")
    args = ap.parse_args()

    if args.selftest:
        return selftest(args.src, args.vocab)

    for p in (args.src, args.vocab):
        if not os.path.exists(p):
            raise SystemExit(f"not found: {p}")

    total_bytes = os.path.getsize(args.src)
    n = args.workers
    parts_bin = args.dst + ".parts"          # scratch dir for part files
    os.makedirs(parts_bin, exist_ok=True)
    b = boundaries(args.src, n)

    print(f"src   {args.src} ({human(total_bytes)})")
    print(f"vocab {args.vocab}")
    print(f"dst   {args.dst}")
    print(f"parts {n} workers -> {parts_bin}", flush=True)
    for i in range(n):
        print(f"  w{i}: [{human(b[i])} .. {human(b[i + 1])}) "
              f"{human(b[i + 1] - b[i])}")

    t0 = time.time()
    # Separate processes on purpose: mp.Pool crashed silently on Windows.
    procs = []
    for i in range(n):
        part = os.path.join(parts_bin, f"part{i:02d}.bin")
        cmd = [sys.executable, "-u", os.path.abspath(__file__),
               args.src, args.vocab, part,
               "--start", str(b[i]), "--end", str(b[i + 1]),
               "--chunk-chars", str(args.chunk_chars)]
        procs.append((i, part, b[i], b[i + 1],
                      subprocess.Popen(cmd)))
    print(flush=True)

    failed = []
    for i, part, _s, _e, p in procs:
        rc = p.wait()
        if rc != 0:
            failed.append((i, rc))
            print(f"  !! w{i} exited rc={rc}", flush=True)

    # Every part accounted for before a single byte is merged.
    total_tokens = 0
    verified = []
    for i, part, start, end in ((i, p, s, e) for i, p, s, e, _ in procs):
        m = read_meta(part + ".meta.json")
        if not m.get("complete"):
            failed.append((i, "incomplete"))
            continue
        if m.get("start") != start or m.get("end") != end:
            failed.append((i, "range mismatch"))
            continue
        size = os.path.getsize(part) if os.path.exists(part) else -1
        if size != m.get("bytes") or size != m["tokens"] * DTYPE.itemsize:
            failed.append((i, f"size {size} != {m.get('bytes')}"))
            continue
        verified.append((i, part))
        total_tokens += m["tokens"]

    if failed:
        print(f"\nFAILED: {failed}\nNo merge performed. Parts kept in {parts_bin}",
              flush=True)
        return 1

    print(f"\nall {len(verified)} parts verified ({total_tokens:,} tokens) "
          f"in {time.time() - t0:,.0f}s -- merging", flush=True)

    # Merge as a byte stream. Never build the whole array in RAM.
    t1 = time.time()
    with open(args.dst, "wb") as out:
        for _i, part in sorted(verified):
            with open(part, "rb") as f:
                while True:
                    blk = f.read(64 * 1024 * 1024)
                    if not blk:
                        break
                    out.write(blk)

    size = os.path.getsize(args.dst)
    expect = total_tokens * DTYPE.itemsize
    if size != expect:
        print(f"\nMERGE MISMATCH: {size}B on disk != {expect}B expected",
              flush=True)
        return 1

    write_meta(args.dst + ".meta.json", {
        "src": os.path.abspath(args.src),
        "vocab": os.path.abspath(args.vocab),
        "chunk_chars": args.chunk_chars,
        "workers": n,
        "chars": total_bytes,
        "tokens": total_tokens,
        "bytes": size,
        "complete": True,
    })
    dt = time.time() - t0
    print(f"\nDONE {total_tokens:,} tokens -> {args.dst} ({human(size)}) "
          f"in {dt / 60:.1f}min ({total_tokens / dt:,.0f} tok/s)", flush=True)
    return 0


if __name__ == "__main__":
    if "--start" in sys.argv:
        p = argparse.ArgumentParser()
        p.add_argument("src"); p.add_argument("vocab"); p.add_argument("part")
        p.add_argument("--start", type=int, required=True)
        p.add_argument("--end", type=int, required=True)
        p.add_argument("--chunk-chars", type=int, default=DEFAULT_CHUNK_CHARS)
        a = p.parse_args()
        sys.exit(encode_range(a.src, a.vocab, a.part, a.start, a.end,
                              a.chunk_chars, 0))
    sys.exit(main())
