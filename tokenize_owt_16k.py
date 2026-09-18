#!/usr/bin/env python3
"""Encode the 80 OpenWebText shards with the 16k BPE tokenizer, one shard at a time.

Why it's built this way
-----------------------
* **Serialized.** One shard, one process, one open file. Earlier versions either
  tried to encode the whole 40GB in one pass (Python list of 10B ints -> dies on
  a 34GB box) or loaded all 80 shard tensors into RAM at the end to torch.cat
  them (also dies). Here the output file is *appended* shard by shard, so peak
  RAM is one 500K-char chunk (~4MB).
* **On D:, never C:.** No tempfile, no %TEMP%. Work files and the output all
  live under llm-lab/data.
* **Resumable.** A manifest records which shards are done and how many tokens
  each produced. Kill it, rerun it, it skips what's finished. Partial shards
  from a crash are truncated away and redone — token streams can't have holes.

Output (defaults):
    data/openwebtext_combined_bpe_owt16k.bin            raw int32 tokens
    data/openwebtext_combined_bpe_owt16k.bin.manifest.json

Train straight off the .bin — token_io/train.py memmap it, so the corpus is
never loaded into RAM.

Usage:
    python tokenize_owt_16k.py                      # full 80-shard run (~6h)
    python tokenize_owt_16k.py --smoke              # 3 shards x 20M chars, ~40s
    python tokenize_owt_16k.py --limit-shards 10    # partial run
"""
from __future__ import annotations

import argparse
import datetime as _dt
import glob
import os
import shutil
import sys
import time

import token_io
from token_io import count_tokens, encode_file_stream, human, fmt_eta, read_meta, write_meta
from model.bpe import BPETokenizer

DATA = token_io.data_dir()
DEFAULT_SHARD_DIR = os.path.join(DATA, "openwebtext", "shards")
DEFAULT_VOCAB = os.path.join(DATA, "bpe_owt16k.json")
DEFAULT_OUT = os.path.join(DATA, "openwebtext_combined_bpe_owt16k.bin")
DEFAULT_CHUNK_CHARS = 500_000          # ~1s of encode work, ~4MB peak RAM


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Serialize OWT shards -> 16k token file")
    ap.add_argument("--shards-dir", default=DEFAULT_SHARD_DIR)
    ap.add_argument("--vocab", default=DEFAULT_VOCAB)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--chunk-chars", type=int, default=DEFAULT_CHUNK_CHARS)
    ap.add_argument("--limit-shards", type=int, default=0,
                    help="encode only the first N shards (0 = all)")
    ap.add_argument("--shard-char-limit", type=int, default=0,
                    help="encode only the first N chars of each shard (smoke)")
    ap.add_argument("--smoke", action="store_true",
                    help="short end-to-end rehearsal: 3 shards x 20M chars, own output")
    ap.add_argument("--no-probe", action="store_true", help="skip the preflight probe")
    return ap.parse_args(argv)


def shard_list(shards_dir: str) -> list[str]:
    shards = sorted(glob.glob(os.path.join(shards_dir, "*.txt")))
    if not shards:
        raise SystemExit(f"no .txt shards under {shards_dir}")
    return shards


def probe_ratio(shard_path: str, vocab_path: str, chars: int = 200_000) -> float:
    """chars-per-token on a small sample — used for preflight size and ETA."""
    tok = BPETokenizer.load(vocab_path)
    with open(shard_path, encoding="utf-8", errors="replace") as f:
        sample = f.read(chars)
    n = len(tok.encode(sample))
    return len(sample) / n if n else 4.0


def reconcile(out_bin: str, records: dict, order: list[str]) -> dict:
    """Drop manifest records that don't match the bytes actually on disk."""
    expected = sum(int(r["tokens"]) for r in records.values()) * token_io.DTYPE.itemsize
    actual = os.path.getsize(out_bin) if os.path.exists(out_bin) else 0
    if actual == expected:
        return records
    if actual > expected:
        # Bytes past the last recorded shard are a half-written shard: cut them,
        # or every later token would be shifted by the garbage prefix.
        print(f"  trimming {human(actual - expected)} of partial shard data", flush=True)
        with open(out_bin, "r+b") as f:
            f.truncate(expected)
        return records
    # File is shorter than the manifest claims — trust the file, walk the order,
    # keep only the records that fully fit.
    print(f"  manifest over-claims by {human(expected - actual)}; rebuilding record", flush=True)
    acc, keep = 0, {}
    for name in order:
        rec = records.get(name)
        if not rec:
            continue
        size = int(rec["tokens"]) * token_io.DTYPE.itemsize
        if acc + size > actual:
            break
        keep[name] = rec
        acc += size
    with open(out_bin, "r+b") as f:
        f.truncate(acc)
    return keep


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.smoke:
        args.limit_shards = args.limit_shards or 3
        args.shard_char_limit = args.shard_char_limit or 20_000_000
        args.chunk_chars = min(args.chunk_chars, 2_000_000)   # force many chunks
        if args.out == DEFAULT_OUT:
            args.out = os.path.join(DATA, "tok16k_smoke.bin")
        print("SMOKE MODE: 3 shards x 20M chars, own output file\n", flush=True)

    if not os.path.exists(args.vocab):
        raise SystemExit(f"vocab not found: {args.vocab}")
    shards = shard_list(args.shards_dir)
    if args.limit_shards:
        shards = shards[:args.limit_shards]

    total_bytes = sum(os.path.getsize(s) for s in shards)
    man_path = args.out + ".manifest.json"
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    tok = BPETokenizer.load(args.vocab)
    print(f"vocab      {os.path.basename(args.vocab)} ({len(tok.vocab):,} tokens)")
    print(f"shards     {len(shards)} files, {human(total_bytes)} of text")
    print(f"output     {args.out}")
    print(f"chunk size {args.chunk_chars:,} chars (peak RAM ~{args.chunk_chars * 8 / 1e6:.0f}MB)")

    # Preflight: what will this cost, and is there room for it?
    ratio = probe_ratio(shards[0], args.vocab) if not args.no_probe else 4.0
    est_bytes = int(total_bytes / ratio) * token_io.DTYPE.itemsize
    free = shutil.disk_usage(os.path.dirname(os.path.abspath(args.out))).free
    print(f"estimate   ~{human(total_bytes / ratio)} tokens -> ~{human(est_bytes)} "
          f"(measured {ratio:.2f} chars/token)")
    if est_bytes > free * 0.9:
        raise SystemExit(f"not enough disk: need ~{human(est_bytes)}, "
                         f"free {human(free)} on {os.path.splitdrive(args.out)[0]}")
    print(f"disk       {human(free)} free on {os.path.splitdrive(os.path.abspath(args.out))[0]} "
          f"({est_bytes / free * 100:.1f}% of it used by this run)\n")

    # Resume: reconcile the manifest against what's really on disk.
    manifest = read_meta(man_path)
    same_run = (manifest.get("vocab") == os.path.abspath(args.vocab)
                and manifest.get("chunk_chars") == args.chunk_chars
                and manifest.get("shard_char_limit") == args.shard_char_limit
                and manifest.get("dst") == os.path.abspath(args.out))
    records = dict(manifest.get("shards", {})) if same_run else {}
    if records:
        records = reconcile(args.out, records, [os.path.basename(s) for s in shards])
    else:
        if os.path.exists(args.out) and manifest:
            print("  different settings than the last run — starting the output over", flush=True)
            open(args.out, "wb").close()
    done = len(records)
    print(f"resume     {done} shard(s) done, {len(shards) - done} to go"
          f"{' (all done — nothing to do)' if done >= len(shards) else ''}\n", flush=True)

    t_start = time.time()
    shard_times: list[float] = []
    written_this_pass = 0

    def save(complete: bool) -> None:
        write_meta(man_path, {
            "vocab": os.path.abspath(args.vocab),
            "dst": os.path.abspath(args.out),
            "chunk_chars": args.chunk_chars,
            "shard_char_limit": args.shard_char_limit,
            "shards": records,
            "total_tokens": sum(int(r["tokens"]) for r in records.values()),
            "complete": complete,
            "updated": _dt.datetime.now().isoformat(timespec="seconds"),
        })

    for i, shard in enumerate(shards):
        name = os.path.basename(shard)
        if name in records:
            r = records[name]
            print(f"[{i + 1:>2}/{len(shards)}] {name} SKIP — {int(r['tokens']):,} tokens "
                  f"already written", flush=True)
            continue

        t0 = time.time()
        print(f"[{i + 1:>2}/{len(shards)}] {name} [{human(os.path.getsize(shard))}]", flush=True)
        result = encode_file_stream(
            shard, args.vocab, args.out, chunk_chars=args.chunk_chars,
            limit_chars=args.shard_char_limit, append=True, write_sidecar=False,
            log_every=max(1, int(20 * 500_000 / args.chunk_chars)), tok=tok)

        if result["tokens"] == 0:
            print(f"  WARNING: no tokens produced — leaving this shard out", flush=True)
            continue
        records[name] = {"tokens": result["tokens"], "chars": result["chars"],
                         "seconds": round(result["seconds"], 1)}
        shard_times.append(result["seconds"])
        written_this_pass += 1
        total_tokens = sum(int(r["tokens"]) for r in records.values())

        remaining = len(shards) - len(records)
        eta = (sum(shard_times) / len(shard_times)) * remaining if shard_times else 0
        print(f"  -> {result['tokens']:,} tokens in {fmt_eta(result['seconds'])} "
              f"({result['tok_per_sec']:,.0f} tok/s) | total {total_tokens:,} | "
              f"ETA {fmt_eta(eta)}" if remaining else
              f"  -> {result['tokens']:,} tokens in {fmt_eta(result['seconds'])}", flush=True)
        save(complete=False)

    all_done = len(records) >= len(shards)
    save(complete=all_done and not args.shard_char_limit)

    total_tokens = sum(int(r["tokens"]) for r in records.values())
    size = os.path.getsize(args.out)
    print(f"\n{'-' * 68}")
    print(f"tokens on disk : {total_tokens:,}")
    print(f"file           : {args.out} ({human(size)})")
    print(f"this pass      : {written_this_pass} shard(s) in {fmt_eta(time.time() - t_start)}")
    print(f"complete       : {all_done and not args.shard_char_limit}")
    print(f"verify         : python verify_tokens.py --bin \"{os.path.basename(args.out)}\" "
          f"--vocab \"{os.path.basename(args.vocab)}\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())
