#!/usr/bin/env python3
"""Encode the OpenWebText shards with the 16k BPE tokenizer.

Serial by default, parallel on request. Either way two things are guaranteed:
**no orphan processes**, and **no work thrown away**.

Design rules (each one earned by a failure)
-------------------------------------------
* One shard, one file, appended. Nothing accumulates in RAM — peak is one
  500K-char chunk (~4MB). Never `torch.cat()` 80 shard tensors again (that
  needed ~80GB on a 34GB box).
* Work files and output live under llm-lab/data on D:. No tempfile, no %TEMP%.
* A manifest records every finished shard. Kill it, rerun it, only the missing
  shards get encoded. Partial shards are truncated and redone — token streams
  can't have holes.
* `--workers N` runs N *separate processes*, each on a contiguous block of
  shards, each writing its own `<out>.partNN.bin`. Separate processes on
  purpose (`mp.Pool` crashed silently on Windows). Contiguous blocks on purpose
  (merging is then a straight concatenation, no offset bookkeeping).
* The parent records worker PIDs *and their process start times*, kills
  survivors of a crashed previous run at startup, and tears its own children
  down on any exception. The earlier parallel attempts left processes holding
  RAM — never again.
* Merge only when every shard is accounted for, then check the byte count
  against the manifest before declaring success.

Usage:
    python tokenize_owt_16k.py                        # serial, all 80 shards (~6h)
    python tokenize_owt_16k.py --workers 8            # parallel, ~50 min
    python tokenize_owt_16k.py --smoke                # 3 shards x 20M chars, ~40s
    python tokenize_owt_16k.py --smoke --workers 3    # parallel rehearsal

Outputs (defaults):
    data/openwebtext_combined_bpe_owt16k.bin                raw int32 tokens
    data/openwebtext_combined_bpe_owt16k.bin.manifest.json  per-shard record

Train straight off the .bin — train.py memmaps it, so the corpus is never
loaded into RAM.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import glob
import math
import os
import shutil
import subprocess
import sys
import time

import token_io
from token_io import encode_file_stream, human, fmt_eta, read_meta, write_meta
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
    # --- parallel layer ---
    ap.add_argument("--workers", type=int, default=1,
                    help="encode this many shard-blocks in parallel (separate processes)")
    ap.add_argument("--keep-parts", action="store_true",
                    help="keep the per-worker .partNN.bin files after merging")
    ap.add_argument("--status-every", type=int, default=30,
                    help="seconds between parent status lines")
    # --- internal: set by the parent when it spawns a worker ---
    ap.add_argument("--worker-id", type=int, default=-1)
    ap.add_argument("--shard-from", type=int, default=0)
    ap.add_argument("--shard-to", type=int, default=0)
    return ap.parse_args(argv)


def shard_list(shards_dir: str, limit: int = 0) -> list[str]:
    shards = sorted(glob.glob(os.path.join(shards_dir, "*.txt")))
    if not shards:
        raise SystemExit(f"no .txt shards under {shards_dir}")
    return shards[:limit] if limit else shards


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


def settings_key(args) -> dict:
    """Anything that changes what the bytes in a file mean."""
    return {"vocab": os.path.abspath(args.vocab),
            "chunk_chars": args.chunk_chars,
            "shard_char_limit": args.shard_char_limit}


def manifest_matches(manifest: dict, args, dst: str) -> bool:
    return (manifest.get("dst") == os.path.abspath(dst)
            and manifest.get("vocab") == os.path.abspath(args.vocab)
            and manifest.get("chunk_chars") == args.chunk_chars
            and manifest.get("shard_char_limit") == args.shard_char_limit)


# --------------------------------------------------------------------------
# The encode loop — used by a serial run and by each parallel worker.
# --------------------------------------------------------------------------

def run_range(shards: list[str], out_bin: str, args, tok) -> dict:
    """Encode one contiguous block of shards into out_bin, resuming as needed."""
    man_path = out_bin + ".manifest.json"
    manifest = read_meta(man_path)
    records = dict(manifest.get("shards", {})) if manifest_matches(manifest, args, out_bin) else {}

    if not records and os.path.exists(out_bin) and manifest:
        open(out_bin, "wb").close()      # settings changed: the old bytes mean nothing
    if records:
        records = reconcile(out_bin, records, [os.path.basename(s) for s in shards])

    def save(complete: bool) -> None:
        write_meta(man_path, {
            **settings_key(args),
            "dst": os.path.abspath(out_bin),
            "shards": records,
            "total_tokens": sum(int(r["tokens"]) for r in records.values()),
            "complete": complete,
            "updated": _dt.datetime.now().isoformat(timespec="seconds"),
        })

    save(complete=False)
    print(f"resume     {len(records)} shard(s) done, {len(shards) - len(records)} to go",
          flush=True)

    shard_times: list[float] = []
    for i, shard in enumerate(shards):
        name = os.path.basename(shard)
        if name in records:
            print(f"[{i + 1:>2}/{len(shards)}] {name} SKIP — "
                  f"{int(records[name]['tokens']):,} tokens already written", flush=True)
            continue

        print(f"[{i + 1:>2}/{len(shards)}] {name} [{human(os.path.getsize(shard))}]", flush=True)
        result = encode_file_stream(
            shard, args.vocab, out_bin, chunk_chars=args.chunk_chars,
            limit_chars=args.shard_char_limit, append=True, write_sidecar=False,
            log_every=max(1, int(20 * 500_000 / args.chunk_chars)), tok=tok)

        if result["tokens"] == 0:
            print("  WARNING: no tokens produced — leaving this shard out", flush=True)
            continue
        if result.get("encode_failures"):
            print(f"  NOTE: encoder recovered from {result['encode_failures']} failure(s) "
                  f"— see {os.path.basename(out_bin)}.encode_errors.jsonl", flush=True)
        records[name] = {"tokens": result["tokens"], "chars": result["chars"],
                         "seconds": round(result["seconds"], 1)}
        shard_times.append(result["seconds"])
        total = sum(int(r["tokens"]) for r in records.values())
        remaining = len(shards) - len(records)
        line = (f"  -> {result['tokens']:,} tokens in {fmt_eta(result['seconds'])} "
                f"({result['tok_per_sec']:,.0f} tok/s) | total {total:,}")
        if remaining and shard_times:
            line += f" | ETA {fmt_eta(sum(shard_times) / len(shard_times) * remaining)}"
        print(line, flush=True)
        save(complete=False)

    save(complete=len(records) >= len(shards) and not args.shard_char_limit)
    return {"records": records, "tokens": sum(int(r["tokens"]) for r in records.values())}


# --------------------------------------------------------------------------
# Orphan control — the thing that bit the earlier parallel attempts.
# --------------------------------------------------------------------------

def workers_file(out_bin: str) -> str:
    return out_bin + ".workers.json"


def tail_lines(path: str, n: int = 3) -> str:
    """Last few non-blank lines of a worker log — for the 'it died' message."""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = [ln.rstrip() for ln in f if ln.strip()]
        return " | ".join(lines[-n:])[-400:]
    except OSError:
        return "(log unreadable)"


def reap_orphans(out_bin: str) -> int:
    """Kill workers left behind by a previous crashed run on this same output."""
    import psutil
    info = read_meta(workers_file(out_bin))
    killed = 0
    for rec in info.get("workers", []):
        pid, started = rec.get("pid"), rec.get("started")
        try:
            p = psutil.Process(pid)
            # PID-reuse guard: only kill it if it's the process we actually spawned.
            if started and abs(p.create_time() - started) > 1.0:
                continue
            print(f"  reaping orphan worker pid {pid} (block {rec.get('block')})", flush=True)
            p.kill()
            killed += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    if os.path.exists(workers_file(out_bin)):
        os.unlink(workers_file(out_bin))
    return killed


def record_workers(out_bin: str, procs: list[dict]) -> None:
    write_meta(workers_file(out_bin), {
        "updated": _dt.datetime.now().isoformat(timespec="seconds"),
        "workers": [{"pid": p["proc"].pid, "block": p["block"],
                     "started": p["started"], "part": p["part"]} for p in procs],
    })


# --------------------------------------------------------------------------
# Parallel parent
# --------------------------------------------------------------------------

def split_blocks(shards: list[str], workers: int) -> list[list[str]]:
    """Contiguous, near-equal blocks — mergeable by concatenation, no offsets."""
    workers = max(1, min(workers, len(shards)))
    base, extra = divmod(len(shards), workers)
    blocks, start = [], 0
    for i in range(workers):
        size = base + (1 if i < extra else 0)
        if size:
            blocks.append(shards[start:start + size])
        start += size
    return blocks


def scan_parts(out_bin: str, shards: list[str]) -> tuple[list[dict], list[str]]:
    """Parts already on disk, validated, in corpus order.

    A rerun must NEVER trust that block boundaries come out the same as last
    time: a different --workers value reshuffles them, and appending a shard run
    to a part that already holds *different* shards would put the corpus out of
    order while looking perfectly healthy. So the plan is derived from what the
    parts already contain, and anything inconsistent is set aside untouched.
    """
    order = {os.path.basename(s): i for i, s in enumerate(shards)}
    parts: list[dict] = []
    covered: set[str] = set()

    for path in sorted(glob.glob(out_bin + ".part*.bin")):
        man = read_meta(path + ".manifest.json")
        recs = dict(man.get("shards", {}))
        if not recs:
            continue                       # empty part: harmless, reusable
        recs = reconcile(path, recs, list(recs))     # trim a half-written shard
        names = list(recs)
        idxs = [order.get(n) for n in names]
        if any(i is None for i in idxs) or idxs != list(range(idxs[0], idxs[0] + len(idxs))):
            print(f"  ignoring {os.path.basename(path)}: shards are not a contiguous "
                  f"run of the corpus (old plan?) — left untouched", flush=True)
            continue
        clash = covered.intersection(names)
        if clash:
            print(f"  ignoring {os.path.basename(path)}: overlaps {len(clash)} shard(s) "
                  f"another part already holds", flush=True)
            continue
        covered.update(names)
        parts.append({"part": path, "log": path + ".log", "shards": names,
                      "tokens": sum(int(r["tokens"]) for r in recs.values()),
                      "first": idxs[0], "block": f"{idxs[0]}-{idxs[-1]}",
                      "stalls": 0, "reported": False})

    parts.sort(key=lambda p: p["first"])              # merge order = corpus order
    gaps = [os.path.basename(s) for s in shards if os.path.basename(s) not in covered]
    return parts, gaps


def contiguous_runs(names: list[str], shards: list[str]) -> list[list[str]]:
    """Split a set of missing shard names into runs that are adjacent in the corpus."""
    order = {os.path.basename(s): i for i, s in enumerate(shards)}
    runs: list[list[str]] = []
    for name in sorted(names, key=lambda n: order[n]):
        if runs and order[name] == order[runs[-1][-1]] + 1:
            runs[-1].append(name)
        else:
            runs.append([name])
    return runs


def chunk_run(run: list[str], size: int) -> list[list[str]]:
    """Cut one contiguous run into pieces — each piece stays contiguous, so it
    can still be merged by concatenation."""
    return [run[i:i + size] for i in range(0, len(run), size)]


def adopt_existing_split(out_bin: str, shards: list[str], args) -> int:
    """Keep the shards a previous *serial* run already encoded.

    A serial file is written in shard order, so its contents are always a
    contiguous prefix — exactly what part00 has to be for the merge to line up.
    """
    if not os.path.exists(out_bin):
        return 0
    manifest = read_meta(out_bin + ".manifest.json")
    if not manifest_matches(manifest, args, out_bin):
        return 0
    order = [os.path.basename(s) for s in shards]
    records = reconcile(out_bin, dict(manifest.get("shards", {})), order)
    if not records:
        return 0

    names = set(records)
    prefix = []
    for name in order:
        if name in names:
            prefix.append(name)
        else:
            break
    if len(prefix) != len(records):
        print(f"  existing output holds {len(records)} shard(s) but not as a clean "
              f"prefix — leaving it alone", flush=True)
        return 0

    part = out_bin + ".part00.bin"
    if os.path.exists(part):
        os.unlink(part)                  # stale part from an aborted attempt
    os.replace(out_bin, part)
    os.unlink(out_bin + ".manifest.json")
    adopted = {**settings_key(args), "dst": os.path.abspath(part), "shards": records,
               "total_tokens": sum(int(r["tokens"]) for r in records.values()),
               "complete": False, "adopted": True,
               "updated": _dt.datetime.now().isoformat(timespec="seconds")}
    write_meta(part + ".manifest.json", adopted)
    print(f"  adopted existing output as part00: {len(prefix)} shard(s), "
          f"{adopted['total_tokens']:,} tokens carried forward (nothing wasted)",
          flush=True)
    return len(prefix)


def spawn_worker(block: list[str], index: int, offset: int, out_bin: str, args) -> dict:
    import psutil
    part = f"{out_bin}.part{index:02d}.bin"
    log = part + ".log"
    cmd = [sys.executable, "-u", os.path.abspath(__file__),
           "--vocab", args.vocab, "--shards-dir", args.shards_dir,
           "--out", part, "--chunk-chars", str(args.chunk_chars),
           "--shard-char-limit", str(args.shard_char_limit),
           "--shard-from", str(offset), "--shard-to", str(offset + len(block)),
           "--worker-id", str(index)]
    if args.no_probe:
        cmd.append("--no-probe")
    fh = open(log, "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT)
    started = psutil.Process(proc.pid).create_time()
    print(f"  worker {index}: {len(block)} shard(s) "
          f"[{os.path.basename(block[0])} .. {os.path.basename(block[-1])}] "
          f"pid {proc.pid} -> {os.path.basename(part)}", flush=True)
    return {"proc": proc, "started": started, "part": part, "log": log,
            "block": f"{offset}-{offset + len(block) - 1}", "shards": block,
            "tokens": 0, "stalls": 0}


def teardown(procs: list[dict], out_bin: str) -> None:
    """No orphans, ever: anything we spawned dies with us."""
    for p in procs:
        if p["proc"].poll() is None:
            p["proc"].terminate()
    for p in procs:
        try:
            p["proc"].wait(timeout=20)
        except subprocess.TimeoutExpired:
            p["proc"].kill()
            p["proc"].wait(timeout=10)
    record_workers(out_bin, procs)


def supervise(procs: list[dict], done_tokens: int, out_bin: str, args) -> None:
    """Watch a batch of workers: status lines, stall warnings, death notices.

    Teardown on every exit path is the "no orphans" guarantee, so a killed or
    crashed parent cannot leave workers running.
    """
    t0 = time.time()
    prev_total, prev_t = done_tokens, t0
    try:
        while True:
            time.sleep(args.status_every)
            alive = 0
            for p in procs:
                # Bytes on disk grow continuously; the manifest only updates when a
                # shard finishes. Prefer file size so status shows real progress.
                live = (os.path.getsize(p["part"]) // token_io.DTYPE.itemsize
                        if os.path.exists(p["part"]) else 0)
                man = read_meta(p["part"] + ".manifest.json")
                now_tokens = max(live, int(man.get("total_tokens", 0) or 0))
                running = p["proc"].poll() is None
                alive += 1 if running else 0
                # A dead worker is news — say so now, with its last log lines,
                # instead of leaving it to be spotted in an "[7/8 workers]" line.
                if not running and not p.get("reported"):
                    p["reported"] = True
                    print(f"  !! WORKER BLOCK {p['block']} EXITED with code "
                          f"{p['proc'].returncode} — log: {p['log']}\n"
                          f"     {tail_lines(p['log'], 3)}", flush=True)
                if running:
                    p["stalls"] = p["stalls"] + 1 if now_tokens == p["tokens"] else 0
                p["tokens"] = now_tokens
            total = done_tokens + sum(p["tokens"] for p in procs)
            now = time.time()
            rate = (total - prev_total) / max(1e-6, now - prev_t)
            prev_total, prev_t = total, now
            print(f"[{alive}/{len(procs)} workers] {total:,} tokens | "
                  f"{(now - t0) / 60:.1f} min | ~{rate:,.0f} tok/s", flush=True)
            for p in procs:
                if p["stalls"] == 3:
                    print(f"  WARNING: block {p['block']} has not finished a shard in "
                          f"~{3 * args.status_every}s — see {p['log']}", flush=True)
            if alive == 0:
                break
    except BaseException:
        print("\n  interrupted — stopping workers (finished shards are kept)", flush=True)
        teardown(procs, out_bin)
        raise


def run_parallel(shards: list[str], out_bin: str, args, tok) -> int:
    reaped = reap_orphans(out_bin)
    if reaped:
        print(f"  {reaped} orphan(s) from a previous run cleaned up before starting",
              flush=True)

    # Fold in a serial output first (its bytes are always a clean prefix), then
    # let the parts on disk define the plan. Never recompute block boundaries
    # over existing parts: that is how shards end up merged out of order.
    adopt_existing_split(out_bin, shards, args)
    parts, gaps = scan_parts(out_bin, shards)
    print(f"parallel   {len(parts)} part(s) already on disk carrying "
          f"{sum(p['tokens'] for p in parts):,} tokens; "
          f"{len(gaps)} shard(s) outstanding", flush=True)

    if not gaps:
        print("  every shard is accounted for — merging only", flush=True)
        return merge_parts(parts, out_bin, shards, args)

    # Only contiguous stretches of missing shards get handed out: a part whose
    # shards aren't adjacent in the corpus could not be merged by concatenation.
    runs = contiguous_runs(gaps, shards)
    per_worker = max(1, math.ceil(sum(len(r) for r in runs) / max(1, args.workers)))
    jobs = [piece for run in runs for piece in chunk_run(run, per_worker)]
    print(f"           {len(runs)} gap run(s) -> {len(jobs)} worker job(s) of "
          f"~{per_worker} shard(s)", flush=True)

    # Continue the part numbering rather than reusing a part that holds other
    # shards — appending to one of those would silently reorder the corpus.
    used = [int(os.path.basename(p["part"]).split(".part")[1].split(".")[0]) for p in parts]
    next_index = max(used, default=-1) + 1

    while jobs:
        batch, jobs = jobs[:args.workers], jobs[args.workers:]
        procs = []
        for job in batch:
            offset = next(i for i, s in enumerate(shards)
                          if os.path.basename(s) == job[0])
            procs.append(spawn_worker(job, next_index, offset, out_bin, args))
            next_index += 1
        record_workers(out_bin, procs)
        supervise(procs, sum(p["tokens"] for p in parts), out_bin, args)
        for p in procs:
            if p["proc"].returncode != 0:
                print(f"  worker for shard(s) {p['block']} failed "
                      f"(code {p['proc'].returncode}) — see {p['log']}", flush=True)
        parts, gaps = scan_parts(out_bin, shards)

    if gaps:
        print(f"  {len(gaps)} shard(s) still missing: {', '.join(gaps[:5])}"
              f"{' ...' if len(gaps) > 5 else ''}", flush=True)
        print("  parts kept — rerun the same command to finish them, nothing is lost",
              flush=True)
        return 1

    return merge_parts(parts, out_bin, shards, args)


def merge_parts(parts: list[dict], out_bin: str, shards: list[str], args) -> int:
    """Concatenate parts in shard order, verify the byte count, then clean up."""
    print(f"\nmerging {len(parts)} part(s) -> {out_bin}", flush=True)
    tmp = out_bin + ".merging"
    total_bytes = 0
    with open(tmp, "wb") as out:
        for p in parts:
            size = os.path.getsize(p["part"])
            with open(p["part"], "rb") as src:
                shutil.copyfileobj(src, out, length=8 << 20)
            total_bytes += size
            print(f"  + {os.path.basename(p['part'])} {human(size)}", flush=True)

    merged = {}
    for name in [os.path.basename(s) for s in shards]:
        for p in parts:
            rec = read_meta(p["part"] + ".manifest.json").get("shards", {}).get(name)
            if rec:
                merged[name] = rec
                break
    total_tokens = sum(int(r["tokens"]) for r in merged.values())

    # Last line of defence: the bytes on disk must match the manifest's math.
    if total_bytes != total_tokens * token_io.DTYPE.itemsize:
        print(f"  MISMATCH: {total_bytes:,} bytes vs {total_tokens:,} tokens "
              f"({total_tokens * 4:,} expected) — keeping {tmp} for inspection", flush=True)
        return 1
    os.replace(tmp, out_bin)

    write_meta(out_bin + ".manifest.json", {
        **settings_key(args), "dst": os.path.abspath(out_bin), "shards": merged,
        "total_tokens": total_tokens,
        "complete": len(merged) >= len(shards) and not args.shard_char_limit,
        "merged_from": [os.path.basename(p["part"]) for p in parts],
        "updated": _dt.datetime.now().isoformat(timespec="seconds"),
    })

    if not args.keep_parts:
        for p in parts:
            for f in (p["part"], p["part"] + ".manifest.json", p.get("log") or ""):
                if f and os.path.exists(f):
                    os.unlink(f)
        print("  parts cleaned up (--keep-parts keeps them)", flush=True)
    if os.path.exists(workers_file(out_bin)):
        os.unlink(workers_file(out_bin))

    print(f"\n{'-' * 68}")
    print(f"tokens on disk : {total_tokens:,}")
    print(f"file           : {out_bin} ({human(total_bytes)})")
    print(f"complete       : {len(merged) >= len(shards) and not args.shard_char_limit}")
    print(f"verify         : python verify_tokens.py --bin \"{os.path.basename(out_bin)}\" "
          f"--vocab \"{os.path.basename(args.vocab)}\" --manifest")
    return 0


# --------------------------------------------------------------------------

def main(argv=None) -> int:
    args = parse_args(argv)
    is_worker = args.worker_id >= 0

    if args.smoke and not is_worker:
        args.limit_shards = args.limit_shards or 3
        args.shard_char_limit = args.shard_char_limit or 20_000_000
        args.chunk_chars = min(args.chunk_chars, 2_000_000)   # force many chunks
        if args.out == DEFAULT_OUT:
            args.out = os.path.join(DATA, "tok16k_smoke.bin")
        print("SMOKE MODE: 3 shards x 20M chars, own output file\n", flush=True)

    if not os.path.exists(args.vocab):
        raise SystemExit(f"vocab not found: {args.vocab}")
    all_shards = shard_list(args.shards_dir, 0 if is_worker else args.limit_shards)

    if is_worker:
        # A worker only ever touches its own block — and never spawns anything.
        shards = all_shards[args.shard_from:args.shard_to]
        print(f"worker {args.worker_id}: shards {args.shard_from}-{args.shard_to - 1} "
              f"({len(shards)} of {len(all_shards)}) -> {os.path.basename(args.out)}",
              flush=True)
        res = run_range(shards, args.out, args, BPETokenizer.load(args.vocab))
        print(f"worker {args.worker_id} finished: {res['tokens']:,} tokens", flush=True)
        return 0

    total_bytes = sum(os.path.getsize(s) for s in all_shards)
    print(f"vocab      {os.path.basename(args.vocab)} "
          f"({len(BPETokenizer.load(args.vocab).vocab):,} tokens)")
    print(f"shards     {len(all_shards)} files, {human(total_bytes)} of text")
    print(f"output     {args.out}")
    print(f"chunk size {args.chunk_chars:,} chars (peak RAM per worker "
          f"~{args.chunk_chars * 8 / 1e6:.0f}MB)")

    # Preflight: what will this cost, and is there room for it?
    ratio = probe_ratio(all_shards[0], args.vocab) if not args.no_probe else 4.0
    est_tokens = total_bytes / ratio
    est_bytes = int(est_tokens) * token_io.DTYPE.itemsize
    free = shutil.disk_usage(os.path.dirname(os.path.abspath(args.out))).free
    print(f"estimate   ~{est_tokens / 1e9:.2f}B tokens -> ~{human(est_bytes)} "
          f"(measured {ratio:.2f} chars/token)")
    if est_bytes > free * 0.9:
        raise SystemExit(f"not enough disk: need ~{human(est_bytes)}, free {human(free)}")
    print(f"disk       {human(free)} free ({est_bytes / free * 100:.1f}% used by this run)")

    tok = BPETokenizer.load(args.vocab)
    workers = max(1, args.workers)
    if workers > 1 and len(all_shards) > 1:
        serial_h = est_tokens / 428_000 / 3600
        print(f"workload   ~{serial_h:.1f}h on one core -> "
              f"~{serial_h / min(workers, len(all_shards)):.1f}h with "
              f"{min(workers, len(all_shards))} workers\n", flush=True)
        return run_parallel(all_shards, args.out, args, tok)

    started = time.time()
    res = run_range(all_shards, args.out, args, tok)
    print(f"\n{'-' * 68}")
    print(f"tokens on disk : {res['tokens']:,}")
    print(f"file           : {args.out} ({human(os.path.getsize(args.out))})")
    print(f"this pass      : {fmt_eta(time.time() - started)}")
    print(f"verify         : python verify_tokens.py --bin \"{os.path.basename(args.out)}\" "
          f"--vocab \"{os.path.basename(args.vocab)}\" --manifest")
    return 0


if __name__ == "__main__":
    sys.exit(main())
