#!/usr/bin/env python3
"""One place that answers "which val ids are we allowed to score on?"

WHY THIS EXISTS (2026-10-06)
    Every scorer used to hardcode `pile_train_full_bpe_pythia70m.bin` -- the
    2.004B-token cache the 70M run trained on. The 160M run trains on
    `pile_train_full_bpe_pythia_hf.bin` (3.2367B tokens, real HF Pythia
    encoder), and the OLD val tail (source chars ~8.04-8.11GB) sits INSIDE the
    160M training set. Scoring the 160M on it would be train-on-val
    contamination, so the fixture had to move -- and a fixture that lives in
    ten hardcoded constants is a fixture that silently drifts.

    The cache a run trains on is already recorded in train_job.json (written
    by the sanctioned launch gate, viz/arm.py), so that is what we read.
    Nothing else may guess.

USAGE
    from val_fixture import resolve, tail_ids, trainer_bpt
    bin_path, meta_path, meta = resolve()            # active run's cache
    ids = tail_ids(bin_path, val_frac=0.01)          # int64, exactly what
                                                    # train.split_corpus holds out
    bpt = trainer_bpt(ids, "data/incoming/pythia70m_hf/tokenizer.json")

THE RULES, each from a way a number went wrong
  1. Never route held-out ids through text (the 1.68% round-trip bug,
     c5a9933). Read the memmap.
  2. bpb must divide by the bytes-per-token of the ids actually scored, using
     the trainer's own recipe (train.val_bytes_per_token: decode the first
     200k ids, count utf-8 bytes). A hardcoded 3.9104 belongs to the 70M's
     old tail and is wrong on any other cut.
  3. split_corpus's cut is int(total * (1 - val_frac)) with no alignment --
     replicate it exactly or the "same ids" claim is a claim again.
"""
from __future__ import annotations

import json
import os

import numpy as np

LAB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FALLBACK_CACHE = os.path.join(LAB, "data", "pile_train_full_bpe_pythia_hf.bin")
ACTIVE_JOB = os.path.join(LAB, "train_job.json")
DTYPE = np.dtype("<i4")


def _cache_from_job() -> str | None:
    """The --tok_cache of the run train_job.json describes, else None.

    A COMPLETED job does not count: it describes finished work, and scoring
    against its cache would pin every future number to a run that is over.
    (train_job.json sat completed=true on the 70M job for days -- without this
    check, resolve() would have kept serving the old 2.004B fixture.)
    """
    try:
        with open(ACTIVE_JOB, encoding="utf-8") as f:
            job = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if job.get("completed") is True:
        return None
    args = job.get("train_args") or []
    for i, tok in enumerate(args):
        if tok == "--tok_cache" and i + 1 < len(args):
            p = args[i + 1]
            return p if os.path.isabs(p) else os.path.join(LAB, p.replace("/", os.sep))
    return None


def resolve(cache: str | None = None) -> tuple[str, str, dict]:
    """(bin, meta, meta_dict) for the cache to score on.

    Priority: explicit --cache arg, then train_job.json's --tok_cache, then
    the full-corpus HF cache. Raises if the chosen cache does not exist --
    falling back silently to a *different* corpus is the bug, not the fix.
    """
    path = cache or _cache_from_job() or FALLBACK_CACHE
    if not os.path.isabs(path):
        path = os.path.join(LAB, path.replace("/", os.sep))
    if not os.path.exists(path):
        raise SystemExit(
            f"val fixture cache not found: {path}\n"
            f"  (train_job.json says: {_cache_from_job()})")
    meta_path = path + ".meta.json"
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
    return path, meta_path, meta


def total_tokens(bin_path: str, meta: dict | None = None) -> int:
    """Token count of the cache. From meta when it agrees with the file, else
    from the file itself -- the file is the truth (token_io.count_tokens)."""
    on_disk = os.path.getsize(bin_path) // DTYPE.itemsize
    if meta and int(meta.get("tokens", -1)) == on_disk:
        return on_disk
    if meta and int(meta.get("tokens", -1)) != on_disk:
        raise SystemExit(
            f"{os.path.basename(bin_path)}: meta says {int(meta['tokens']):,} "
            f"tokens but the file holds {on_disk:,} -- sidecar is stale")
    return on_disk


def cut_index(total: int, val_frac: float) -> int:
    """train.split_corpus's cut, replicated: int(total * (1 - val_frac))."""
    return int(total * (1.0 - val_frac))


def tail_ids(bin_path: str, val_frac: float = 0.01,
             meta: dict | None = None) -> np.ndarray:
    """The held-out ids as int64 -- identical to what split_corpus passes on."""
    total = total_tokens(bin_path, meta)
    cut = cut_index(total, val_frac)
    arr = np.memmap(bin_path, dtype=DTYPE, mode="r")
    if len(arr) != total:
        raise SystemExit(f"{bin_path}: memmap {len(arr):,} != expected {total:,}")
    return np.asarray(arr[cut:total], dtype=np.int64)


def trainer_bpt(ids: np.ndarray, tokenizer_path: str, sample: int = 200_000) -> float:
    """bytes-per-token, replicating train.val_bytes_per_token exactly."""
    import sys
    if LAB not in sys.path:
        sys.path.insert(0, LAB)
    from train.train import load_tokenizer
    tok = load_tokenizer(tokenizer_path)
    take = ids[:sample]
    if take.size == 0:
        return 1.0
    text = tok.decode([int(i) for i in take.tolist()])
    nbytes = len(text.encode("utf-8", errors="ignore"))
    return nbytes / take.size if nbytes else 1.0


def encoder_path(meta: dict) -> str:
    """The tokenizer that produced this cache, from its sidecar (`vocab`).

    Re-derived instead of hardcoded because a cache scored with a DIFFERENT
    encoder than it was written with is a different corpus wearing the same
    filename -- that was the bpe_pythia_native trap.
    """
    p = (meta or {}).get("vocab") or os.path.join(
        LAB, "data", "incoming", "pythia70m_hf", "tokenizer.json")
    if not os.path.isabs(p):
        p = os.path.join(LAB, p.replace("/", os.sep))
    if not os.path.exists(p):
        raise SystemExit(f"encoder for this cache not found: {p}")
    return p


def describe(bin_path: str, meta: dict, val_frac: float = 0.01) -> str:
    total = total_tokens(bin_path, meta)
    cut = cut_index(total, val_frac)
    return (f"{os.path.basename(bin_path)}: total {total:,}  "
            f"train [:, {cut:,}]  val [{cut:,}:,] = {total - cut:,} ids")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--val-frac", type=float, default=0.01)
    a = ap.parse_args()
    b, _m, meta = resolve(a.cache)
    print(describe(b, meta, a.val_frac))
    print(f"  job-selected cache: {_cache_from_job() or '(none in train_job.json)'}")
    print(f"  fallback constant : {FALLBACK_CACHE}")
