#!/usr/bin/env python3
"""Is the fixture we score on the ids the trainer actually holds out?

Three claims get checked here, in order of how often each one has been wrong:

  1. val_fixture's tail == train.split_corpus's val slice. Both are supposed to
     be `corpus[int(total*(1-val_frac)):]`. This runs the trainer's OWN
     function on the same memmap and compares the arrays -- if they differ,
     every "same ids" claim in every scorer is a comment again.
  2. The cache's sidecar agrees with the file (tokens == bytes/4), and the
     encoder of record is the one that wrote the cache. A stale sidecar
     silently shifts the cut; a mismatched encoder redefines every id.
  3. data/incoming/pile_val_slice.txt -- the retired text fixture -- is
     reported: which cache it came from, and that nobody may score on it.

History (why this file exists): the original version measured the
decode->write->re-encode round trip instead of trusting the meta's claim of an
"exact replay", and found 1.68% id identity with divergence at index 42,843 --
the bug behind the invalid 1.2727 reference. extract_val_slice.py is retired;
do not re-run it.

Run: .venv/Scripts/python.exe scripts/verify_val_slice_identity.py
"""
import json
import os
import sys

import numpy as np
import torch

LAB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(LAB, "scripts") not in sys.path:
    sys.path.insert(0, os.path.join(LAB, "scripts"))
import val_fixture  # noqa: E402

SLICE = os.path.join(LAB, "data", "incoming", "pile_val_slice.txt")
VAL_FRAC = 0.01
ok = True


def check(label: str, passed: bool, detail: str = "") -> None:
    global ok
    ok = ok and passed
    print(f"  [{'PASS' if passed else 'FAIL'}] {label}"
          + (f"  {detail}" if detail else ""))


def main() -> int:
    from train.train import split_corpus

    cache_bin, _meta_path, meta = val_fixture.resolve()
    print(f"=== active fixture: {os.path.basename(cache_bin)}")
    print(f"  {val_fixture.describe(cache_bin, meta, VAL_FRAC)}")

    print("\n=== 1. val_fixture tail == train.split_corpus val ===")
    total = val_fixture.total_tokens(cache_bin, meta)
    cut = val_fixture.cut_index(total, VAL_FRAC)
    # int32 view over the memmap: NO copy, and no 25.9GB int64 cast of a
    # 3.24B-token cache (that is how a verification script OOMs a 32GB box).
    memmap = np.memmap(cache_bin, dtype=val_fixture.DTYPE, mode="r")
    tensor = torch.from_numpy(memmap)
    train_part, val_part = split_corpus(tensor, VAL_FRAC, ctx=512)
    fixture_ids = val_fixture.tail_ids(cache_bin, VAL_FRAC, meta)
    check("split_corpus val shape",
          val_part is not None and val_part.shape[0] == fixture_ids.shape[0],
          f"{0 if val_part is None else val_part.shape[0]:,} "
          f"vs {fixture_ids.shape[0]:,}")
    if val_part is not None:
        # cast only the val slice (32M ids -> 256MB), never the whole cache
        val_np = np.asarray(val_part.numpy(), dtype=np.int64)
        check("ids identical elementwise",
              bool(np.array_equal(val_np, fixture_ids)), f"cut {cut:,}")
        check("train + val == total",
              len(train_part) + len(val_part) == total,
              f"{len(train_part):,} + {len(val_part):,} == {total:,}")

    print("\n=== 2. sidecar + encoder of record ===")
    on_disk = os.path.getsize(cache_bin) // val_fixture.DTYPE.itemsize
    check("meta tokens == file bytes/4",
          int(meta.get("tokens", -1)) == on_disk,
          f"meta {int(meta.get('tokens', 0)):,} vs file {on_disk:,}")
    check("meta complete flag", meta.get("complete") is True,
          str(meta.get("complete")))
    enc = val_fixture.encoder_path(meta).replace("\\", "/")
    check("encoder is the real HF Pythia tokenizer", "pythia70m_hf" in enc, enc)

    print("\n=== 3. retired text fixture (reported, never scored) ===")
    if not os.path.exists(SLICE):
        print("  pile_val_slice.txt absent -- nothing to retire")
    else:
        # sidecar is named after the stem: pile_val_slice.meta.json, not
        # pile_val_slice.txt.meta.json (extract_val_slice wrote it that way)
        stem = SLICE[:-4] if SLICE.endswith(".txt") else SLICE
        m = json.load(open(stem + ".meta.json", encoding="utf-8"))
        old_cache = str(m.get("token_cache", "?"))
        print(f"  built from {old_cache} cut {int(m.get('cut_index', 0)):,} "
              f"({int(m.get('val_tokens', 0)):,} ids, "
              f"{os.path.getsize(SLICE):,} bytes)")
        print(f"  active fixture is {os.path.basename(cache_bin)} cut {cut:,} "
              f"({fixture_ids.shape[0]:,} ids)")
        if old_cache == os.path.basename(cache_bin):
            print("  note: same cache, but the .txt is still a lossy round trip")
        else:
            print("  different region entirely -- and that region now sits "
                  "inside the 160M training set")
        check("its meta carries a retired/INVALID warning",
              any(k in json.dumps(m).upper()
                  for k in ("INVALID", "RETIRED", "DO NOT")),
              str(m.get("note", ""))[:70])

    if ok:
        print("\nALL CHECKS PASSED -- the scored ids are the trainer's ids.")
        return 0
    print("\nCHECKS FAILED -- do not quote numbers off this fixture.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
