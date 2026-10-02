#!/usr/bin/env python3
"""RETIRED 2026-10-01 -- this script produced an INVALID eval fixture. Do not re-run it.

WHAT WENT WRONG
    This wrote data/incoming/pile_val_slice.txt by DECODING the held-out token ids to
    text. The scorers then re-tokenized that text. decode -> write -> re-encode is LOSSY,
    and the round trip does not reproduce the original ids. Measured against the cache:

        cache tail      20,039,921 tokens
        re-tokenized    20,039,764 tokens      (-157)
        first divergence at index 42,843
        overlapping ids identical: 1.6760%       <- 1.68%, not "exact"

    The old meta.json called this "Exact replay of split_corpus". That was a claim in a
    comment, never a measurement, and it was false.

CONSEQUENCE
    reports/pythia70m_reference.json (bpb 1.2727) was measured on the WRONG ids, so every
    "gap to pythia-70m" built on it is suspect. Fixed 2026-10-01: both scorers now read the
    ids straight from the memmapped .bin, with no tokenizer in the path at all.

THE RULE THIS FILE EXISTS TO TEACH
    Never route held-out eval data through text. Score the raw token ids. A "clean"
    human-readable fixture is worth nothing if it is not token-identical to what the
    trainer actually held out -- and there is no cheap way to be sure it is.

See: scripts/verify_val_slice_identity.py (the proof), and
     .hermes/plans/2026-10-01_eval-fixture-bug.md

The code below is kept only so the diff that produced the bad fixture stays readable.
Everything from here on is retained verbatim and unsafe.
"""
import json
import os

import numpy as np

LAB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
META = os.path.join(LAB, "data", "pile_train_full_bpe_pythia70m.bin.meta.json")
BIN = os.path.join(LAB, "data", "pile_train_full_bpe_pythia70m.bin")
TOK = os.path.join(LAB, "data", "incoming", "pythia70m_hf", "tokenizer.json")
OUT = os.path.join(LAB, "data", "incoming", "pile_val_slice.txt")
OUTMETA = os.path.join(LAB, "data", "incoming", "pile_val_slice.meta.json")
VAL_FRAC = 0.01


def main():
    meta = json.load(open(META, encoding="utf-8"))
    total = int(meta["tokens"])
    cut = int(total * (1.0 - VAL_FRAC))
    n_val = total - cut
    print(f"corpus {total:,} tokens; train [:{cut:,}]; val [{cut:,}:] = {n_val:,} tokens")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(os.path.dirname(TOK))

    # memmap the int32 cache and read ONLY the val tail (8GB file, ~80MB read).
    arr = np.memmap(BIN, dtype=np.int32, mode="r")
    val_ids = np.asarray(arr[cut:total], dtype=np.int64)
    print(f"read {val_ids.shape[0]:,} ids from the cache")

    text = tok.decode(val_ids, skip_special_tokens=True)
    with open(OUT, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    nbytes = os.path.getsize(OUT)
    print(f"wrote {OUT} ({nbytes:,} bytes, {len(text):,} chars)")

    # !! DO NOT USE THIS OUTPUT. Kept only for the historical diff; see the module docstring.
    out_meta = {
        "source_txt": "data/pile_train_full.txt",
        "token_cache": "data/pile_train_full_bpe_pythia70m.bin",
        "tokenizer": "data/incoming/pythia70m_hf/tokenizer.json",
        "val_frac": VAL_FRAC,
        "corpus_tokens": total,
        "cut_index": cut,
        "val_tokens": int(val_ids.shape[0]),
        "bytes": nbytes,
        "INVALID": True,
        "invalidated": "2026-10-01",
        "note": "THIS FIXTURE IS NOT TOKEN-IDENTICAL TO THE TRAINER'S VAL SET. "
                "1.68% of positions match the cache tail; it diverges at index 42843. "
                "Do NOT score models on this file and do NOT quote the 1.2727 reference "
                "derived from it. Read ids from the .bin directly "
                "(see score_pythia70m_reference.py --source cache).",
    }
    json.dump(out_meta, open(OUTMETA, "w", encoding="utf-8"), indent=2)
    print(f"wrote {OUTMETA}")


if __name__ == "__main__":
    main()
