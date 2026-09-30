#!/usr/bin/env python3
"""Extract the EXACT held-out val slice our Pile runs were scored on, as raw text.

WHY THIS FILE EXISTS
    The open-weights comparison is only valid if both models are scored on the
    SAME bytes with the SAME encoder. Our trainer's val set is the last 1% of
    the token cache by token index (`split_corpus`: cut = int(len*0.99)), i.e.
    the tail of data/pile_train_full.txt. The existing pile_deduped_slice.txt
    is a DIFFERENT file, so scoring pythia-70m against it would compare two
    models on two different texts and the "gap" would be meaningless.

    So: replay the tokenizer over the tail of the corpus, take exactly the
    tokens past the cut index, and decode them back to text. That text is what
    both models must be scored on.

    Tokens are not byte-aligned, so decoding the tail can land mid-sequence at
    the edges. That is fine and expected: it is the same val text our runs saw,
    so it is the correct denominator. Do NOT try to make it prettier.

Usage:  python scripts/extract_val_slice.py
Writes: data/incoming/pile_val_slice.txt  + a .meta.json recording provenance
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

    out_meta = {
        "source_txt": "data/pile_train_full.txt",
        "token_cache": "data/pile_train_full_bpe_pythia70m.bin",
        "tokenizer": "data/incoming/pythia70m_hf/tokenizer.json",
        "val_frac": VAL_FRAC,
        "corpus_tokens": total,
        "cut_index": cut,
        "val_tokens": int(val_ids.shape[0]),
        "bytes": nbytes,
        "note": "Exact replay of split_corpus(corpus, 0.01, ctx): the last 1% of "
                "the token cache. Score EVERY model on this file to compare.",
    }
    json.dump(out_meta, open(OUTMETA, "w", encoding="utf-8"), indent=2)
    print(f"wrote {OUTMETA}")


if __name__ == "__main__":
    main()
