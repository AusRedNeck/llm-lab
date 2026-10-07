#!/usr/bin/env python3
"""Score an EleutherAI Pythia checkpoint on OUR held-out Pile val tail (the away game).

WHY THIS EXISTS
    The full-diet Pile arms were launched to measure the open-weights gap, but
    pythia-70m weights had never been downloaded and eval/pile_eval.py had
    never been run, so the comparison the runs exist to produce did not exist.
    This produces the reference number. It is now parametrised so the same
    harness produces the pythia-160m reference too.

    VALIDITY RULES ENFORCED HERE
    - Same IDS as our runs: the val tail of the token cache itself, read from
      the memmap by scripts/val_fixture.py -- never decoded to text. The tail
      is the ACTIVE run's cache (train_job.json --tok_cache), because the old
      2.004B cache's tail sits inside the 160M training set.
    - Same ENCODER: whichever tokenizer wrote that cache (read from its
      sidecar `vocab`), so ids mean what they meant when they were trained on.
    - Same bpb convention: bpb = mean NLL * log2(e) / bytes-per-token, with
      bytes-per-token recomputed for the scored tail by the trainer's own
      recipe (train.val_bytes_per_token). The old hardcoded 3.9104 belonged to
      the OLD tail and is wrong here -- the new tail measures 3.8743.
    - Fixed seed, deterministic sampling, no training-loop contamination.

Usage:
    python scripts/score_pythia70m_reference.py                       # pythia-70m main
    python scripts/score_pythia70m_reference.py \
        --weights data/incoming/pythia160m_step1000 --name EleutherAI/pythia-160m \
        --out reports/pythia160m_step1000_reference.json
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch

LAB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(LAB, "scripts") not in sys.path:
    sys.path.insert(0, os.path.join(LAB, "scripts"))
import val_fixture  # noqa: E402

SLICE = os.path.join(LAB, "data", "incoming", "pile_val_slice.txt")
LEGACY_BPT = 3.9104      # the OLD tail's bytes/token; only for --source slice


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=os.path.join(LAB, "data", "incoming",
                                                      "pythia70m_weights"),
                    help="HF snapshot dir to score (default: pythia-70m main)")
    ap.add_argument("--name", default="EleutherAI/pythia-70m",
                    help="model label written into the report")
    ap.add_argument("--out", default=os.path.join(LAB, "reports",
                                                  "pythia70m_reference.json"))
    ap.add_argument("--cache", default=None,
                    help="token cache to take the val tail from "
                         "(default: val_fixture.resolve = the active run's cache)")
    ap.add_argument("--batches", type=int, default=60)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--source", choices=["cache", "slice"], default="cache",
                    help="cache = score the TOKEN CACHE tail directly (correct, and "
                         "what train.py's split_corpus actually holds); slice = score "
                         "pile_val_slice.txt, a decode->write->re-encode round trip "
                         "that is NOT token-identical to any cache tail (1.68% match). "
                         "'slice' exists only to reproduce the old, invalid number.")
    ap.add_argument("--val-frac", type=float, default=0.01)
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(a.weights, torch_dtype=torch.float32)
    model.to(dev).eval()

    cache_bin = cache_desc = None
    if a.source == "cache":
        # Score the ids the trainer held out. Read straight from the memmap:
        # going through decoded text silently changes the token stream and makes
        # the comparison meaningless (see --source help).
        cache_bin, _meta_path, meta = val_fixture.resolve(a.cache)
        ids = torch.from_numpy(val_fixture.tail_ids(cache_bin, a.val_frac, meta))
        bpt = val_fixture.trainer_bpt(
            ids.numpy(), val_fixture.encoder_path(meta))
        cut = val_fixture.cut_index(
            val_fixture.total_tokens(cache_bin, meta), a.val_frac)
        cache_desc = (f"{os.path.basename(cache_bin)} tail [{cut:,}:] "
                      f"({val_fixture.describe(cache_bin, meta, a.val_frac)})")
        print(f"cache tail: {ids.shape[0]:,} tokens  bpt {bpt:.4f}  device {dev}")
        print(f"  {cache_desc}")
    else:
        tok = AutoTokenizer.from_pretrained(os.path.dirname(
            val_fixture.encoder_path({})))
        raw = open(SLICE, "rb").read().decode("utf-8", errors="replace")
        print(f"slice: {len(raw):,} chars  <-- LOSSY: re-tokenized text, "
              f"NOT a cache tail")
        ids = tok(raw, return_tensors="pt").input_ids[0]
        bpt = LEGACY_BPT
        print(f"tokenized: {ids.shape[0]:,} tokens (vocab {tok.vocab_size})")

    n = ids.shape[0]
    need = a.batches * a.batch * a.ctx
    if n < need:
        print(f"WARNING: only {n:,} tokens for {need:,} needed; reducing batches")
        a.batches = max(1, n // (a.batch * a.ctx))

    g = torch.Generator().manual_seed(a.seed)
    tot_nll, tot_tok = 0.0, 0
    for bi in range(a.batches):
        # NON-OVERLAPPING windows. Overlapping random starts double-count the
        # same text and make the estimate look better than a clean held-out
        # sweep; stride the starts so each scored token is distinct.
        starts = torch.arange(a.batch, dtype=torch.long) * a.ctx + \
            int(torch.randint(0, max(1, n - (a.batches * a.batch * a.ctx)),
                              (1,), generator=g))
        starts = starts.clamp(max=max(0, n - a.ctx - 1))
        batch = torch.stack([ids[s:s + a.ctx] for s in starts]).to(dev)
        # Do NOT slice here. Passing labels= makes HF shift internally
        # (logits[..., :-1, :] vs labels[..., 1:]); slicing ourselves as well
        # double-shifts, so every token is scored against its own predecessor
        # and the loss reads ~9.1 instead of ~3.8. That bug produced a
        # nonsense bpb of 45.6 on the first run of this script.
        with torch.no_grad():
            out = model(batch, labels=batch)
        nll_sum = out.loss.item() * (batch.shape[0] * a.ctx)
        tot_nll += nll_sum
        tot_tok += batch.shape[0] * a.ctx
        if bi % 10 == 0:
            print(f"  batch {bi+1}/{a.batches}  running nll/token {tot_nll/tot_tok:.4f}")

    nll_per_tok = tot_nll / tot_tok
    # bpb must use the SAME formula the trainer uses, or the gap is arithmetic
    # noise. train.py: bpb = nll_nats * log2(e) / bytes_per_token.
    # The log2(e) factor is not optional: without it every number comes out
    # ~1.44x too large (nats vs bits) and reads 13.49 instead of ~9.4.
    bpb = nll_per_tok * math.log2(math.e) / bpt
    res = {
        "model": a.name,
        "weights": a.weights,
        "source": a.source,
        "source_note": ("val tail of the ACTIVE run's token cache; ids identical "
                        "to train.py split_corpus, bytes-per-token measured on "
                        "those same ids"
                        if a.source == "cache" else
                        "pile_val_slice.txt - LOSSY decode/encode round trip, NOT "
                        "token-identical to any cache tail; historical number only"),
        "cache": cache_bin if a.source == "cache" else None,
        "val_frac": a.val_frac,
        "seed": a.seed, "batches": a.batches, "batch": a.batch, "ctx": a.ctx,
        "tokens_scored": tot_tok,
        "bytes_per_token": round(bpt, 6),
        "nll_per_token_nats": round(nll_per_tok, 6),
        "bpb": round(bpb, 6),
        "device": dev,
        "note": "bpb = nll_nats * log2(e) / bytes_per_token, matching train.py's "
                "bpb_factor exactly so this is directly comparable to the "
                "training runs' val bpb. Windows are non-overlapping and HF "
                "shifts labels internally (do NOT slice inputs yourself).",
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w", encoding="utf-8"), indent=2)
    print(f"\n=== {a.name} reference ===")
    print(f"  nll/token {nll_per_tok:.4f} nats   bpb {bpb:.4f} bits/byte "
          f"(x log2e / {bpt:.4f} b/tok)")
    print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()
