#!/usr/bin/env python3
"""Score EleutherAI/pythia-70m on our held-out Pile val slice (the away game).

WHY THIS EXISTS
    The full-diet Pile arms were launched to measure the open-weights gap, but
    pythia-70m weights had never been downloaded and eval/pile_eval.py had
    never been run, so the comparison the runs exist to produce did not exist.
    This produces the reference number.

    VALIDITY RULES ENFORCED HERE
    - Same BYTES as our runs: data/incoming/pile_val_slice.txt, the exact last
      1% of the token cache (see scripts/extract_val_slice.py).
    - Same ENCODER: the real HF Pythia tokenizer, vocab 50304.
    - Same bpb convention: bpb = mean NLL per BYTE, so it is encoder-independent
      and directly comparable to the training runs' bpb.
    - Fixed seed, deterministic sampling, no training-loop contamination.

Usage:
    python scripts/score_pythia70m_reference.py [--batches 60] [--ctx 512]
"""
import argparse
import math
import json
import os

import numpy as np
import torch

LAB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS = os.path.join(LAB, "data", "incoming", "pythia70m_weights")
SLICE = os.path.join(LAB, "data", "incoming", "pile_val_slice.txt")
OUT = os.path.join(LAB, "reports", "pythia70m_reference.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, default=60)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(WEIGHTS)
    model = AutoModelForCausalLM.from_pretrained(WEIGHTS, torch_dtype=torch.float32)
    model.to(dev).eval()

    raw = open(SLICE, "rb").read().decode("utf-8", errors="replace")
    total_bytes = len(raw.encode("utf-8"))
    print(f"slice: {len(raw):,} chars, {total_bytes:,} bytes, device {dev}")

    ids = tok(raw, return_tensors="pt").input_ids[0]
    n = ids.shape[0]
    print(f"tokenized: {n:,} tokens (vocab {tok.vocab_size})")
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
    # noise. train.py line ~766: bpb = nll_nats * log2(e) / bytes_per_token.
    # The log2(e) factor is not optional: without it every number comes out
    # ~1.44x too large (nats vs bits) and reads 13.49 instead of ~9.4.
    BPT_TRAINER = 3.9104
    bpt_recomputed = total_bytes / n
    bpb = nll_per_tok * math.log2(math.e) / BPT_TRAINER
    res = {
        "model": "EleutherAI/pythia-70m",
        "weights": WEIGHTS,
        "slice": SLICE,
        "seed": a.seed, "batches": a.batches, "batch": a.batch, "ctx": a.ctx,
        "tokens_scored": tot_tok,
        "bytes_per_token_trainer": BPT_TRAINER,
        "bytes_per_token_recomputed": round(bpt_recomputed, 4),
        "nll_per_token_nats": round(nll_per_tok, 6),
        "bpb": round(bpb, 6),
        "device": dev,
        "note": "bpb = nll_nats * log2(e) / 3.9104, matching train.py's "
                "bpb_factor exactly so this is directly comparable to the "
                "training runs' val bpb. Windows are non-overlapping and HF "
                "shifts labels internally (do NOT slice inputs yourself).",
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w", encoding="utf-8"), indent=2)
    print("\n=== pythia-70m reference ===")
    print(f"  nll/token {nll_per_tok:.4f} nats   bpb {bpb:.4f} bits/byte (x log2e / {BPT_TRAINER} b/tok)")
    print(f"  wrote {OUT}")


if __name__ == "__main__":
    main()
