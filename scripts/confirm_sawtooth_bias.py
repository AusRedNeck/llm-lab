#!/usr/bin/env python3
"""Why does the trainer log val_bpb 1.40325 when independent scoring says ~1.4431?

HYPOTHESIS (from the variance study): best_bpb is a MINIMUM over noisy single-sample
evals. The curve has a repeating sawtooth of roughly +/-0.04 bpb, so the tracked minimum
sits about one oscillation-amplitude BELOW the true envelope. The trainer's arithmetic is
correct; the STATISTIC it reports is biased low by construction.

This measures the sawtooth directly and then re-scores ours vs pythia-70m in the
TRAINER'S sampling regime (scattered random crops across the whole tail) rather than the
strided-contiguous regime the reference scorer used. Absolute numbers improve; the gap is
already fair because both models shared one harness, but a representative sample is better.

Run: .venv/Scripts/python.exe scripts/confirm_sawtooth_bias.py
"""
import json
import math
import os
import statistics

import numpy as np
import torch

LAB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN = os.path.join(LAB, "runs", "20260930_2055_pythia_tokenizer_pile_train_full")
CACHE_BIN = os.path.join(LAB, "data", "pile_train_full_bpe_pythia70m.bin")
CACHE_META = os.path.join(LAB, "data", "pile_train_full_bpe_pythia70m.bin.meta.json")
CKPT = os.path.join(LAB, "checkpoints",
                    "exp002_pythia_tokenizer_202609302055_step13500.pt")
WEIGHTS = os.path.join(LAB, "data", "incoming", "pythia70m_weights")
BPT_TRAINER = 3.9104
VAL_FRAC = 0.01
CTX = 512


def sawtooth():
    print("=== 1. the logged val curve around the reported best (step 13400) ===")
    evals = []
    with open(os.path.join(RUN, "loss.jsonl"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            # NOTE: rows are keyed by step; there is no separate "eval" flag. val_bpb is
            # present only on eval steps, so a non-null val_bpb IS the eval marker.
            if d.get("val_bpb") is not None:
                evals.append((d["step"], d["val_bpb"]))
    tail = [(s, v) for s, v in evals if 11800 <= s <= 14200]
    for s, v in tail:
        print(f"    step {s:6d}  val_bpb {v:.5f}")
    if not tail:
        print("    (no evals in window)")
        return evals
    best = min(tail, key=lambda t: t[1])
    print(f"\n    min in window:  {best[1]:.5f} @ step {best[0]}")
    print(f"    mean in window: {statistics.mean(v for _, v in tail):.5f}")
    print(f"    bias (min-mean): {best[1] - statistics.mean(v for _, v in tail):+.5f}")
    print(f"    peak-to-peak:    {max(v for _, v in tail) - min(v for _, v in tail):.5f}")

    # The decisive observation: values REPEAT. If these were independent samples from
    # an ordered corpus they would scatter; instead each value recurs ~1000 steps later
    # to the 4th decimal. That means the eval is drawing from a small, FIXED cycle of
    # window sets -- the sampler advances by a constant amount per eval, so it keeps
    # landing on the same 5 batches in rotation. best_bpb is then a min over ~5
    # correlated draws, not over the distribution, and sits a fixed distance low.
    print("\n    period check (values recurring ~1000 steps apart):")
    d = dict(tail)
    for s, v in tail:
        for off in (1000, -1000):
            if s + off in d:
                print(f"      step {s} = {v:.5f}   step {s+off} = {d[s+off]:.5f}   "
                      f"delta {v - d[s+off]:+.5f}")
    return evals


def main():
    sawtooth()

    print("\n=== 2. head-to-head in the TRAINER's sampling regime ===")
    from model.transformer import Transformer
    from transformers import AutoModelForCausalLM

    meta = json.load(open(CACHE_META, encoding="utf-8"))
    total = int(meta["tokens"])
    cut = int(total * (1.0 - VAL_FRAC))
    arr = np.memmap(CACHE_BIN, dtype="int32", mode="r")
    ids = torch.from_numpy(np.asarray(arr[cut:total], dtype=np.int64))

    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    c = ck["cfg"]
    ours = Transformer(vocab_size=c["vocab_size"], context_length=c["context_length"],
                       embedding_dim=c["embedding_dim"], num_heads=c["num_heads"],
                       num_layers=c["num_layers"], use_rope=True,
                       rotary_pct=c.get("rotary_pct", 0.25), dropout=0.0,
                       parallel_residual=c.get("parallel_residual", True)).to(torch.float32)
    ours.load_state_dict(ck["model"])
    ref = AutoModelForCausalLM.from_pretrained(WEIGHTS, dtype=torch.float32)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ours.to(dev).eval()
    ref.to(dev).eval()

    log2e = math.log2(math.e)
    hi = ids.shape[0] - CTX - 1

    def nll_of(lg, y):
        return torch.nn.functional.cross_entropy(
            lg.reshape(-1, lg.shape[-1]).float(), y.reshape(-1),
            reduction="mean").item()

    def trial(model, is_hf, n_batches, batch, seed):
        g = torch.Generator().manual_seed(seed)
        tot, ntok = 0.0, 0
        for _ in range(n_batches):
            idx = torch.randint(0, hi, (batch,), generator=g).tolist()
            xb = torch.stack([ids[i:i + CTX] for i in idx]).to(dev)
            yb = torch.stack([ids[i + 1:i + CTX + 1] for i in idx]).to(dev)
            with torch.no_grad():
                out = model(xb)
            # HF returns CausalLMOutputWithPast; ours returns raw logits. Unwrap
            # unconditionally so the cross-entropy below sees the same tensor either way.
            lg = out.logits if is_hf else out
            tot += nll_of(lg, yb) * yb.numel()
            ntok += yb.numel()
        return tot / ntok

    def bpb(n):
        return n * log2e / BPT_TRAINER

    N, B, TRIALS = 200, 16, 3
    o = [trial(ours, False, N, B, 200 + s) for s in range(TRIALS)]
    r = [trial(ref, True, N, B, 200 + s) for s in range(TRIALS)]
    ob = [bpb(v) for v in o]
    rb = [bpb(v) for v in r]
    print(f"  ours        {statistics.mean(ob):.4f} (sd {statistics.stdev(ob):.4f})")
    print(f"  pythia-70m  {statistics.mean(rb):.4f} (sd {statistics.stdev(rb):.4f})")
    gap = statistics.mean(ob) - statistics.mean(rb)
    print(f"  gap         {gap:+.4f} bpb  ({100*gap/statistics.mean(rb):+.1f}%)")
    print(f"\n  trainer logged best 1.40325 -- a MIN over noisy evals, not the envelope.")
    print(f"  our true envelope is ~{statistics.mean(ob):.3f}; the logged best is")
    print(f"  {statistics.mean(ob) - 1.40325:+.4f} below it. Quoting the minimum")
    print(f"  flatters the model by roughly the sawtooth amplitude.")

    out = {
        "head_to_head_scattered": {
            "ours_bpb": round(statistics.mean(ob), 4),
            "pythia70m_bpb": round(statistics.mean(rb), 4),
            "gap_bpb": round(gap, 4),
            "gap_pct": round(100 * gap / statistics.mean(rb), 1),
            "samples_per_trial": N * B * CTX,
            "trials": TRIALS,
        },
        "trainer_logged_best": 1.40325,
        "note": "both models scored on the token-cache tail ids, one harness, same seeds",
    }
    p = os.path.join(LAB, "reports", "ours_vs_pythia70m.json")
    d = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}
    d["head_to_head_scattered"] = out["head_to_head_scattered"]
    d["note"] = out["note"]
    json.dump(d, open(p, "w", encoding="utf-8"), indent=2)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
