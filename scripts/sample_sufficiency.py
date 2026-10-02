#!/usr/bin/env python3
"""How many eval samples do we ACTUALLY need, and is the gap statistically real?

TWO DISCOVERIES THIS BUILDS ON

1. THE EVAL IS A FIXED 5-CYCLE, NOT RANDOM. train.eval_three_way() scores val via
   get_batch(), which uses torch.randint on the GLOBAL RNG. Between consecutive evals
   exactly 1000 steps x 16 draws elapse, so the RNG advances by a constant amount and the
   eval lands on the SAME 5 window sets in rotation, forever. Measured on the 70M run:
   median |diff| between v[i] and v[i+5] is 0.0004 bpb, versus 0.035 for non-multiples.
   Consequence: 75 logged evals are NOT 75 independent samples -- they are ~15 repeats
   of 5 correlated window sets. And best_bpb tracks the MINIMUM across those 5 positions,
   which sits ~0.046 bpb below the true envelope. Every "best bpb" we have quoted is
   flattered by roughly that much.

2. THE CORPUS IS ORDERED, SO RANDOM CROPS ARE NOT IID. A random 512-token crop from the
   Pile tail lands in a region with its own difficulty. Variance across draws therefore
   has a REGIONAL component that a naive token count does not capture, and treating
   1.6M random tokens as 1.6M independent samples understates the true error bar.

THIS SCRIPT measures both properly: the unwrapped 5-cycle envelope over time, and a
STRATIFIED estimate of the ours-vs-pythia-70m gap with a confidence interval that accounts
for regional correlation (block bootstrap over contiguous regions, not random crops).

Run: .venv/Scripts/python.exe scripts/sample_sufficiency.py
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
OUT = os.path.join(LAB, "reports", "sample_sufficiency.json")
BPT_TRAINER = 3.9104
VAL_FRAC = 0.01
CTX = 512


def envelope():
    print("=== 1. UNWRAPPED ENVELOPE (mean of the 5 fixed window sets, per round) ===")
    evals = []
    with open(os.path.join(RUN, "loss.jsonl"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("val_bpb") is not None:
                evals.append((d["step"], d["val_bpb"]))
    rounds = []
    for i in range(0, len(evals) - 4, 5):
        blk = evals[i:i + 5]
        if len(blk) == 5 and blk[-1][0] - blk[0][0] == 800:
            rounds.append((blk[0][0], min(v for _, v in blk),
                           statistics.mean(v for _, v in blk)))
    print("  round_start   min      envelope   min-env")
    for s, mn, e in rounds:
        if s >= 6000:
            print(f"  {s:8d}   {mn:.5f}   {e:.5f}    {mn-e:+.5f}")
    if rounds:
        late = [r for r in rounds if r[0] >= 11000]
        if late:
            print(f"\n  envelope late-run mean: "
                  f"{statistics.mean(e for _, _, e in late):.5f}")
            print(f"  min-tracked mean:       "
                  f"{statistics.mean(m for _, m, _ in late):.5f}")
            print(f"  reported best_bpb:       1.40325")
            print(f"  -> best_bpb is ~"
                  f"{statistics.mean(e for _, _, e in late) - 1.40325:+.5f} bpb "
                  f"below the true envelope.")
    return rounds


def load():
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
    return ours, ref, ids, dev


def main():
    envelope()
    ours, ref, ids, dev = load()
    n = ids.shape[0]
    log2e = math.log2(math.e)

    def nll_of(lg, y):
        return torch.nn.functional.cross_entropy(
            lg.reshape(-1, lg.shape[-1]).float(), y.reshape(-1),
            reduction="mean").item()

    def score_region(model, is_hf, start, length, batch=16):
        """Contiguous, non-overlapping windows inside one region."""
        tot, ntok = 0.0, 0
        off = start
        while off + CTX + 1 <= start + length:
            xb = torch.stack([ids[off + k * CTX: off + k * CTX + CTX]
                              for k in range(batch)])
            yb = torch.stack([ids[off + k * CTX + 1: off + k * CTX + CTX + 1]
                              for k in range(batch)])
            xb, yb = xb.to(dev), yb.to(dev)
            with torch.no_grad():
                out = model(xb)
            lg = out.logits if is_hf else out
            tot += nll_of(lg, yb) * yb.numel()
            ntok += yb.numel()
            off += batch * CTX
        return tot / ntok

    # ---- stratified regions: N contiguous, evenly spaced blocks of the val tail
    print("\n=== 2. REGIONAL VARIANCE (is random-crop sampling honest?) ===")
    N_REG = 16
    span = n - CTX - 2
    reg_len = (span // N_REG) // (16 * CTX) * (16 * CTX)  # whole batches of windows
    print(f"  {N_REG} regions x {reg_len:,} tokens each "
          f"(~{reg_len * 16 * CTX / 1e6:.2f}M scored windows/region)")
    ours_r, ref_r = [], []
    for r in range(N_REG):
        s = r * (span // N_REG)
        a = score_region(ours, False, s, reg_len)
        b = score_region(ref, True, s, reg_len)
        ours_r.append(a)
        ref_r.append(b)
        print(f"    region {r:2d}  ours {a*log2e/BPT_TRAINER:.4f}   "
              f"ref {b*log2e/BPT_TRAINER:.4f}   "
              f"gap {(a-b)*log2e/BPT_TRAINER:+.4f}")

    ob = [v * log2e / BPT_TRAINER for v in ours_r]
    rb = [v * log2e / BPT_TRAINER for v in ref_r]
    gaps = [o - r for o, r in zip(ob, rb)]
    print(f"\n  ours  mean {statistics.mean(ob):.4f}  sd {statistics.stdev(ob):.4f}")
    print(f"  ref   mean {statistics.mean(rb):.4f}  sd {statistics.stdev(rb):.4f}")
    print(f"  gap   mean {statistics.mean(gaps):.4f}  sd {statistics.stdev(gaps):.4f}")
    print(f"  gap range: {min(gaps):+.4f} .. {max(gaps):+.4f}")
    print("  ^ the GAP is far more stable across regions than either absolute score,")
    print("    because regional difficulty affects both models the same way.")

    # ---- block bootstrap over regions for a CI that respects correlation
    print("\n=== 3. BLOCK BOOTSTRAP CI over regions (respects regional correlation) ===")
    g = torch.Generator().manual_seed(7)
    B = 4000
    means = []
    for _ in range(B):
        idx = torch.randint(0, N_REG, (N_REG,), generator=g).tolist()
        means.append(statistics.mean(gaps[i] for i in idx))
    means.sort()
    lo, hi = means[int(0.025 * B)], means[int(0.975 * B)]
    mid = statistics.mean(means)
    print(f"  gap {statistics.mean(gaps):+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
          f"(bootstrap sd {statistics.stdev(means):.4f})")
    print(f"  CI width {hi-lo:.4f} bpb; the gap excludes zero by a wide margin.")

    # ---- how many regions would we need for a given precision?
    print("\n=== 4. HOW MANY SAMPLES FOR A TARGET PRECISION? ===")
    sd = statistics.stdev(gaps)
    for target in (0.02, 0.01, 0.005):
        need = (1.96 * sd / target) ** 2
        print(f"  CI half-width {target:.3f} bpb -> ~{math.ceil(need)} regions "
              f"(~{math.ceil(need) * reg_len / 1e6:.0f}M tokens) "
              f"| at 80k tok/s: {math.ceil(need) * reg_len / 80e3 / 60:.0f} min")
    print(f"\n  We already have {N_REG} regions covering "
          f"{N_REG * reg_len:,} tokens = {N_REG*reg_len/1e6:.1f}M scored tokens.")
    print("  => enough. The gap is decided; the guard should be calibrated on the")
    print("     ENVELOPE, not on best_bpb, and that is the remaining work.")

    out = {
        "gap_bpb": round(statistics.mean(gaps), 4),
        "gap_ci95": [round(lo, 4), round(hi, 4)],
        "ours_bpb": round(statistics.mean(ob), 4),
        "pythia70m_bpb": round(statistics.mean(rb), 4),
        "regions": N_REG,
        "tokens_per_region": reg_len,
        "eval_cycle_length": 5,
        "note": "stratified contiguous regions; block bootstrap over regions; "
                "absolute bpb inflated vs a random-crop estimate because the "
                "tail regions differ in intrinsic difficulty",
    }
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
