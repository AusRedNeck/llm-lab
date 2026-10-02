#!/usr/bin/env python3
"""Settle whether the trainer's val_bpb (1.40325) is real or under-reported.

STATE OF THE EVIDENCE (2026-10-01)
  - Val DATA identity was the first suspect and it WAS a real bug: pile_val_slice.txt
    matches the cache tail on only 1.68% of positions. Both scorers now read the cache
    directly. Fixed.
  - But after that fix the trainer still logs 1.40325 while independent scoring of the
    SAME ids gives 1.44-1.51 depending on how the windows are drawn.
  - Ruled out by measurement: bf16 vs fp32 (0.0001 bpb), 511 vs 512 window (0.0010 bpb).

WHAT REMAINS: estimator variance. The trainer draws batches=20 x batch=16 RANDOM crops
per eval and reads off a single number. Region matters a lot in an ordered corpus -- the
Pile tail is not i.i.d. -- so a 320-window draw can land anywhere.

This runs the trainer's EXACT regime at two sample sizes:
  - small (20 x 16, = what the trainer uses)  -> shows the SPREAD the trainer lives with
  - large (200 x 16, 10x)                    -> shows the VALUE it converges to
If large-sample converges to ~1.40, the trainer is right and small samples are just noisy.
If it converges to ~1.46, the trainer is under-reporting and every bpb we have quoted
needs a confidence interval.

Run: .venv/Scripts/python.exe scripts/eval_variance_study.py
"""
import json
import math
import os
import statistics

import numpy as np
import torch

LAB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_BIN = os.path.join(LAB, "data", "pile_train_full_bpe_pythia70m.bin")
CACHE_META = os.path.join(LAB, "data", "pile_train_full_bpe_pythia70m.bin.meta.json")
CKPT = os.path.join(LAB, "checkpoints",
                    "exp002_pythia_tokenizer_202609302055_step13500.pt")
BPT_TRAINER = 3.9104
VAL_FRAC = 0.01
CTX = 512


def main():
    from model.transformer import Transformer
    meta = json.load(open(CACHE_META, encoding="utf-8"))
    total = int(meta["tokens"])
    cut = int(total * (1.0 - VAL_FRAC))
    arr = np.memmap(CACHE_BIN, dtype="int32", mode="r")
    ids = torch.from_numpy(np.asarray(arr[cut:total], dtype=np.int64))

    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    c = ck["cfg"]
    m = Transformer(vocab_size=c["vocab_size"], context_length=c["context_length"],
                    embedding_dim=c["embedding_dim"], num_heads=c["num_heads"],
                    num_layers=c["num_layers"], use_rope=True,
                    rotary_pct=c.get("rotary_pct", 0.25), dropout=0.0,
                    parallel_residual=c.get("parallel_residual", True)).to(torch.float32)
    m.load_state_dict(ck["model"])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m.to(dev).eval()

    log2e = math.log2(math.e)
    hi = ids.shape[0] - CTX - 1
    print(f"val ids {ids.shape[0]:,}  device {dev}")

    def trial(n_batches, batch, seed):
        """Exactly train.py's regime: random crops, x=[i:i+ctx], y=[i+1:i+ctx+1]."""
        g = torch.Generator().manual_seed(seed)
        tot, ntok = 0.0, 0
        for _ in range(n_batches):
            idx = torch.randint(0, hi, (batch,), generator=g).tolist()
            xb = torch.stack([ids[i:i + CTX] for i in idx]).to(dev)
            yb = torch.stack([ids[i + 1:i + CTX + 1] for i in idx]).to(dev)
            with torch.no_grad():
                lg = m(xb)
            tot += torch.nn.functional.cross_entropy(
                lg.reshape(-1, lg.shape[-1]).float(), yb.reshape(-1),
                reduction="mean").item() * yb.numel()
            ntok += yb.numel()
        return tot / ntok

    def bpb(n):
        return n * log2e / BPT_TRAINER

    print("\n=== SMALL: the trainer's own eval size (20 batches x 16 = 320 windows) ===")
    print("    12 independent draws, each what ONE logged eval looks like:")
    small = [trial(20, 16, s) for s in range(12)]
    sb = [bpb(v) for v in small]
    for k, v in enumerate(sb):
        print(f"    draw {k:2d}: {v:.4f}")
    print(f"    min {min(sb):.4f}  max {max(sb):.4f}  mean {statistics.mean(sb):.4f}  "
          f"sd {statistics.stdev(sb):.4f}")
    print(f"    trainer logged 1.40325 -> "
          f"{'WITHIN' if min(sb) <= 1.40325 <= max(sb) else 'OUTSIDE'} this spread")

    print("\n=== LARGE: 10x the tokens (200 x 16 = 3,200 windows) ===")
    big = [trial(200, 16, 100 + s) for s in range(3)]
    bb = [bpb(v) for v in big]
    for k, v in enumerate(bb):
        print(f"    draw {k}: {v:.4f}")
    print(f"    mean {statistics.mean(bb):.4f}  sd {statistics.stdev(bb):.4f}")

    print("\n=== VERDICT ===")
    conv = statistics.mean(bb)
    print(f"  small-sample mean  {statistics.mean(sb):.4f} (sd {statistics.stdev(sb):.4f})")
    print(f"  large-sample mean  {conv:.4f} (sd {statistics.stdev(bb):.4f})")
    print(f"  trainer logged     1.40325")
    print(f"  bias (large - logged) {conv - 1.40325:+.4f} bpb")
    n_tok = 200 * 16 * (CTX - 1)
    se = statistics.stdev(bb) / math.sqrt(3)
    print(f"\n  large-sample SE {se:.4f} bpb on ~{n_tok:,} tokens/draw")
    print(f"  95% CI ~ {conv-1.96*se:.4f} .. {conv+1.96*se:.4f}")
    print("\n  IF large-sample lands within ~0.02 of the logged value: the trainer is")
    print("  accurate and single-eval bpb just needs an error bar (+/- ~0.04 at n=320).")
    print("  IF it lands near 1.46: the trainer under-reports and every bpb we have")
    print("  quoted needs to be restated as a range.")


if __name__ == "__main__":
    main()
