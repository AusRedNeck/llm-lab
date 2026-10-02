#!/usr/bin/env python3
"""Isolate WHY our checkpoint scores 1.5038 through the reference harness but the
trainer logged 1.40325 on what is supposed to be the same held-out data.

Three candidate causes, tested independently rather than argued about:

  A. PRECISION. The trainer evaluates under bf16 autocast; the reference scorer runs fp32.
     bf16 has ~8 mantissa bits, so per-token cross-entropy carries real quantisation error.
  B. WINDOW LENGTH. train.get_batch() returns x = data[i:i+ctx] and y = data[i+1:i+ctx+1],
     so the y window is ctx-1 tokens (511 at ctx=512). A scorer that feeds 512 tokens scores
     511 predictions per window from a 512-token input. Different count, different mean.
  C. SAMPLE SIZE. The trainer uses batches=20 x batch=16 x 511 = ~163k tokens per eval, on
     RANDOM crops, re-drawn every eval. The reference uses 60 x 8 x 512 = 245,760 tokens on
     a fixed seed. Random crops of a 20M-token val set have real variance.

Run: .venv/Scripts/python.exe scripts/diagnose_eval_gap.py
"""
import json
import math
import os

import numpy as np
import torch

LAB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SLICE = os.path.join(LAB, "data", "incoming", "pile_val_slice.txt")
CKPT = os.path.join(LAB, "checkpoints",
                    "exp002_pythia_tokenizer_202609302055_step13500.pt")
BPT_TRAINER = 3.9104


def nll_of(logits, y):
    return torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(), y.reshape(-1),
        reduction="mean").item()


def build():
    from transformers import AutoTokenizer
    from model.transformer import Transformer
    tok = AutoTokenizer.from_pretrained(os.path.join(LAB, "data", "incoming", "pythia70m_hf"))
    raw = open(SLICE, "rb").read().decode("utf-8", errors="replace")
    ids = tok(raw, return_tensors="pt").input_ids[0]
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    c = ck["cfg"]
    m = Transformer(vocab_size=c["vocab_size"], context_length=c["context_length"],
                    embedding_dim=c["embedding_dim"], num_heads=c["num_heads"],
                    num_layers=c["num_layers"], use_rope=True,
                    rotary_pct=c.get("rotary_pct", 0.25), dropout=0.0,
                    parallel_residual=c.get("parallel_residual", True)).to(torch.float32)
    m.load_state_dict(ck["model"])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return m.to(dev).eval(), ids, dev, tok


def score_fixed(m, ids, dev, n_windows, window, dtype):
    """Non-overlapping windows from the start of the slice, in a given dtype."""
    tot, ntok = 0.0, 0
    for i in range(n_windows):
        s = i * window
        if s + window + 1 > ids.shape[0]:
            break
        chunk = ids[s:s + window + 1].unsqueeze(0).to(dev)
        x, y = chunk[:, :-1], chunk[:, 1:]
        with torch.no_grad():
            if dtype is None:
                lg = m(x)
            else:
                with torch.amp.autocast("cuda", dtype=dtype, enabled=True):
                    lg = m(x)
        tot += nll_of(lg, y) * y.numel()
        ntok += y.numel()
    return tot / ntok


def main():
    m, ids, dev, tok = build()
    print(f"device={dev} slice_tokens={ids.shape[0]:,}")
    bf16 = torch.bfloat16 if torch.cuda.is_bf16_supported() else None
    log2e = math.log2(math.e)

    def bpb(nll):
        return nll * log2e / BPT_TRAINER

    print("\n=== A. PRECISION: fp32 vs bf16 autocast, identical windows ===")
    W, NW = 512, 60
    a32 = score_fixed(m, ids, dev, NW, W, None)
    a16 = score_fixed(m, ids, dev, NW, W, bf16) if bf16 else None
    print(f"  fp32          nll {a32:.4f}  bpb {bpb(a32):.4f}")
    if a16:
        print(f"  bf16 autocast nll {a16:.4f}  bpb {bpb(a16):.4f}   delta bpb {bpb(a16)-bpb(a32):+.4f}")

    print("\n=== B. WINDOW LENGTH: 512-token input vs the trainer's 511-token pairs ===")
    b511 = score_fixed(m, ids, dev, NW, 511, None)
    print(f"  511-token y    nll {b511:.4f}  bpb {bpb(b511):.4f}   delta vs 512: {bpb(b511)-bpb(a32):+.4f}")

    print("\n=== C. SAMPLE SIZE + RANDOM CROPS: the trainer's own regime ===")
    # Trainer: batches=20, batch=16, ctx=512 -> x is 512 tokens, y is 511. Random offsets.
    gen = torch.Generator().manual_seed(0)
    res = {}
    for trial in range(5):
        tot, ntok = 0.0, 0
        for _ in range(20):
            idx = torch.randint(0, ids.shape[0] - 512 - 1, (16,), generator=gen).tolist()
            xb = torch.stack([ids[i:i + 512] for i in idx]).to(dev)
            yb = torch.stack([ids[i + 1:i + 512 + 1] for i in idx]).to(dev)
            with torch.no_grad():
                lg = m(xb)
            tot += nll_of(lg, yb) * yb.numel()
            ntok += yb.numel()
        res[trial] = tot / ntok
    vals = list(res.values())
    print(f"  5 trials x (20 batches x 16 x 512 random crops):")
    for k, v in res.items():
        print(f"    trial {k}: nll {v:.4f}  bpb {bpb(v):.4f}")
    print(f"  spread {bpb(max(vals)) - bpb(min(vals)):.4f} bpb across trials")
    print(f"  mean   {np.mean([bpb(v) for v in vals]):.4f} bpb")

    print("\n=== SUMMARY ===")
    print(f"  trainer logged          1.40325")
    print(f"  our fp32 harness, 512   {bpb(a32):.4f}")
    print(f"  our fp32 harness, 511   {bpb(b511):.4f}")
    if a16:
        print(f"  our bf16 harness, 512   {bpb(a16):.4f}")
    print(f"  trainer-regime mean     {np.mean([bpb(v) for v in vals]):.4f}")


if __name__ == "__main__":
    main()
