#!/usr/bin/env python3
"""Decisive test: is the trainer's val set the SAME data as pile_val_slice.txt?

Ruled out already (scripts/diagnose_eval_gap.py):
  - precision (bf16 vs fp32): 0.0001 bpb
  - window length (511 vs 512): 0.0010 bpb
  - sample size / random-crop variance: spread 0.0443 bpb, mean 1.4338
And yet the trainer logged 1.40325, which is BELOW the minimum of all five trials.

That leaves the data itself. The trainer splits the TOKEN CACHE (split_corpus on the
memmapped .bin), while the reference harness re-tokenizes pile_val_slice.txt, which was
produced by DECODING the cache tail and WRITING IT OUT AS TEXT. Decode -> write -> re-encode
is lossy and not guaranteed to reproduce the original ids. This measures the round trip
instead of assuming it is exact, because the meta file's claim of an exact replay is a
comment, not a measurement.

Run: .venv/Scripts/python.exe scripts/verify_val_slice_identity.py
"""
import json
import math
import os

import numpy as np
import torch

LAB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(LAB, "data", "pile_train_full_bpe_pythia70m.bin")
SLICE = os.path.join(LAB, "data", "incoming", "pile_val_slice.txt")
META = os.path.join(LAB, "data", "pile_train_full_bpe_pythia70m.bin.meta.json")
CKPT = os.path.join(LAB, "checkpoints",
                    "exp002_pythia_tokenizer_202609302055_step13500.pt")
BPT_TRAINER = 3.9104
VAL_FRAC = 0.01


def main():
    from transformers import AutoTokenizer
    meta = json.load(open(META, encoding="utf-8"))
    total = int(meta["tokens"])
    cut = int(total * (1.0 - VAL_FRAC))
    n_val = total - cut

    print("=== 1. token counts: cache tail vs re-tokenized slice ===")
    arr = np.memmap(BIN, dtype=np.int32, mode="r")
    cache_ids = np.asarray(arr[cut:total], dtype=np.int64)
    tok = AutoTokenizer.from_pretrained(os.path.join(LAB, "data", "incoming", "pythia70m_hf"))
    raw = open(SLICE, "rb").read().decode("utf-8", errors="replace")
    re_ids = tok(raw, return_tensors="pt").input_ids[0].numpy().astype(np.int64)
    print(f"  cache tail      {cache_ids.shape[0]:,} tokens")
    print(f"  re-tokenized    {re_ids.shape[0]:,} tokens")
    print(f"  difference      {re_ids.shape[0] - cache_ids.shape[0]:+,} tokens")

    # Where do they first diverge?
    n = min(cache_ids.shape[0], re_ids.shape[0])
    diff = np.nonzero(cache_ids[:n] != re_ids[:n])[0]
    print(f"  first divergence at index {diff[0]:,} " if diff.size else "  identical over the overlap")
    if diff.size:
        i = diff[0]
        lo = max(0, i - 6)
        print(f"    cache : ...{cache_ids[lo:i+6].tolist()}...")
        print(f"    re-tok: ...{re_ids[lo:i+6].tolist()}...")
        print(f"    decoded around it: {tok.decode(cache_ids[lo:i+6])!r}")
    print(f"  overlapping ids identical: {100*(n-diff.size)/n:.4f}%"
          if diff.size else "  overlapping ids identical: 100%")

    print("\n=== 2. score our checkpoint on BOTH token sets, same harness ===")
    from model.transformer import Transformer
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

    def score(ids_t, n_windows=60, window=512, seed=1234):
        g = torch.Generator().manual_seed(seed)
        tot, ntok = 0.0, 0
        for _ in range(n_windows):
            base = int(torch.randint(0, max(1, ids_t.shape[0] - window - 1), (1,),
                                     generator=g))
            # One window per draw, contiguous. Index with python ints, not a tensor --
            # a float/long tensor index silently raises or, worse, selects one element.
            chunk = ids_t[base:base + window + 1].unsqueeze(0).to(dev)
            if chunk.shape[1] < 2:
                continue
            x, y = chunk[:, :-1], chunk[:, 1:]
            with torch.no_grad():
                lg = m(x)
            tot += torch.nn.functional.cross_entropy(
                lg.reshape(-1, lg.shape[-1]).float(), y.reshape(-1),
                reduction="mean").item() * y.numel()
            ntok += y.numel()
        return tot / ntok

    t_cache = torch.from_numpy(cache_ids)
    t_retok = torch.from_numpy(re_ids)
    a = score(t_cache)
    b = score(t_retok)
    print(f"  cache tail ids    nll {a:.4f}  bpb {a*log2e/BPT_TRAINER:.4f}")
    print(f"  re-tokenized ids  nll {b:.4f}  bpb {b*log2e/BPT_TRAINER:.4f}")
    print(f"  delta             {(a-b)*log2e/BPT_TRAINER:+.4f} bpb")
    print(f"\n  trainer logged    1.40325")
    print("  -> whichever set reproduces ~1.403 is the set the trainer actually scored.")


if __name__ == "__main__":
    main()
