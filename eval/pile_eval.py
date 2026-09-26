#!/usr/bin/env python3
"""eval/pile_eval.py -- Away-game score: our checkpoint on Pile turf.

Same tail every run, same bpb metric as training val, so presets compare
fair out-of-distribution. Home turf flatters; Pile turf tells the truth.

Usage:
    python -m eval.pile_eval --ckpt checkpoints/exp002_s17m_rope_bpe_owt4k_202609161847_step5000.pt
    python -m eval.pile_eval --ckpt <ckpt> --tokenizer data/incoming/bpe_owt4k.json --batches 100

Slice lives at data/incoming/pile_deduped_slice.txt (STATE section 20:
shard 0 of EleutherAI/the_pile_deduplicated, 2927 docs, 20MB).
"""
from __future__ import annotations
import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.generate import load_model
from model.bpe import BPETokenizer


def score_pile(ckpt_path: str, tokenizer_path: str, slice_path: str,
               batches: int = 60, batch: int = 8,
               tail_frac: float = 0.10, device: str = "auto") -> dict:
    """Score a checkpoint on the Pile slice tail. Returns loss + bpb."""
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    dev = torch.device(device)

    model, c = load_model(ckpt_path, dev)
    model.eval()
    ctx, vocab = c["context_length"], c["vocab_size"]

    tok = BPETokenizer.load(tokenizer_path)
    raw = Path(slice_path).read_bytes().decode("utf-8", errors="replace")
    tail = raw[int(len(raw) * (1.0 - tail_frac)):]  # never trained on: held-out by construction
    tail_bytes = len(tail.encode("utf-8"))
    ids = torch.tensor(tok.encode(tail), dtype=torch.long)
    bpt = tail_bytes / len(ids)  # bpt from the SAME ids we score, not a prefix probe

    print(f"  tail: {len(ids):,} toks, bpt={bpt:.2f}, ctx={ctx}, device={device}", flush=True)
    total, n = 0.0, 0
    with torch.no_grad():
        for i in range(batches):
            # Fixed stride, not random: same windows every run, diffs are model not luck.
            s = (i * 7919) % (len(ids) - ctx - 1)
            x = ids[s:s + ctx].unsqueeze(0).repeat(batch, 1).to(dev)
            y = ids[s + 1:s + ctx + 1].unsqueeze(0).repeat(batch, 1).to(dev)
            total += F.cross_entropy(model(x).reshape(-1, vocab), y.reshape(-1)).item()
            n += 1
    loss = total / n
    return {"ckpt": ckpt_path, "loss": loss,
            "bpb": loss * math.log2(math.e) / bpt,
            "bytes_per_token": bpt, "tail_tokens": len(ids),
            "batches": n, "device": device}


def main():
    ap = argparse.ArgumentParser(description="Away-game eval on the Pile slice")
    ap.add_argument("--ckpt", required=True, help="checkpoint .pt to score")
    ap.add_argument("--tokenizer", default="data/incoming/bpe_owt4k.json")
    ap.add_argument("--slice", default="data/incoming/pile_deduped_slice.txt")
    ap.add_argument("--batches", type=int, default=60)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--tail_frac", type=float, default=0.10)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()

    print(f"Scoring {a.ckpt} on Pile turf...", flush=True)
    r = score_pile(a.ckpt, a.tokenizer, a.slice, a.batches, a.batch,
                   a.tail_frac, a.device)
    print(f"\n  loss (nats/token): {r['loss']:.4f}")
    print(f"  bpb:               {r['bpb']:.4f}")
    print(f"  bytes/token:       {r['bytes_per_token']:.2f}")


if __name__ == "__main__":
    main()
