"""Prompt it like a real model.

Usage:
    python -m inference.generate --ckpt checkpoints/exp002_bytes10m_step5000.pt
    python -m inference.generate --ckpt <file> --prompt "Once upon a time" --tokens 200 --temp 0.8 --topk 40
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.transformer import Transformer


def load_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    c = ckpt["cfg"]
    model = Transformer(
        vocab_size=c["vocab_size"],
        context_length=c["context_length"],
        embedding_dim=c["embedding_dim"],
        num_heads=c["num_heads"],
        num_layers=c["num_layers"],
        use_rope=c.get("use_rope", False),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, c


def sample_next(logits: torch.Tensor, temp: float, topk: int) -> torch.Tensor:
    # logits: [V] for the last position
    if temp <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    logits = logits / temp
    if topk > 0 and topk < logits.numel():
        vals, _ = torch.topk(logits, topk)
        logits = torch.where(logits < vals[-1], torch.tensor(float("-inf")), logits)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--prompt", default="Once upon a time")
    ap.add_argument("--tokens", type=int, default=200)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--topk", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, c = load_model(args.ckpt, device)
    ctx = c["context_length"]

    # Byte-level: prompt -> raw utf-8 bytes.
    ids = list(args.prompt.encode("utf-8"))
    x = torch.tensor([ids], dtype=torch.long, device=device)

    out = ids[:]
    with torch.no_grad():
        for _ in range(args.tokens):
            window = x[:, -ctx:]
            logits = model(window)
            nxt = sample_next(logits[0, -1], args.temp, args.topk)
            out.append(int(nxt.item()))
            x = torch.cat([x, nxt.view(1, 1)], dim=1)

    text = bytes(b % 256 for b in out).decode("utf-8", errors="replace")
    print(text)


if __name__ == "__main__":
    main()
