"""Experiment 002 — first real training loop.

Usage:
    python -m train.train --steps 500 --batch 32 --preset bytes10m
    python -m train.train --steps 5000 --batch 64 --preset bytes10m --data tinystories

Presets:
    toy      = current 4-layer toy (proves the loop works)
    bytes10m = 10M shape, byte-level vocab 256 (train THIS first)
    tiny10m  = 10M shape, GPT-2 BPE vocab 50304 (needs BPE tokenizer, later)

Data:
    default  = synthetic byte stream (no downloads, proves loss decreases)
    tinystories = TinyStories train split, byte-encoded (real English words,
                 still byte tokens — better signal before you build BPE)

Checkpoints land in checkpoints/exp002_<preset>_step<N>.pt
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import torch
import torch.nn.functional as F

# Allow `python -m train.train` from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.config import TINY_10M, TINY_10M_BYTES, TOY_1M
from model.transformer import Transformer

PRESETS = {"toy": TOY_1M, "bytes10m": TINY_10M_BYTES, "tiny10m": TINY_10M}


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synthetic_batch(batch: int, ctx: int, vocab: int, device) -> torch.Tensor:
    # Random bytes with local structure (repeating motifs) so there is
    # something learnable — pure uniform noise trains to flat loss.
    data = torch.randint(0, vocab, (batch, ctx + 1), device=device)
    # Inject a repeat rule: every 16th token copies the token 8 back.
    # A working transformer should beat chance on this.
    data[:, 16::16] = data[:, 8:-7:16]
    return data


def load_tinystories(ctx: int, device, cache="data/TinyStories.txt") -> torch.Tensor | None:
    """Download TinyStories once, cache as raw text, return byte-encoded tensor."""
    if not os.path.exists(cache):
        try:
            import urllib.request
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            url = ("https://huggingface.co/datasets/roneneldan/TinyStories/"
                   "resolve/main/TinyStoriesV2-GPT4-train.txt")
            print(f"downloading TinyStories (~2GB) -> {cache} ...")
            urllib.request.urlretrieve(url, cache)
        except Exception as e:
            print(f"download failed ({e}); falling back to synthetic data")
            return None
    print(f"loading {cache} ...")
    with open(cache, "rb") as f:
        raw = f.read()
    print(f"  {len(raw) / 1e6:.1f}M bytes")
    return torch.from_numpy(
        __import__("numpy").frombuffer(raw, dtype="uint8").copy()
    )


def get_batch(source: torch.Tensor | None, batch: int, ctx: int,
              vocab: int, device, pos: list) -> tuple[torch.Tensor, torch.Tensor]:
    if source is None:
        data = synthetic_batch(batch, ctx, vocab, device)
        return data[:, :-1], data[:, 1:]
    # Random crops from the corpus.
    idx = torch.randint(0, len(source) - ctx - 1, (batch,)).tolist()
    x = torch.stack([source[i:i + ctx] for i in idx]).long().to(device)
    y = torch.stack([source[i + 1:i + ctx + 1] for i in idx]).long().to(device)
    return x, y


def lr_schedule(step: int, warmup: int, total: int, peak: float) -> float:
    # Linear warmup, then cosine decay to 10% of peak.
    if step < warmup:
        return peak * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return 0.1 * peak + 0.9 * peak * 0.5 * (1 + math.cos(math.pi * p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="bytes10m", choices=list(PRESETS))
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--data", default="synthetic", choices=["synthetic", "tinystories"])
    ap.add_argument("--use_rope", action="store_true",
                    help="Exp 003: rotary positions instead of learned absolute")
    ap.add_argument("--out", default="checkpoints")
    args = ap.parse_args()

    cfg = PRESETS[args.preset]
    device = get_device()
    print(f"preset={args.preset} params~{cfg.num_params() / 1e6:.1f}M "
          f"ctx={cfg.context_length} device={device}")

    torch.manual_seed(0)
    model = Transformer(
        vocab_size=cfg.vocab_size,
        context_length=cfg.context_length,
        embedding_dim=cfg.embedding_dim,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        use_rope=args.use_rope,
    ).to(device)
    model.train()

    # 4070 Ti SUPER: tf32 + fused AdamW is the free speedup.
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=0.1)
    warmup = min(200, args.steps // 10)

    corpus = None
    if args.data == "tinystories":
        corpus = load_tinystories(cfg.context_length, device)
        if corpus is None:
            print("TinyStories unavailable, using synthetic.")

    pos = [0]
    os.makedirs(args.out, exist_ok=True)
    running = 0.0

    for step in range(1, args.steps + 1):
        lr = lr_schedule(step, warmup, args.steps, args.lr)
        for g in opt.param_groups:
            g["lr"] = lr

        x, y = get_batch(corpus, args.batch, cfg.context_length,
                         cfg.vocab_size, device, pos)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), y.reshape(-1))

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        running += (loss.item() - running) / min(step, 50)
        if step % 25 == 0 or step == 1:
            print(f"step {step:5d}/{args.steps} loss={loss.item():.4f} "
                  f"avg50={running:.4f} lr={lr:.1e}", flush=True)
        if step % 500 == 0 or step == args.steps:
            tag = f"{args.preset}{'_rope' if args.use_rope else ''}"
            ckpt = os.path.join(args.out, f"exp002_{tag}_step{step}.pt")
            saved_cfg = dict(vars(cfg))
            saved_cfg["use_rope"] = args.use_rope
            torch.save({"cfg": saved_cfg, "model": model.state_dict(),
                        "step": step}, ckpt)
            print(f"  saved {ckpt}")

    print(f"done. final avg50 loss={running:.4f}")


if __name__ == "__main__":
    main()
