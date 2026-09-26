#!/usr/bin/env python3
"""Train-mode vs eval-mode loss on IDENTICAL batches.

Settles the "lying train loss" question in one run: same weights, same
batches, both modes. Three possible verdicts:
  * train-mode ~2.x, eval-mode ~4.x -> MODE GAP. Forward differs by mode
    (dropout/norm path). Bug is in model code, not data.
  * both ~online-train -> weights are fine, the run's eval rescoring is broken.
  * both ~run's random_train -> weights genuinely score there, the ONLINE
    train number is the liar (loss logging / loader story).

Usage (on the machine that owns the checkpoint + tok_cache):
  uv run python -m eval.mode_probe --checkpoint checkpoints/<run>_step<N>.pt \
      --tok_cache data/<corpus>.bin [--batches 10] [--val_frac 0.01]
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F

from model.transformer import Transformer
from train.train import get_batch, get_device, load_token_cache, split_corpus


def score(model, batches, use_amp, dtype) -> list[float]:
    # One forward per batch, no grad either way: grad does not change
    # forward math, and this keeps VRAM flat across modes.
    losses = []
    with torch.no_grad():
        for x, y in batches:
            with torch.amp.autocast("cuda", dtype=dtype, enabled=use_amp):
                logits = model(x)
                # Transformer returns logits; be loud if that ever changes.
                if isinstance(logits, tuple):
                    logits = logits[0]
                losses.append(F.cross_entropy(
                    logits.reshape(-1, model.cfg_vocab),
                    y.reshape(-1)).item())
    return losses


def main() -> None:
    ap = argparse.ArgumentParser(description="train-mode vs eval-mode probe")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tok_cache", required=True)
    ap.add_argument("--batches", type=int, default=10)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--val_frac", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    device = get_device()
    print(f"device={device}")

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt["cfg"]
    print(f"ckpt step={ckpt.get('step')} keys={sorted(cfg.keys())}")

    # Checkpoint cfg is truth: same fields train.py built the model with.
    # rotary_pct is NOT stored (CLI-only) -- 1.0 is train.py's default.
    model = Transformer(
        vocab_size=cfg["vocab_size"],
        context_length=cfg["context_length"],
        embedding_dim=cfg["embedding_dim"],
        num_heads=cfg["num_heads"],
        num_layers=cfg["num_layers"],
        use_rope=cfg.get("use_rope", False),
        rotary_pct=cfg.get("rotary_pct", 1.0),
        dropout=cfg.get("dropout", 0.0),
    )
    model.cfg_vocab = cfg["vocab_size"]  # scoring needs V, not stored on model
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    print(f"load: missing={len(missing)} unexpected={len(unexpected)}")
    if missing or unexpected:
        print(f"  missing={missing[:5]} unexpected={unexpected[:5]}")
    model.to(device)

    # Same AMP regime as training: cuda bf16-if-supported else fp16, else off.
    use_amp, dtype = False, torch.float16
    if device.type == "cuda":
        use_amp = True
        try:
            if torch.cuda.is_bf16_supported():
                dtype = torch.bfloat16
        except Exception:
            pass
    print(f"amp: cuda={use_amp} dtype={dtype if use_amp else 'n/a'}")

    corpus = load_token_cache(args.tok_cache)
    train_data, _ = split_corpus(corpus, args.val_frac, cfg["context_length"])
    print(f"train tokens={len(train_data):,} ctx={cfg['context_length']}")

    # Fix the batches ONCE so both modes score identical data.
    torch.manual_seed(args.seed)
    pos = [0]
    batches = [get_batch(train_data, args.batch, cfg["context_length"],
                         cfg["vocab_size"], device, pos)
               for _ in range(args.batches)]

    model.train()
    train_losses = score(model, batches, use_amp, dtype)
    model.eval()
    eval_losses = score(model, batches, use_amp, dtype)

    # Dropout-zero pass: if the gap vanishes here, dropout is the culprit.
    saved_p = [m.p for m in model.modules() if isinstance(m, torch.nn.Dropout)]
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0
    model.train()
    nodrop_losses = score(model, batches, use_amp, dtype)
    for m, p in zip((m for m in model.modules()
                     if isinstance(m, torch.nn.Dropout)), saved_p):
        m.p = p

    print(f"\n{'batch':>5} {'train_mode':>10} {'eval_mode':>10} {'no_dropout':>10}")
    for i, (t, e, n) in enumerate(zip(train_losses, eval_losses, nodrop_losses)):
        print(f"{i:>5} {t:>10.4f} {e:>10.4f} {n:>10.4f}")
    mt = sum(train_losses) / len(train_losses)
    me = sum(eval_losses) / len(eval_losses)
    mn = sum(nodrop_losses) / len(nodrop_losses)
    print(f"{'mean':>5} {mt:>10.4f} {me:>10.4f} {mn:>10.4f}")
    print(f"\ngap eval-train = {me - mt:+.4f} | nodrop-train = {mn - mt:+.4f}")
    if me - mt > 0.5 and abs(mn - mt) < 0.2:
        print("VERDICT: MODE GAP, dropout is the culprit.")
    elif me - mt > 0.5:
        print("VERDICT: MODE GAP, but NOT dropout -- some other train/eval path.")
    else:
        print("VERDICT: no mode gap -- compare these means against the run's "
              "online train vs random_train to see which number lies.")


if __name__ == "__main__":
    main()
