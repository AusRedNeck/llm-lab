#!/usr/bin/env python3
"""Eval HuggingFace models on our val set — fair comparison via nats-per-byte.

Usage:
    uv run python -m eval.hf_eval --model HuggingFaceTB/SmolLM2-135M
    uv run python -m eval.hf_eval --model HuggingFaceTB/SmolLM2-360M
    uv run python -m eval.hf_eval --model HuggingFaceTB/SmolLM2-1.7B
"""
from __future__ import annotations
import argparse
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_val_text(path: str = "data/incoming/openwebtext_sample_1gb.txt",
                  val_frac: float = 0.01) -> str:
    """Load the last val_frac of the text file as held-out val data."""
    size = os.path.getsize(path)
    val_bytes = int(size * val_frac)
    with open(path, "rb") as f:
        f.seek(size - val_bytes)
        return f.read().decode("utf-8", errors="replace")


def eval_hf_model(model_name: str, val_text: str, max_samples: int = 500,
                  ctx: int = 512, device: str = "auto") -> dict:
    """Evaluate a HuggingFace model on val text, return nats-per-byte."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {model_name}...", flush=True)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)

    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    model = model.to(device)
    model.eval()

    vocab = model.config.vocab_size
    hf_ctx = getattr(model.config, "max_position_embeddings",
                     getattr(model.config, "n_positions", ctx))
    ctx = min(ctx, hf_ctx)

    print(f"  vocab={vocab}, ctx={ctx}, device={device}", flush=True)

    # Tokenize val text
    print(f"  Tokenizing {len(val_text):,} val bytes...", flush=True)
    input_ids = tok.encode(val_text, return_tensors="pt")
    total_tokens = input_ids.shape[1]
    total_bytes = len(val_text.encode("utf-8"))
    bytes_per_token = total_bytes / total_tokens
    print(f"  {total_tokens:,} tokens, {bytes_per_token:.2f} bytes/token", flush=True)

    # Evaluate
    nats_total = 0.0
    count = 0
    stride = ctx // 2  # overlapping windows for more coverage

    print(f"  Evaluating ({max_samples} samples, ctx={ctx})...", flush=True)
    with torch.no_grad():
        for i in range(0, min(total_tokens - ctx - 1, max_samples * stride), stride):
            if count >= max_samples:
                break
            chunk = input_ids[:, i:i + ctx + 1].to(device)
            x = chunk[:, :-1]
            y = chunk[:, 1:]
            logits = model(x).logits
            loss = F.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1)).item()
            nats_total += loss
            count += 1

    avg_nats = nats_total / count
    nats_per_byte = avg_nats / bytes_per_token
    ppl = math.exp(avg_nats)

    return {
        "model": model_name,
        "val_loss": avg_nats,
        "perplexity": ppl,
        "nats_per_byte": nats_per_byte,
        "bytes_per_token": bytes_per_token,
        "total_tokens": total_tokens,
        "samples": count,
        "device": device,
    }


def main():
    ap = argparse.ArgumentParser(description="Eval HuggingFace models")
    ap.add_argument("--model", nargs="+",
                     default=["HuggingFaceTB/SmolLM2-135M",
                              "HuggingFaceTB/SmolLM2-360M"])
    ap.add_argument("--val_text", default="data/incoming/openwebtext_sample_1gb.txt")
    ap.add_argument("--max_samples", type=int, default=500)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    val_text = load_val_text(args.val_text)
    print(f"Val text: {len(val_text):,} chars\n")

    results = []
    for model_name in args.model:
        print(f"\n{'='*60}")
        r = eval_hf_model(model_name, val_text, args.max_samples, args.ctx, args.device)
        results.append(r)
        print(f"\n  Results:")
        print(f"    val_loss (nats/token): {r['val_loss']:.4f}")
        print(f"    perplexity:            {r['perplexity']:.2f}")
        print(f"    nats_per_byte:         {r['nats_per_byte']:.4f}")
        print(f"    bytes_per_token:       {r['bytes_per_token']:.2f}")
        print(f"    device:                {r['device']}")

    # Summary table
    print(f"\n{'='*60}")
    print(f"\n{'Model':<40} {'val_loss':>9} {'ppl':>8} {'nats/B':>8} {'B/tok':>6}")
    print("-" * 75)
    for r in sorted(results, key=lambda x: x["nats_per_byte"]):
        print(f"{r['model']:<40} {r['val_loss']:>9.4f} {r['perplexity']:>8.2f} "
              f"{r['nats_per_byte']:>8.4f} {r['bytes_per_token']:>6.2f}")


if __name__ == "__main__":
    main()
