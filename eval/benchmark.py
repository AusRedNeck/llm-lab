"""Eval harness: fair metrics across vocab sizes.

The one rule: raw perplexity and token accuracy are NOT comparable
across different vocab sizes. The fair metric is nats-per-byte.

Usage:
    from eval.benchmark import eval_checkpoint, eval_report
    result = eval_checkpoint("checkpoints/exp004_best.pt")
    results = eval_report(["checkpoints/exp004_best.pt",
                           "checkpoints/exp006_best.pt"])
"""
import math
import os

import torch
import torch.nn.functional as F

from model.loss import cross_entropy_loss


def perplexity(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """ppl = exp(avg CE). 0.73 val loss -> ~2.08 ppl. Lower = less surprised."""
    return math.exp(cross_entropy_loss(logits, targets).item())


def token_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Fraction of positions where argmax == target."""
    preds = logits.argmax(dim=-1)
    return (preds == targets).float().mean().item()


def nats_per_byte(val_nll: float, bytes_per_token: float) -> float:
    """Fair cross-vocab metric: surprise per byte of raw text.

    val_nll:      average cross-entropy loss in nats per token
    bytes_per_token: how many raw bytes each token represents on average
                     (1.0 for byte-level, ~4.1 for bpe2k on TinyStories)

    Returns nats per byte — lower is better, comparable across vocab sizes.
    """
    if bytes_per_token <= 0:
        raise ValueError(f"bytes_per_token must be positive, got {bytes_per_token}")
    return val_nll / bytes_per_token


def _measure_bytes_per_token(val_data: torch.Tensor, ctx: int, batches: int = 20) -> float:
    """Estimate average bytes-per-token from the val corpus.

    For byte-level (vocab 256) this is always 1.0.
    For BPE, tokens compress multiple bytes — measure empirically.
    """
    # We need the raw byte counts vs token counts.
    # val_data is already tokenized (ints). For byte-level, 1 token = 1 byte.
    # For BPE, we can't reverse-engineer bytes from token IDs without the
    # tokenizer. So we measure differently: the val data length in tokens
    # vs the original text length in bytes.
    #
    # Shortcut: if vocab_size <= 256, it's byte-level -> 1.0
    # Otherwise, we need to load the tokenizer or accept a provided value.
    # For now, return 1.0 as default (byte-level assumption).
    # The caller can override via the bytes_per_token kwarg.
    return 1.0


def _load_val_data(tok=None, ctx: int = 128, device: torch.device = None):
    """Load TinyStories val split. Returns (val_tensor, bytes_per_token)."""
    if device is None:
        device = torch.device("cpu")

    if tok is not None:
        # BPE path
        cache_pt = "data/TinyStories_bpe2k.pt"
        if not os.path.exists(cache_pt):
            return None, 1.0
        corpus = torch.load(cache_pt, map_location="cpu", weights_only=True)
        cut = int(len(corpus) * 0.99)
        val = corpus[cut:]
        # Empirical bytes-per-token: total file bytes / total tokens
        # We stored this when encoding — compute from file if available.
        cache_txt = "data/TinyStories.txt"
        if os.path.exists(cache_txt):
            txt_bytes = os.path.getsize(cache_txt)
            bpt = txt_bytes / len(corpus)
        else:
            bpt = 4.0  # rough estimate for BPE on English
        return val, bpt
    else:
        # Byte path
        cache = "data/TinyStories.txt"
        if not os.path.exists(cache):
            return None, 1.0
        with open(cache, "rb") as f:
            raw = f.read()
        corpus = torch.from_numpy(
            __import__("numpy").frombuffer(raw, dtype="uint8").copy()
        )
        cut = int(len(corpus) * 0.99)
        val = corpus[cut:]
        return val, 1.0


def _get_batch(source, batch, ctx, vocab, device, pos):
    """Random crop from corpus. Matches train.py's get_batch."""
    idx = torch.randint(0, len(source) - ctx - 1, (batch,)).tolist()
    x = torch.stack([source[i:i + ctx] for i in idx]).long().to(device)
    y = torch.stack([source[i + 1:i + ctx + 1] for i in idx]).long().to(device)
    return x, y


def eval_checkpoint(ckpt_path: str, val_batches: int = 20,
                    device: torch.device = None,
                    bytes_per_token: float = None) -> dict:
    """Load a checkpoint and evaluate on the held-out val set.

    Returns dict with: val_loss, perplexity, nats_per_byte, token_accuracy,
    step, and bytes_per_token used.
    """
    from inference.generate import load_model

    if device is None:
        device = torch.device("cpu")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model, cfg = load_model(ckpt_path, device)
    model.eval()

    # step lives at the top level of the checkpoint, not inside cfg
    step = ckpt.get("step", -1)

    # Determine vocab type
    tok = None
    tok_path = cfg.get("tokenizer")
    if tok_path:
        from model.bpe import BPETokenizer
        tok = BPETokenizer.load(tok_path)

    ctx = cfg["context_length"]
    vocab = cfg["vocab_size"]

    val_data, measured_bpt = _load_val_data(tok, ctx, device)
    if val_data is None:
        return {"error": "val data not found", "step": cfg.get("step", -1)}

    # Allow override
    if bytes_per_token is not None:
        measured_bpt = bytes_per_token

    pos = [0]
    total_loss = 0.0
    total_acc = 0.0
    for _ in range(val_batches):
        x, y = _get_batch(val_data, 64, ctx, vocab, device, pos)
        with torch.no_grad():
            logits = model(x)
        total_loss += F.cross_entropy(
            logits.reshape(-1, vocab), y.reshape(-1)
        ).item()
        total_acc += token_accuracy(logits, y)

    val_loss = total_loss / val_batches
    acc = total_acc / val_batches
    ppl = math.exp(val_loss)
    npb = nats_per_byte(val_loss, measured_bpt)

    return {
        "label": os.path.splitext(os.path.basename(ckpt_path))[0],
        "step": step,
        "val_loss": val_loss,
        "perplexity": ppl,
        "nats_per_byte": npb,
        "token_accuracy": acc,
        "bytes_per_token": measured_bpt,
    }


def eval_report(ckpt_paths: list[str], val_batches: int = 20,
                device: torch.device = None) -> list[dict]:
    """Evaluate multiple checkpoints and return sorted results.

    Sorted by nats_per_byte (lower = better) — the only fair comparison.
    """
    if device is None:
        device = torch.device("cpu")

    results = []
    for path in ckpt_paths:
        r = eval_checkpoint(path, val_batches=val_batches, device=device)
        if "error" not in r:
            results.append(r)

    results.sort(key=lambda r: r["nats_per_byte"])
    return results


def print_report(results: list[dict]):
    """Pretty-print an eval report as a table."""
    if not results:
        print("No results to display.")
        return

    print(f"\n{'label':<45} {'step':>6} {'val_loss':>9} {'ppl':>8} "
          f"{'nats/B':>8} {'acc':>6}")
    print("-" * 85)
    for r in results:
        print(f"{r['label']:<45} {r['step']:>6} {r['val_loss']:>9.4f} "
              f"{r['perplexity']:>8.2f} {r['nats_per_byte']:>8.4f} "
              f"{r['token_accuracy']:>6.3f}")
    print()
