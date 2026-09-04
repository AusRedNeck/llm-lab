"""Probability helpers: top-k + distribution summary for the viewer."""


def top_k(probs, k: int = 5) -> list[tuple[int, float]]:
    """Top-k token IDs + probs at the LAST position. probs: [B, T, V]."""
    last = probs[0, -1]
    vals, idx = last.topk(k)
    return [(int(i), float(v)) for i, v in zip(idx.tolist(), vals.tolist())]


def summary(probs) -> dict:
    """One-line distribution health: max prob, entropy, top-1 id."""
    import torch
    last = probs[0, -1].float()
    return {
        "top1_id": int(last.argmax()),
        "top1_prob": float(last.max()),
        "entropy": float(-(last * (last + 1e-12).log()).sum()),
    }
