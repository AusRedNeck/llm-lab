"""Attention helpers for the X-ray view: causal mask check + text heatmap."""


def is_causal(weights) -> bool:
    """Above-diagonal entries must be ~zero (can't see the future)."""
    import torch
    T = weights.shape[-1]
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    upper = weights[..., mask]
    return bool((upper < 1e-6).all())


def text_heatmap(weights, labels=None, width: int = 40) -> str:
    """Render one [T, T] head as block characters for terminal viewing."""
    import torch
    if hasattr(weights, "detach"):
        weights = weights.detach().float()
    # Average heads if given [H, T, T].
    if weights.dim() == 3:
        weights = weights.mean(dim=0)
    T = weights.shape[0]
    labels = labels or [str(i) for i in range(T)]
    blocks = " ░▒▓█"
    lines = ["     " + " ".join(f"{l:>2}" for l in labels[:T])]
    for i in range(T):
        row = "".join(f"  {blocks[min(4, int(w * 5))]}" for w in weights[i].tolist())
        lines.append(f"{labels[i]:>4} {row}")
    return "\n".join(lines)
