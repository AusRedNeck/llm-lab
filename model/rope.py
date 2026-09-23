"""Experiment 003 — RoPE (Rotary Position Embedding).

Why: your current model adds a learned absolute position vector to each
token (model/transformer.py InputEmbedding). That works to ctx 256 but
memorizes positions — it can't stretch past training length and wastes
capacity learning "position 47" separately for every layer.

RoPE instead rotates each query/key vector by an angle proportional to
its position. Relative distance falls out of the dot product naturally,
no position table to learn, extrapolates better.

Shape: head_dim must be even (we rotate pairs).

Partial RoPE (GPT-NeoX style): rotary_pct controls what fraction of each
head's features get rotated. Pythia uses 0.25 — most features stay raw,
the network learns positional dependence rather than having it forced from
step 1. Set rotary_pct < 1.0 to slice the tensor before rotation;
everything past the slice boundary passes through unchanged.
"""
import torch


def precompute_freqs(head_dim: int, max_seq: int, theta: float = 10000.0,
                     device=None, dtype=torch.float32):
    assert head_dim % 2 == 0, "RoPE needs even head_dim"
    # One frequency per pair: [head_dim/2]
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device,
                                         dtype=dtype) / head_dim))
    pos = torch.arange(max_seq, device=device, dtype=dtype)
    angles = torch.outer(pos, freqs)  # [T, head_dim/2]
    return torch.cos(angles), torch.sin(angles)  # each [T, D/2]


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
               rotary_slice: int = -1) -> torch.Tensor:
    """x: [B, H, T, head_dim] -> rotated same shape. cos/sin: [T, D/2].

    If rotary_slice > 0, only the first rotary_slice features (pairs only —
    must be even) are rotated; the rest pass through as-is. This implements
    GPT-NeoX partial RoPE (Pythia uses 0.25 × head_dim).
    """
    # Split into pairs: (x1, x2) rotate together.
    x1 = x[..., 0::2]  # [B, H, T, D/2]
    x2 = x[..., 1::2]

    if rotary_slice <= 0:
        # Full RoPE: rotate everything, interleave back.
        cos = cos[:x.shape[2], :].unsqueeze(0).unsqueeze(0).to(x.dtype)
        sin = sin[:x.shape[2], :].unsqueeze(0).unsqueeze(0).to(x.dtype)
        out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return out.flatten(-2)

    # Partial RoPE: split x1/x2 into rotated + untouched halves.
    half = rotary_slice // 2  # number of PAIRS to rotate (input is pair-count)
    assert rotary_slice % 2 == 0, f"rotary_slice must be even, got {rotary_slice}"
    assert half <= x1.shape[-1], f"rotary_slice={rotary_slice} exceeds feature dims ({x1.shape[-1]} pairs)"

    cos = cos[:x.shape[2], :].unsqueeze(0).unsqueeze(0).to(x.dtype)
    sin = sin[:x.shape[2], :].unsqueeze(0).unsqueeze(0).to(x.dtype)

    # Rotated halves
    rot1 = x1[..., :half]
    rot2 = x2[..., :half]
    rotated = torch.stack([rot1 * cos[..., :half] - rot2 * sin[..., :half],
                           rot1 * sin[..., :half] + rot2 * cos[..., :half]], dim=-1)

    # Untouched halves (pass through)
    unrot1 = x1[..., half:]
    unrot2 = x2[..., half:]
    untouched = torch.stack([unrot1, unrot2], dim=-1)

    # Interleave pair-wise: concat along pairs axis (-2), then flatten back.
    out = torch.cat([rotated, untouched], dim=-2)  # [B,H,T,half+rem,2]
    return out.flatten(-2)  # [B,H,T,D]
