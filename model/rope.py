"""Experiment 003 — RoPE (Rotary Position Embedding).

Why: your current model adds a learned absolute position vector to each
token (model/transformer.py InputEmbedding). That works to ctx 256 but
memorizes positions — it can't stretch past training length and wastes
capacity learning "position 47" separately for every layer.

RoPE instead rotates each query/key vector by an angle proportional to
its position. Relative distance falls out of the dot product naturally,
no position table to learn, extrapolates better.

Shape: head_dim must be even (we rotate pairs).
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


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [B, H, T, head_dim] -> rotated same shape. cos/sin: [T, D/2]."""
    # Split each head vector into pairs: (x1, x2) rotate together.
    x1 = x[..., 0::2]  # [B, H, T, D/2]
    x2 = x[..., 1::2]
    # Broadcast cos/sin over batch + heads.
    cos = cos[:x.shape[2], :].unsqueeze(0).unsqueeze(0).to(x.dtype)
    sin = sin[:x.shape[2], :].unsqueeze(0).unsqueeze(0).to(x.dtype)
    # Rotation: [x1*cos - x2*sin, x1*sin + x2*cos], interleaved back.
    out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return out.flatten(-2)
