import torch.nn as nn

from model.attention import MultiHeadAttention


class TransformerBlock(nn.Module):
    # One layer of thinking: listen to context, then think on your own.
    def __init__(self, embedding_dim: int, num_heads: int, use_rope: bool = False,
                 rotary_pct: float = 1.0, dropout: float = 0.0):
        super().__init__()

        self.attention = MultiHeadAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            use_rope=use_rope,
            rotary_pct=rotary_pct,
            dropout=dropout,
        )
        self.attn_drop = nn.Dropout(dropout)
        self.ffn_drop = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.GELU(),
            nn.Linear(embedding_dim * 4, embedding_dim),
        )

    def forward(self, x, return_weights: bool = False):
        # Parallel residual (GPT-NeoX style): both branches see the SAME
        # pre-norm'd input, each is dropout-gated, then summed into x.
        # This matches EleutherAI/pythia configs (use_parallel_residual=true)
        # and gives independent additive signals instead of cascade coupling.
        attn_out = self.attn_drop(self.attention(self.norm1(x),
                  return_weights=return_weights))
        if return_weights:
            attn_out, attn_weights = attn_out
        ff_out = self.ffn_drop(self.feed_forward(self.norm2(x)))
        output = x + attn_out + ff_out  # Pythia-style: no post-sum norm

        if return_weights:
            return output, attn_weights
        return output
