import torch.nn as nn

from model.attention import MultiHeadAttention


class TransformerBlock(nn.Module):
    # One layer of thinking: listen to context, then think on your own.
    def __init__(self, embedding_dim: int, num_heads: int, use_rope: bool = False,
                 rotary_pct: float = 1.0, dropout: float = 0.0,
                 parallel_residual: bool = True):
        super().__init__()
        # True  = GPT-NeoX/Pythia (use_parallel_residual): both branches see
        #         the same pre-norm'd input.
        # False = GPT-2 legacy: FFN consumes the post-attention hidden state.
        self.parallel_residual = parallel_residual

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
        # Branches are dropout-gated, then summed into x.
        # parallel_residual=True  -> GPT-NeoX/Pythia (use_parallel_residual):
        #   both branches see the SAME pre-norm'd input, giving independent
        #   additive signals instead of cascade coupling.
        # parallel_residual=False -> GPT-2 legacy: the FFN consumes the
        #   post-attention hidden state.
        attn_result = self.attention(self.norm1(x), return_weights=return_weights)
        if return_weights:
            attn_out, attn_weights = attn_result
        else:
            attn_out = attn_result
        # Apply the residual dropout to the tensor, not the (tensor, weights)
        # capture tuple. The attention module already applies weight dropout.
        attn_out = self.attn_drop(attn_out)
        ffn_in = x if self.parallel_residual else x + attn_out
        ff_out = self.ffn_drop(self.feed_forward(self.norm2(ffn_in)))
        output = x + attn_out + ff_out  # Pythia-style: no post-sum norm

        if return_weights:
            return output, attn_weights
        return output
