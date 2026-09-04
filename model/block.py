import torch.nn as nn

from model.attention import MultiHeadAttention


class TransformerBlock(nn.Module):
    # One layer of thinking: listen to context, then think on your own.
    def __init__(self, embedding_dim: int, num_heads: int, use_rope: bool = False):
        super().__init__()

        self.attention = MultiHeadAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            use_rope=use_rope,
        )

        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.GELU(),
            nn.Linear(embedding_dim * 4, embedding_dim),
        )
            

    def forward(self, x, return_weights: bool = False):
        # Attention lets each token gather information from
        # the earlier tokens in the sequence.
        if return_weights:
            attention_output, weights = self.attention(x, return_weights=True)
        else:
            attention_output = self.attention(x)
            weights = None

        # Residual connection preserves the original representation
         # while adding the information produced by attention.
        output = x + attention_output

        # Keep the representation numerically well-behaved.
        output = self.norm1(output)

        # The feed-forward network transforms each token's
        # representation independently after attention has mixed
        # information between tokens.
        feed_forward_output = self.feed_forward(output)

        # Second residual connection preserves the representation
        # while adding the FFN's learned transformation.
        output = output + feed_forward_output

        # Normalize again before passing the result to the next block.
        output = self.norm2(output)

        if return_weights:
            return output, weights
        return output