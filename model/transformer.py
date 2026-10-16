import torch
import torch.nn as nn

from model.block import TransformerBlock

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, tokens):
        return self.embedding(tokens)

class PositionalEmbedding(nn.Module):
    def __init__(self, context_length: int, embedding_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(context_length, embedding_dim)

    def forward(self, positions):
        return self.embedding(positions) 

class InputEmbedding(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            context_length: int,
            embedding_dim: int,
            use_learned_position_embeddings: bool = True,
    ):

        super().__init__()

        self.token_embedding = TokenEmbedding(vocab_size, embedding_dim)
        self.position_embedding = (
            PositionalEmbedding(context_length, embedding_dim)
            if use_learned_position_embeddings else None
        )

    def forward(self, tokens):
        token_vectors = self.token_embedding(tokens)
        if self.position_embedding is None:
            return token_vectors
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        return token_vectors + self.position_embedding(positions)





class Transformer(nn.Module):
    # The whole engine: embeddings in, stack of blocks, back to word scores.
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        embedding_dim: int,
        num_heads: int,
        num_layers: int,
        use_rope: bool = False,
        rotary_pct: float = 1.0,
        dropout: float = 0.0,
        parallel_residual: bool = True,
        use_learned_position_embeddings: bool = True,
        lm_head_bias: bool = True,
    ):
        super().__init__()

        self.use_rope = use_rope
        self.rotary_pct = rotary_pct
        self.parallel_residual = parallel_residual
        self.use_learned_position_embeddings = use_learned_position_embeddings
        self.lm_head_bias = lm_head_bias
        # Embedding dropout: same vectors, occasional blind spots in train.
        self.emb_drop = nn.Dropout(dropout)
        # Convert token IDs into vectors that the Transformer can process.
        self.embedding = InputEmbedding(
            vocab_size=vocab_size,
            context_length=context_length,
            embedding_dim=embedding_dim,
            use_learned_position_embeddings=use_learned_position_embeddings,
        )

        # Stack multiple Transformer blocks.
        # Each block refines the representation produced by the previous one.
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    use_rope=use_rope,
                    rotary_pct=rotary_pct,
                    dropout=dropout,
                    parallel_residual=parallel_residual,
                )
                for _ in range(num_layers)
            ]
        )

        # Normalize the final representation before producing token scores.
        self.norm = nn.LayerNorm(embedding_dim)

        # Convert each final token representation into one score
        # for every possible token in the vocabulary.
        self.lm_head = nn.Linear(
            embedding_dim,
            vocab_size,
            bias=lm_head_bias,
        )

    @classmethod
    def from_config(cls, config: dict):
        """Build from preset/checkpoint config; missing new keys stay legacy-safe."""
        return cls(
            vocab_size=config["vocab_size"],
            context_length=config["context_length"],
            embedding_dim=config["embedding_dim"],
            num_heads=config["num_heads"],
            num_layers=config["num_layers"],
            use_rope=config.get("use_rope", False),
            rotary_pct=config.get("rotary_pct", 1.0),
            dropout=config.get("dropout", 0.0),
            parallel_residual=config.get("parallel_residual", True),
            use_learned_position_embeddings=config.get(
                "use_learned_position_embeddings", True),
            lm_head_bias=config.get("lm_head_bias", True),
        )

    def forward(self, tokens):
        # [B, T] → [B, T, D]
        x = self.emb_drop(self.embedding(tokens))

        # Run the representation through every Transformer block.
        for block in self.blocks:
            x = block(x)

        # Normalize the final representation.
        x = self.norm(x)

        # [B, T, D] → [B, T, vocab_size]
        logits = self.lm_head(x)

        return logits

    def forward_with_capture(self, tokens):
        """Cross-section connector (Phase 1 glass box).

        Normal forward() is untouched. This parallel path returns the
        same logits PLUS every stage the X-ray view needs:
        embeddings, per-layer hidden states, per-layer attention weights.
        The viz layer reads this dict — never model internals directly.
        """
        import torch.nn.functional as F

        captures = {"layer_hiddens": [], "layer_attention": [], "layer_details": []}
        x = self.emb_drop(self.embedding(tokens))
        captures["embeddings"] = x.detach()
        for block in self.blocks:
            x, weights, details = block(x, return_weights=True, return_capture=True)
            captures["layer_hiddens"].append(x.detach())
            captures["layer_attention"].append(weights.detach())
            captures["layer_details"].append(details)

        x = self.norm(x)
        logits = self.lm_head(x)
        captures["logits"] = logits.detach()
        captures["probs"] = F.softmax(logits, dim=-1).detach()
        return logits, captures