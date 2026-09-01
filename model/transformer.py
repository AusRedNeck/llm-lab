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
    ):

        super().__init__()

        self.token_embedding = TokenEmbedding(vocab_size, embedding_dim)
        self.position_embedding = PositionalEmbedding(context_length, embedding_dim)

    def forward(self, tokens):
        positions = torch.arange(tokens.shape[1], device=tokens.device)

        token_vectors = self.token_embedding(tokens)
        position_vectors = self.position_embedding(positions)

        return token_vectors + position_vectors





class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        embedding_dim: int,
        num_heads: int,
        num_layers: int,
    ):
        super().__init__()

        # Convert token IDs into vectors that the Transformer can process.
        self.embedding = InputEmbedding(
            vocab_size=vocab_size,
            context_length=context_length,
            embedding_dim=embedding_dim,
        )

        # Stack multiple Transformer blocks.
        # Each block refines the representation produced by the previous one.
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
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
        )

    def forward(self, tokens):
        # [B, T] → [B, T, D]
        x = self.embedding(tokens)

        # Run the representation through every Transformer block.
        for block in self.blocks:
            x = block(x)

        # Normalize the final representation.
        x = self.norm(x)

        # [B, T, D] → [B, T, vocab_size]
        logits = self.lm_head(x)

        return logits