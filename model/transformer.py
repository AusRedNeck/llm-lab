import torch
import torch.nn as nn

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