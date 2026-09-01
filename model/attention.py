import torch
import torch.nn as nn

class CausalSelfAttention(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()

        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)
        self.output = nn.Linear(embedding_dim, embedding_dim)


    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # compare every query against every key
        scores = q @ k.transpose(-2, -1)

        # Scale by sqrt(d) so dot products don't grow too large as the embedding dimension increases
        scores = scores / (self.query.out_features ** 0.5)

        sequence_length = x.shape[1]

        # A token may attend to itself and anything before it, but never to a token that comes later in the sequence
        mask = torch.tril(
            torch.ones(sequence_length, sequence_length, device=x.device)
        )

        # Turn forbidden future positions into zero attention after softmax
        scores = scores.masked_fill(mask == 0, float("-inf"))

        # Convert raw compatibility scores into attention weights.
        weights = torch.softmax(scores, dim=-1)


        # Combine the value vectors according to those weights
        output = weights @ v        

        # mix the attention result back into the model's ebedding space.
        output = self.output(output)

        return output

class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int):
        super().__init__()

        # Every head gets an equal-sized slice of the embedding dimension.
        if embedding_dim % num_heads != 0:
            raise ValueError(
                "embedding_dim must be divisible by num_heads"
            )

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        # Each projection produces the full embedding dimension
        # We split that dimension into separate heads in forward().
        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)

        self.output = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x):
        batch_size, sequence_length, _ = x.shape

        # Project the input into queries, keys and values.
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # Split the embedding dimension across the attention heads.
        # [B, T, D] -> [B, T, H, head_dim]
        q = q.view(batch_size, sequence_length, self.num_heads, self.head_dim)
        k = k.view(batch_size, sequence_length, self.num_heads, self.head_dim)
        v = v.view(batch_size, sequence_length, self.num_heads, self.head_dim)

        # Put the head dimension before sequence length so each
        # attention head can operate independently
        # [B, T, H, head_dim] -> [B, H, T, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Each head compares every query against every key.
        # [B, H, T, head_dim] @ [B, H, head_dim, T]
        # -> [B, H, T, T]
        scores = q @ k.transpose(-2, -1)

        # Scale the scores by sqrt(head_dim).
        scores = scores / (self.head_dim ** 0.5)

        # Each head gets the same causal constraint:
        # a token can attend to itself and earlier tokens,
        # but never to future tokens.
        mask = torch.tril(
            torch.ones(
             sequence_length,
                sequence_length,
                device=x.device,
            )
        )

        scores = scores.masked_fill(mask == 0, float("-inf"))

        # Convert scores into probabilities along the key/sequence dimension.
        weights = torch.softmax(scores, dim=-1)

        # Combine value vectors according to the attention weights.
        # [B, H, T, T] @ [B, H, T, head_dim]
        # -> [B, H, T, head_dim]
        output = weights @ v

        # Put the heads back together.
        # [B, H, T, head_dim] -> [B, T, H, head_dim]
        output = output.transpose(1, 2).contiguous()

        # Flatten H and head_dim back into the original embedding dimension.
        # [B, T, H, head_dim] -> [B, T, D]
        output = output.view(
            batch_size,
            sequence_length,
            self.embedding_dim,
        )

        # Let the model learn how to combine information across heads.
        output = self.output(output)

        return output