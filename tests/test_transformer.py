import torch

from model.transformer import TokenEmbedding, PositionalEmbedding, InputEmbedding

def test_embedding_output_shape():
    embedding = TokenEmbedding(vocab_size=256, embedding_dim=128)

    tokens = torch.tensor([[1, 2, 3, 4]])

    output = embedding(tokens)

    assert output.shape == (1, 4, 128)

def test_positional_embedding_output_shape():
    embedding = PositionalEmbedding(context_length=128, embedding_dim=128)

    positions = torch.arange(4)

    output = embedding(positions)

    assert output.shape == (4, 128)

def test_input_embedding_output_shape():
    embedding = InputEmbedding(
        vocab_size=256,
        context_length=128,
        embedding_dim=128,
    )

    tokens = torch.tensor([[1, 2, 3, 4]])

    output = embedding(tokens)

    assert output.shape == (1, 4, 128)