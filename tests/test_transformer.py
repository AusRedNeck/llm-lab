import torch

from model.transformer import TokenEmbedding

def test_embedding_output_shape():
    embedding = TokenEmbedding(vocab_size=256, embedding_dim=128)

    tokens = torch.tensor([[1, 2, 3, 4]])

    output = embedding(tokens)

    assert output.shape == (1, 4, 128)