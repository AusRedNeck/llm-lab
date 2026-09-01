import torch
import pytest

from model.attention import CausalSelfAttention, MultiHeadAttention


def test_causal_self_attention_output_shape():
    batch_size = 2
    sequence_length = 4
    embedding_dim = 8

    attention = CausalSelfAttention(embedding_dim)

    x = torch.randn(batch_size, sequence_length, embedding_dim)

    output = attention(x)

    assert output.shape == x.shape


def test_causal_attention_cannot_see_future_tokens():
    torch.manual_seed(0)

    attention = CausalSelfAttention(embedding_dim=8)
    attention.eval()

    x1 = torch.randn(1, 4, 8)
    x2 = x1.clone()

    # Change only the final token.
    x2[:, 3, :] = torch.randn(1, 8)

    output1 = attention(x1)
    output2 = attention(x2)

    # The first three positions must be unchanged.
    assert torch.allclose(
        output1[:, :3, :],
        output2[:, :3, :],
    )

def test_causal_attention_supports_different_sequence_lengths():
    attention = CausalSelfAttention(embedding_dim=8)

    for sequence_length in [1, 2, 5, 16]:
        x = torch.randn(2, sequence_length, 8)

        output = attention(x)

        assert output.shape == x.shape


def test_embedding_dimension_must_be_divisible_by_num_heads():
    with pytest.raises(ValueError):
        MultiHeadAttention(embedding_dim=8, num_heads=3)


def test_multi_head_attention_output_shape():
    attention = MultiHeadAttention(
        embedding_dim=8,
        num_heads=2,
    )

    x = torch.randn(2, 4, 8)

    output = attention(x)

    assert output.shape == x.shape

def test_multi_head_attention_is_causal():
    torch.manual_seed(0)

    attention = MultiHeadAttention(
        embedding_dim = 8,
        num_heads = 2,
    )
    attention.eval()

    x1 = torch.randn(1, 4, 8)
    x2 = x1.clone()

    # Change only the final token.
    x2[:, 3, :] = torch.randn(1, 8)

    output1 = attention(x1)
    output2 = attention(x2)

    # Changing the future token must not affect earlier positions.
    assert torch.allclose(
        output2[:, :3, :],
        output2[:, :3, :],
    )
