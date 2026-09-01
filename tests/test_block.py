import torch

from model.block import TransformerBlock


def test_transformer_block_preserves_shape():
    block = TransformerBlock(
        embedding_dim = 8,
        num_heads = 2,
    )

    x = torch.randn(2, 4, 8)

    output = block(x)

    assert output.shape == x.shape

def test_transformer_block_has_layer_norm():
    block = TransformerBlock(
        embedding_dim = 8,
        num_heads=2,
    )

    assert isinstance(block.norm1, torch.nn.LayerNorm)

def test_transformer_block_has_feed_forward():
    block = TransformerBlock(
        embedding_dim=8,
        num_heads=2,
    )

    assert isinstance(block.feed_forward, torch.nn.Sequential)

def test_transformer_block_feed_forward_preserves_shape():
    block = TransformerBlock(
        embedding_dim=8,
        num_heads=2,
    )

    x = torch.randn(2, 4, 8)
    output = block(x)

    assert output.shape == x.shape


def test_transformer_block_has_second_layer_norm():
    block = TransformerBlock(
        embedding_dim=8,
        num_heads=2,
    )

    assert isinstance(block.norm2, torch.nn.LayerNorm)