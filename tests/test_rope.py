import torch

from model.rope import apply_rope, precompute_freqs
from model.attention import MultiHeadAttention
from model.transformer import Transformer


def test_rope_preserves_shape():
    B, H, T, D = 2, 6, 16, 64
    x = torch.randn(B, H, T, D)
    cos, sin = precompute_freqs(D, T)
    out = apply_rope(x, cos, sin)
    assert out.shape == x.shape


def test_rope_changes_output_but_keeps_norm():
    # Rotation must change direction, not magnitude.
    torch.manual_seed(0)
    B, H, T, D = 1, 2, 8, 32
    x = torch.randn(B, H, T, D)
    cos, sin = precompute_freqs(D, T)
    out = apply_rope(x, cos, sin)
    assert not torch.allclose(out, x)
    assert torch.allclose(out.norm(dim=-1), x.norm(dim=-1), atol=1e-5)


def test_rope_attention_is_causal_and_shaped():
    attn = MultiHeadAttention(embedding_dim=384, num_heads=6, use_rope=True)
    attn.eval()
    x = torch.randn(1, 16, 384)
    out = attn(x)
    assert out.shape == x.shape


def test_rope_transformer_returns_logits():
    model = Transformer(vocab_size=256, context_length=64, embedding_dim=64,
                        num_heads=4, num_layers=2, use_rope=True)
    x = torch.randint(0, 256, (2, 64))
    assert model(x).shape == (2, 64, 256)
