import torch

from model.transformer import Transformer
from viz.hooks import capture_forward
from viz.attention import is_causal, text_heatmap
from viz.probabilities import top_k, summary


def test_capture_matches_normal_forward():
    torch.manual_seed(0)
    model = Transformer(vocab_size=64, context_length=16, embedding_dim=32,
                        num_heads=4, num_layers=2)
    model.eval()
    x = torch.randint(0, 64, (1, 8))
    with torch.no_grad():
        plain = model(x)
    cap = capture_forward(model, x)
    assert torch.allclose(plain, cap["logits"])
    assert cap["embeddings"].shape == (1, 8, 32)
    assert len(cap["layer_hiddens"]) == 2
    assert len(cap["layer_attention"]) == 2
    assert cap["layer_attention"][0].shape == (1, 4, 8, 8)


def test_captured_attention_is_causal():
    torch.manual_seed(1)
    model = Transformer(vocab_size=64, context_length=16, embedding_dim=32,
                        num_heads=4, num_layers=2)
    x = torch.randint(0, 64, (1, 8))
    cap = capture_forward(model, x)
    for layer_w in cap["layer_attention"]:
        assert is_causal(layer_w[0, 0])


def test_topk_and_summary():
    torch.manual_seed(2)
    model = Transformer(vocab_size=64, context_length=16, embedding_dim=32,
                        num_heads=4, num_layers=2)
    x = torch.randint(0, 64, (1, 8))
    cap = capture_forward(model, x)
    tk = top_k(cap["probs"], k=5)
    assert len(tk) == 5
    s = summary(cap["probs"])
    assert 0.0 <= s["top1_prob"] <= 1.0
