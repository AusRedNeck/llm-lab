import torch
from model.transformer import Transformer
from viz.xray import render_xray


def test_xray_renders_tokens_topk_and_heatmap():
    torch.manual_seed(0)
    model = Transformer(vocab_size=64, context_length=16, embedding_dim=32,
                        num_heads=4, num_layers=2)
    model.eval()
    x = torch.randint(0, 64, (1, 6))
    labels = ["t%d" % i for i in range(6)]
    html = render_xray(model, x, labels=labels, layer=0, head=0, k=5)
    assert "top-5" in html.lower() or "top_5" in html or "top 5" in html.lower()
    for lab in labels:
        assert lab in html
    assert "<table" in html  # heatmap grid
    assert "layer" in html.lower() and "head" in html.lower()
