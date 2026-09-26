import torch
from model.transformer import Transformer
from viz.replay import render_replay


def _tiny():
    torch.manual_seed(0)
    m = Transformer(vocab_size=64, context_length=16, embedding_dim=32,
                    num_heads=4, num_layers=2)
    m.eval()
    return m


def test_replay_renders_one_column_per_ckpt():
    m1, m2 = _tiny(), _tiny()
    x = torch.randint(0, 64, (1, 6))
    labels = ["t%d" % i for i in range(6)]
    html = render_replay([(m1, "step500"), (m2, "step2500")], x,
                         labels=labels, layer=0, head=0, k=5)
    assert "step500" in html and "step2500" in html
    assert "top 5" in html.lower()
    assert "<table" in html
    assert "entropy" in html.lower()
