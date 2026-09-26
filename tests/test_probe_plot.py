import json
from viz.probe_plot import render_probe_plot


def _row(step, ent=2.0, loc=0.5, rank=20.0, top1=0.3, ce=3.0):
    L = 4
    return {"ckpt": "run_step%d" % step, "step": step,
            "attn_entropy": [ent] * L, "locality": [loc] * L,
            "eff_rank": [rank] * L, "layer_norms": [1.0] * L,
            "top1_p": top1, "ce_nats": ce, "layers": 3}


def test_probe_plot_renders_curve_per_metric():
    rows = [_row(500), _row(2500, ent=1.5, top1=0.7), _row(5000, ent=2.1, top1=0.6)]
    html = render_probe_plot(rows)
    for token in ["step500", "step2500", "step5000", "entropy", "locality",
                  "eff_rank", "top1", "<canvas", "5000"]:
        assert token in html, token
