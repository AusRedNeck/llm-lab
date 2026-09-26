#!/usr/bin/env python3
"""viz/xray.py -- glass-box X-ray viewer, Phase 1 milestone.

Input tokens -> attention heatmap -> top-5 probs -> predicted token.
Zero-deps: one self-contained HTML, file:// works.

Library:
    from viz.xray import render_xray
    html = render_xray(model, token_ids, labels=[...], layer=0, head=0, k=5)

CLI (offline checkpoint X-ray):
    python -m viz.xray --ckpt checkpoints/<run>.pt --prompt "Once upon a time" \
        --layer 0 --head 0 --out viz/xray.html
"""
from __future__ import annotations

import argparse
import html as htmlmod
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def render_xray(model, token_ids, labels=None, layer: int = 0, head: int = 0,
                k: int = 5) -> str:
    """Render one layer/head X-ray as standalone HTML.

    model: Transformer with forward_with_capture. token_ids: [1, T].
    labels: per-position strings (decoded tokens). Falls back to t0..tN.
    Returns full HTML doc string.
    """
    import torch
    from viz.hooks import capture_forward
    from viz.probabilities import top_k, summary

    model.eval()
    T = token_ids.shape[1]
    labels = labels or ["t%d" % i for i in range(T)]
    cap = capture_forward(model, token_ids)
    n_layers = len(cap["layer_attention"])
    n_heads = cap["layer_attention"][0].shape[1]
    layer = max(0, min(layer, n_layers - 1))
    head = max(0, min(head, n_heads - 1))

    # Full attention cube inlined so layer/head pickers work without rerun.
    attn_all = [[[round(float(v), 4) for v in row]
                 for row in cap["layer_attention"][li][0, h].tolist()]
                for li in range(n_layers) for h in range(n_heads)]
    tk = top_k(cap["probs"], k=k)
    summ = summary(cap["probs"])
    pred = summ["top1_id"]

    # Heatmap cell color: alpha-scaled orange on dark.
    def cell(v: float) -> str:
        a = max(0.0, min(1.0, v))
        return ('<td style="background:rgba(249,115,22,%.2f)">%.2f</td>' % (a, v))

    head_opt = lambda li: "".join(
        '<option value="%d"%s>head %d</option>' % (h, " selected" if h == head else "", h)
        for h in range(n_heads))
    layer_opts = "".join(
        '<option value="%d"%s>layer %d</option>' % (li, " selected" if li == layer else "", li)
        for li in range(n_layers))
    # Real top-5 rows: id + prob.
    top5_rows = "".join(
        "<tr><td>%d</td><td>%.4f</td>%s</tr>"
        % (tid, p, "<td><b>PREDICTED</b></td>" if tid == pred else "<td></td>")
        for tid, p in tk)
    header = "<tr><th></th>" + "".join(
        "<th>%s</th>" % htmlmod.escape(l) for l in labels) + "</tr>"

    return """<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>x-ray layer %(layer)d head %(head)d</title>
<style>
:root{color-scheme:dark}
body{background:#0b0e14;color:#e6edf3;font:14px/1.5 system-ui,sans-serif;margin:0;padding:16px}
h1{font-size:17px;margin:0 0 2px}.sub{color:#8b949e;font-size:12px;margin-bottom:10px}
.row{display:flex;gap:10px;margin:8px 0;align-items:center;flex-wrap:wrap}
select{background:#1c2330;color:#e6edf3;border:1px solid #30363d;border-radius:6px;padding:4px 10px}
table{border-collapse:collapse;font-size:12px}th,td{border:1px solid #21262d;padding:3px 7px;text-align:center}
th{background:#161b22;max-width:90px;overflow:hidden;text-overflow:ellipsis}
#top5 td{text-align:left}.pred{color:#a3e635;font-weight:600}
pre{background:#11161f;border:1px solid #30363d;border-radius:8px;padding:10px;white-space:pre-wrap;font:12px/1.6 ui-monospace,monospace}
</style></head><body>
<h1>glass-box x-ray &mdash; layer %(layer)d head %(head)d</h1>
<div class="sub">top-5 at last position &middot; predicted id <b>%(pred)d</b> p=<b>%(topp).4f</b> entropy=<b>%(ent).3f</b> &middot; causal upper triangle must read ~0.00</div>
<div class="row">
<label>layer <select id="layer" onchange="pick()">%(layers)s</select></label>
<label>head <select id="head" onchange="pick()">%(heads)s</select></label>
<span class="sub">%(n_layers)d layers x %(nh)d heads &middot; %(ntok)d tokens</span>
</div>
<table id="map"><thead>%(header)s</thead><tbody id="mapbody"></tbody></table>
<h1 style="margin-top:14px">top 5 next-token</h1>
<table id="top5"><tr><th>id</th><th>prob</th><th></th></tr>%(top5)s</table>
<h1 style="margin-top:14px">input tokens</h1>
<pre>%(toks)s</pre>
<script>
const ATTN=%(attn)s, LABELS=%(labs)s;
function pick(){const li=+document.getElementById('layer').value,
h=+document.getElementById('head').value,
W=ATTN[(li*%(nh)d)+h],tb=document.getElementById('mapbody');tb.innerHTML='';
for(let i=0;i<W.length;i++){const tr=document.createElement('tr');
const th=document.createElement('th');th.textContent=LABELS[i]??('t'+i);tr.appendChild(th);
for(let j=0;j<W[i].length;j++){const td=document.createElement('td');
td.textContent=W[i][j].toFixed(2);
td.style.background=`rgba(249,115,22,${Math.max(0,Math.min(1,W[i][j])).toFixed(2)})`;
tr.appendChild(td);}tb.appendChild(tr);}
document.title=`x-ray layer ${li} head ${h}`;}
pick();
</script></body></html>""" % {
        "layer": layer, "head": head, "pred": pred,
        "topp": summ["top1_prob"], "ent": summ["entropy"],
        "layers": layer_opts, "heads": head_opt(layer),
        "nh": n_heads, "n_layers": n_layers, "ntok": T,
        "header": header, "top5": top5_rows,
        "toks": htmlmod.escape(" ".join("[%d]%s" % (i, l) for i, l in enumerate(labels))),
        "attn": json.dumps(attn_all), "labs": json.dumps(labels),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--prompt", default="Once upon a time")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--head", type=int, default=0)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--rotary-pct", type=float, default=1.0)
    ap.add_argument("--out", default="viz/xray.html")
    ns = ap.parse_args()

    import torch
    from model.bpe import BPETokenizer
    from model.transformer import Transformer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ns.ckpt, map_location=device)
    c = ckpt.get("cfg", {})
    sd = ckpt["model"] if "model" in ckpt else ckpt
    model = Transformer(
        vocab_size=c["vocab_size"], context_length=c["context_length"],
        embedding_dim=c["embedding_dim"], num_heads=c["num_heads"],
        num_layers=c["num_layers"], use_rope=c.get("use_rope", True),
        rotary_pct=ns.rotary_pct, dropout=0.0).to(device)
    model.load_state_dict(sd, strict=True)

    tok_path = ns.tokenizer or c.get("tokenizer")
    tok = BPETokenizer.load(tok_path) if tok_path else None
    if tok is not None:
        ids = tok.encode(ns.prompt)
        labels = [tok.decode([i]) for i in ids]
    else:
        ids = list(ns.prompt.encode("utf-8"))
        labels = list(ns.prompt)
    ctx = c["context_length"]
    ids = ids[-ctx:]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    html = render_xray(model, x, labels=labels, layer=ns.layer,
                       head=ns.head, k=ns.topk)
    open(ns.out, "w", encoding="utf-8").write(html)
    print("wrote %s (%d tokens)" % (ns.out, len(ids)))


if __name__ == "__main__":
    main()
