#!/usr/bin/env python3
"""viz/replay.py -- same prompt through N checkpoints, side by side.

Checkpoint replay: random init -> early -> trained, one page.
Each column: top-5 next-token, entropy, attention heatmap (shared
layer/head pickers). Answers "what did training change" at a glance.

Library:
    from viz.replay import render_replay
    html = render_replay([(model_a, "step500"), (model_b, "step2500")], ids)

CLI:
    python -m viz.replay --ckpts ckpts/*step500.pt ckpts/*step2500.pt \\
        ckpts/*step5000.pt --prompt "Once upon a time" --out viz/replay.html
"""
from __future__ import annotations

import argparse
import html as htmlmod
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def render_replay(models, token_ids, labels=None, layer: int = 0,
                  head: int = 0, k: int = 5) -> str:
    """One HTML page, one column per (model, name) pair."""
    from viz.hooks import capture_forward
    from viz.probabilities import top_k, summary

    for m, _ in models:
        m.eval()
    T = token_ids.shape[1]
    labels = labels or ["t%d" % i for i in range(T)]

    cols = []
    for m, name in models:
        cap = capture_forward(m, token_ids)
        nL = len(cap["layer_attention"])
        nH = cap["layer_attention"][0].shape[1]
        li = max(0, min(layer, nL - 1))
        hh = max(0, min(head, nH - 1))
        cube = [[[round(float(v), 4) for v in row]
                 for row in cap["layer_attention"][a][0, b].tolist()]
                for a in range(nL) for b in range(nH)]
        tk = top_k(cap["probs"], k=k)
        sm = summary(cap["probs"])
        cols.append({"name": name, "cube": cube, "topk": tk,
                     "pred": sm["top1_id"], "topp": sm["top1_prob"],
                     "ent": sm["entropy"], "nL": nL, "nH": nH})
    nL, nH = cols[0]["nL"], cols[0]["nH"]
    layer_opts = "".join('<option value="%d">layer %d</option>' % (i, i) for i in range(nL))
    head_opts = "".join('<option value="%d">head %d</option>' % (i, i) for i in range(nH))

    cards = []
    for ci, c in enumerate(cols):
        rows = "".join(
            "<tr><td>%d</td><td>%.4f</td>%s</tr>"
            % (tid, p, "<td><b>TOP</b></td>" if tid == c["pred"] else "<td></td>")
            for tid, p in c["topk"])
        cards.append(
            '<div class="card"><h2>%s</h2>'
            '<div class="sub">pred <b>%d</b> p=<b>%.4f</b> entropy=<b>%.3f</b></div>'
            '<table><tr><th>id</th><th>prob</th><th></th></tr>%s</table>'
            '<table class="map" id="map%d"><thead><tr><th></th>%s</tr></thead>'
            "<tbody></tbody></table></div>"
            % (htmlmod.escape(c["name"]), c["pred"], c["topp"], c["ent"],
               rows, ci, "".join("<th>%s</th>" % htmlmod.escape(l) for l in labels)))

    return """<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>replay x N checkpoints</title>
<style>
:root{color-scheme:dark}
body{background:#0b0e14;color:#e6edf3;font:14px/1.5 system-ui,sans-serif;margin:0;padding:16px}
h1{font-size:17px;margin:0 0 2px}.sub{color:#8b949e;font-size:12px;margin-bottom:8px}
.row{display:flex;gap:10px;margin:8px 0;align-items:center}
select{background:#1c2330;color:#e6edf3;border:1px solid #30363d;border-radius:6px;padding:4px 10px}
.wrap{display:flex;gap:14px;flex-wrap:wrap}
.card{background:#11161f;border:1px solid #30363d;border-radius:8px;padding:12px;min-width:280px}
.card h2{font-size:14px;margin:0 0 4px}
table{border-collapse:collapse;font-size:12px;margin:6px 0}th,td{border:1px solid #21262d;padding:3px 7px;text-align:center}
th{background:#161b22}pre{background:#11161f;border:1px solid #30363d;border-radius:8px;padding:10px;font:12px/1.6 ui-monospace,monospace}
</style></head><body>
<h1>checkpoint replay &mdash; same prompt, N brains</h1>
<div class="sub">top 5 at last position per checkpoint &middot; heatmaps share layer/head pickers &middot; watch entropy fall as it learns</div>
<div class="row">
<label>layer <select id="layer" onchange="pick()">%(layers)s</select></label>
<label>head <select id="head" onchange="pick()">%(heads)s</select></label>
</div>
<div class="wrap">%(cards)s</div>
<h1 style="margin-top:14px">input tokens</h1>
<pre>%(toks)s</pre>
<script>
const CUBES=%(cubes)s, LABELS=%(labs)s, NL=%(nl)d, NH=%(nh)d;
function pick(){const li=+document.getElementById('layer').value,
h=+document.getElementById('head').value;
CUBES.forEach((cube,ci)=>{const W=cube[(li*NH)+h],
tb=document.querySelector('#map'+ci+' tbody');tb.innerHTML='';
for(let i=0;i<W.length;i++){const tr=document.createElement('tr');
const th=document.createElement('th');th.textContent=LABELS[i]??('t'+i);tr.appendChild(th);
for(let j=0;j<W[i].length;j++){const td=document.createElement('td');
td.textContent=W[i][j].toFixed(2);
td.style.background=`rgba(249,115,22,${Math.max(0,Math.min(1,W[i][j])).toFixed(2)})`;
tr.appendChild(td);}tb.appendChild(tr);}});}
pick();
</script></body></html>""" % {
        "layers": layer_opts, "heads": head_opts,
        "cards": "".join(cards),
        "toks": htmlmod.escape(" ".join("[%d]%s" % (i, l) for i, l in enumerate(labels))),
        "cubes": json.dumps([c["cube"] for c in cols]),
        "labs": json.dumps(labels), "nl": nL, "nh": nH,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--prompt", default="Once upon a time")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--head", type=int, default=0)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--rotary-pct", type=float, default=1.0)
    ap.add_argument("--out", default="viz/replay.html")
    ns = ap.parse_args()

    import glob
    import torch
    from model.bpe import BPETokenizer
    from model.transformer import Transformer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    files = []
    for pat in ns.ckpts:
        files.extend(sorted(glob.glob(pat)) or [pat])

    # Tokenizer from first ckpt that records one, else byte fallback.
    first = torch.load(files[0], map_location=device)
    tok_path = ns.tokenizer or first.get("cfg", {}).get("tokenizer")
    tok = BPETokenizer.load(tok_path) if tok_path else None
    if tok is not None:
        ids = tok.encode(ns.prompt)
        labels = [tok.decode([i]) for i in ids]
    else:
        ids = list(ns.prompt.encode("utf-8"))
        labels = list(ns.prompt)

    models = []
    ctx = first.get("cfg", {}).get("context_length", 256)
    for path in files:
        ck = torch.load(path, map_location=device)
        c = ck.get("cfg", {})
        ctx = c.get("context_length", 256)
        m = Transformer(
            vocab_size=c["vocab_size"], context_length=c["context_length"],
            embedding_dim=c["embedding_dim"], num_heads=c["num_heads"],
            num_layers=c["num_layers"], use_rope=c.get("use_rope", True),
            rotary_pct=ns.rotary_pct, dropout=0.0).to(device)
        m.load_state_dict(ck["model"] if "model" in ck else ck)
        name = os.path.basename(path).replace(".pt", "")
        models.append((m, name))
    x = torch.tensor([ids[-ctx:]], dtype=torch.long, device=device)
    lab = labels[-ctx:]
    html = render_replay(models, x, labels=lab, layer=ns.layer,
                         head=ns.head, k=ns.topk)
    open(ns.out, "w", encoding="utf-8").write(html)
    print("wrote %s (%d ckpts, %d tokens)" % (ns.out, len(models), len(lab)))


if __name__ == "__main__":
    main()
