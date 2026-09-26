#!/usr/bin/env python3
"""viz/probe_plot.py -- attention-over-training curves from probe JSONs.

Reads probes/<ckpt>.json rows (viz/probe.py output) and renders one
standalone HTML: mid-layer attention entropy, locality, end-layer
effective rank, top1 confidence + bpb, all vs training step.
Collapse/overfit shows here first: entropy crash, rank fall, locality
spike while bpb stalls.

Usage:
    python viz/probe.py checkpoints/<run>_step*.pt --out probes/
    python -m viz.probe_plot --probes "probes/<run>*.json" --out viz/probe_curve.html
"""
from __future__ import annotations

import argparse
import glob
import html as htmlmod
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BPB_BYTES = 4.413  # OWT-50k stream bytes/token; bpb = ce * 1.4427 / bytes


def mid(v):
    """Middle-layer value (representative depth)."""
    if not v:
        return None
    return v[len(v) // 2]


def render_probe_plot(rows) -> str:
    """rows: probe JSON dicts with step/ckpt/attn_entropy/locality/eff_rank/top1_p/ce_nats."""
    rows = sorted([r for r in rows if r.get("step") is not None],
                  key=lambda r: r["step"])
    pts = []
    for r in rows:
        ce = r.get("ce_nats")
        pts.append({
            "step": r["step"], "ckpt": r.get("ckpt", ""),
            "ent": mid(r.get("attn_entropy") or []),
            "loc": mid(r.get("locality") or []),
            "rank": (r.get("eff_rank") or [None])[-1],
            "top1": r.get("top1_p"),
            "bpb": ce * 1.4427 / BPB_BYTES if ce is not None else None,
        })
    names = " ".join(htmlmod.escape(p["ckpt"]) for p in pts)
    return """<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>probe curves over training</title>
<style>
:root{color-scheme:dark}
body{background:#0b0e14;color:#e6edf3;font:14px/1.5 system-ui,sans-serif;margin:0;padding:16px}
h1{font-size:17px;margin:0 0 2px}.sub{color:#8b949e;font-size:12px;margin-bottom:10px}
canvas{background:#11161f;border:1px solid #30363d;border-radius:8px;width:100%%;display:block;margin:6px 0}
.legend{font-size:11.5px;color:#8b949e;margin:2px 0 6px}
</style></head><body>
<h1>attention over training &mdash; probe curves</h1>
<div class="sub">%(n)d probes &middot; %(names)s</div>
<div class="legend"><b>entropy</b>: mid-layer attention entropy (fall = focus, crash = collapse) &middot;
<b>locality</b>: mass within distance 1 (spike = copy/induction regime) &middot;
<b>rank</b>: end-layer effective rank (fall = representational collapse) &middot;
<b>top1/bpb</b>: confidence vs probe loss.</div>
<canvas id="c1" width="1100" height="200"></canvas>
<canvas id="c2" width="1100" height="200"></canvas>
<canvas id="c3" width="1100" height="200"></canvas>
<canvas id="c4" width="1100" height="200"></canvas>
<script>
const P=%(data)s;
function panel(id,series,label){const cv=document.getElementById(id),
ctx=cv.getContext('2d'),W=cv.width,H=cv.height,Pd=52;
ctx.clearRect(0,0,W,H);let lo=Infinity,hi=-Infinity,xmax=1;
series.forEach(s=>s.pts.forEach(([x,v])=>{if(v==null||!isFinite(v))return;
lo=Math.min(lo,v);hi=Math.max(hi,v);xmax=Math.max(xmax,x);}));
ctx.strokeStyle='#30363d';ctx.beginPath();ctx.moveTo(Pd,8);
ctx.lineTo(Pd,H-22);ctx.lineTo(W-8,H-22);ctx.stroke();
ctx.fillStyle='#8b949e';ctx.font='11px system-ui';ctx.fillText(label,Pd+6,14);
if(!isFinite(lo)){ctx.fillText('need 2+ probes, have '+P.length,Pd+12,H/2);return;}
const sp=(hi-lo)||Math.abs(hi)||1;lo-=sp*0.06;hi+=sp*0.06;
const X=x=>Pd+(x/xmax)*(W-Pd-12),Y=v=>H-22-(v-lo)/(hi-lo)*(H-34);
ctx.fillText(hi.toFixed(2),4,16);ctx.fillText(lo.toFixed(2),4,H-24);
ctx.fillText('step '+xmax,W-110,H-8);
series.forEach(s=>{ctx.strokeStyle=s.color;ctx.lineWidth=2;ctx.beginPath();
let st=false;s.pts.forEach(([x,v])=>{if(v==null||!isFinite(v))return;
st?ctx.lineTo(X(x),Y(v)):ctx.moveTo(X(x),Y(v));st=true;});ctx.stroke();
ctx.fillStyle=s.color;s.pts.forEach(([x,v])=>{if(v==null||!isFinite(v))return;
ctx.beginPath();ctx.arc(X(x),Y(v),3,0,7);ctx.fill();});});}
const S=k=>P.map(p=>[p.step,p[k]]);
panel('c1',[{color:'#38bdf8',pts:S('ent')}],'mid-layer attention entropy');
panel('c2',[{color:'#a3e635',pts:S('loc')}],'mid-layer locality (copy signal)');
panel('c3',[{color:'#e879f9',pts:S('rank')}],'end-layer eff_rank');
panel('c4',[{color:'#f97316',pts:S('bpb')},{color:'#e6edf3',pts:S('top1')}],'probe bpb (orange) / top1 (white)');
</script></body></html>""" % {
        "n": len(pts), "names": names, "data": json.dumps(pts),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes", nargs="+", required=True)
    ap.add_argument("--out", default="viz/probe_curve.html")
    ns = ap.parse_args()
    files = []
    for pat in ns.probes:
        files.extend(sorted(glob.glob(pat)) or [pat])
    rows = [json.load(open(f, encoding="utf-8")) for f in files]
    html = render_probe_plot(rows)
    open(ns.out, "w", encoding="utf-8").write(html)
    print("wrote %s (%d probes)" % (ns.out, len(rows)))


if __name__ == "__main__":
    main()
