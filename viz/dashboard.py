#!/usr/bin/env python3
"""viz/dashboard.py v2 -- decision board, not just curves.

Same zero-deps rule: one self-contained HTML, data inlined, file:// works,
--watch rebuilds on a timer. What v2 adds over "watch it learn":

  * x-axis = TOKENS SEEN (steps x eff_batch x ctx). Steps are incomparable
    across batch/vocab changes; the capacity series only makes sense here.
  * three stacked panels on that axis: loss (train avg50 / tval probe / val,
    all bpb), stability (gnorm / lmax / entropy -- the pre-collapse signature),
    loader (served vs random-train recycling tripwire).
  * the arms board: experiments.json rendered with best bpb @ step @ tokens,
    abort marker, lever-vs-control claims, validator problems. The answer to
    "what did this lever do / have we run this before" lives ON the page.
  * a run whose loss.jsonl moved in the last 5 minutes gets a LIVE badge.

Usage:
    python viz/dashboard.py                  # rebuild once -> viz/training.html
    python viz/dashboard.py --watch 10       # rebuild every 10s while a run is on
    python viz/dashboard.py --runs DIR --out FILE
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "viz" / "training.html"
EXP = ROOT / "experiments.json"
LIVE_WINDOW_S = 300
MAX_POINTS = 600          # per run after downsampling (val rows always kept)
LEGACY_MAX_POINTS = 80    # pre-registry-era runs: trend context only, stay lean
CUTOFF = "20260921"       # registry era (matches experiments.json legacy_cutoff)
SERIES_KEYS = ("train", "avg50", "val", "val_bpb", "tval", "served", "served_bpb",
               "random_train", "random_train_bpb", "gnorm", "lmax", "lstd", "ent", "lr")


def load_runs(runs_dir):
    """Harvest run dirs -> compact dicts. Header args give eff_batch/ctx (token axis)."""
    runs = []
    for d in sorted(Path(runs_dir).iterdir()):
        f = d / "loss.jsonl"
        if not d.is_dir() or not f.exists():
            continue
        try:
            lines = f.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        header, rows = {}, []
        for i, line in enumerate(lines):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if i == 0 and "args" in row:
                header = row
                continue
            if "step" in row and "train" in row:
                rows.append(row)
            elif "early_stop" in row:
                rows.append(row)
        if not rows:
            continue
        args = header.get("args", {})
        eff = (args.get("batch") or 0) * (args.get("accum") or 1)
        ctx_len = (header.get("cfg") or {}).get("context_length") or 0
        bpb_factor = None
        for r in rows:
            if r.get("val") and r.get("val_bpb"):
                bpb_factor = r["val_bpb"] / r["val"]
                break
        # downsample: keep stride-lattice + every val-cadence row + endpoints.
        # Pre-registry-era runs are background context only -> much coarser.
        legacy = d.name[:8] < CUTOFF
        cap = 80 if legacy else MAX_POINTS
        keep = {i for i, r in enumerate(rows)
                if (r.get("val") is not None or "early_stop" in r) and not legacy}
        stride = max(1, len(rows) // cap)
        keep.update(range(0, len(rows), stride))
        keep.add(len(rows) - 1)
        dense = []
        for i in sorted(keep):
            r = rows[i]
            e = {"step": r.get("step"), "tokens": (r.get("step") or 0) * eff * ctx_len}
            for k in SERIES_KEYS:
                if r.get(k) is not None:
                    e[k] = r[k]
            if "early_stop" in r:
                e["early_stop"] = True
            dense.append(e)
        samples = []
        sdir = d / "samples"
        if sdir.is_dir():
            def stepkey(p):
                try:
                    return int(p.stem.replace("step", ""))
                except ValueError:
                    return -1
            for p in sorted(sdir.glob("step*.txt"), key=stepkey)[-20:]:
                try:
                    samples.append({"step": stepkey(p),
                                    "text": p.read_text(encoding="utf-8",
                                                        errors="replace")})
                except OSError:
                    continue
        live = (time.time() - f.stat().st_mtime) < LIVE_WINDOW_S
        vals = [r for r in dense if r.get("val_bpb")]
        best = min(vals, key=lambda r: r["val_bpb"]) if vals else None
        stop = next((r for r in dense if r.get("early_stop")), None)
        runs.append({
            "name": d.name, "live": live,
            "params_m": round(header.get("params_m") or 0, 2),
            "preset": args.get("preset"), "lr": args.get("lr"),
            "eff": eff, "ctx": ctx_len,
            "bpb_factor": bpb_factor,
            "rows": dense,
            "best_bpb": best["val_bpb"] if best else None,
            "best_step": best["step"] if best else None,
            "stop_step": stop["step"] if stop else None,
            "last_step": rows[-1].get("step"),
            "samples": samples,
        })
    return runs


def build_arms(meta_arms, runs_by_name):
    """Registry rows + computed stats (from the newest run dir) + validator pass."""
    rows = []
    for a in meta_arms:
        newest = a["run_dirs"][-1] if a["run_dirs"] else None
        r = runs_by_name.get(newest)
        best_step = r["best_step"] if r else None
        rows.append({
            "id": a["id"], "family": a["family"], "verdict": a.get("verdict"),
            "control": a.get("control_id"), "lever": a.get("lever") or {},
            "goal": a.get("goal"), "note": (a.get("note") or "")[:400],
            "confounded_flag": bool(a.get("accepted_confounded")),
            "live": bool(r and r["live"]),
            "best_bpb": r["best_bpb"] if r else None,
            "best_step": best_step,
            "best_tokens": (best_step * r["eff"] * r["ctx"]) if (r and best_step) else None,
            "stop_step": r["stop_step"] if r else None,
            "params_m": r["params_m"] if r else None,
            "n_dirs": len(a["run_dirs"]),
            "match_dirs": a["run_dirs"],
        })
    problems = []
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        import registry as reg
        by_id = {a["id"]: a for a in meta_arms}
        for a in meta_arms:
            if not a["run_dirs"]:
                continue
            args_, _, _ = reg.header_of(a["run_dirs"][-1])
            ctl = by_id.get(a.get("control_id"))
            if args_ is None or not ctl or not ctl["run_dirs"]:
                continue
            cargs, _, _ = reg.header_of(ctl["run_dirs"][-1])
            if cargs is None:
                continue
            for p in reg.validate(a, cargs, args_):
                problems.append(f"{a['id']}: {p}")
    except Exception as e:
        problems.append(f"validator unavailable: {e}")
    return rows, problems


HTML = """<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>llm-lab decision board</title>
<style>
:root{color-scheme:dark}
body{background:#0b0e14;color:#e6edf3;font:14px/1.5 system-ui,sans-serif;margin:0;padding:16px}
h1{font-size:18px;margin:0 0 2px}
h2{font-size:14px;margin:18px 0 6px;color:#8b949e;text-transform:uppercase;letter-spacing:.05em}
.sub{color:#8b949e;font-size:12px;margin-bottom:10px}
.row{display:flex;gap:10px;margin:8px 0;flex-wrap:wrap;align-items:center}
button,select{background:#1c2330;color:#e6edf3;border:1px solid #30363d;border-radius:6px;padding:4px 10px;cursor:pointer;font-size:13px}
canvas{background:#11161f;border:1px solid #30363d;border-radius:8px;width:100%;display:block;margin:6px 0}
#famlist label{display:inline-block;margin:0 10px 6px 0;cursor:pointer;font-size:13px}
.live{color:#a3e635;font-weight:600}
table{border-collapse:collapse;width:100%;font-size:12.5px}
th,td{border:1px solid #21262d;padding:5px 8px;text-align:left;vertical-align:top}
th{background:#161b22;color:#8b949e;position:sticky;top:0}
.v-pass{color:#a3e635}.v-fail{color:#fb7185}.v-running{color:#38bdf8}
.v-closed,.v-inconclusive,.v-superseded,.v-aborted{color:#8b949e}
.badge{border:1px solid #30363d;border-radius:4px;padding:0 5px;font-size:11px;color:#facc15}
#problems{color:#fb7185;white-space:pre-wrap;font-size:12.5px}
#sample{background:#11161f;border:1px solid #30363d;border-radius:8px;padding:12px;white-space:pre-wrap;font:12.5px/1.6 ui-monospace,monospace;max-height:300px;overflow:auto}
.legend{font-size:11.5px;color:#8b949e;margin:2px 0 6px}
</style></head><body>
<h1>llm-lab decision board</h1>
<div class="sub">built __STAMP__ &middot; rebuild: <code>python viz/dashboard.py [--watch N]</code></div>
<div class="row">
  <span id="famlist"></span>
  <button id="xmode">axis: tokens</button>
  <button id="onlysig">show: long arms</button>
</div>
<div class="legend"><b>loss</b>: run-hue = train avg50 (bpb) &middot; dashed white = tval fixed probe &middot; orange = held-out val (dot = best). train&gt;tval&gt;val widening = memorization.
<b>stability</b>: run-hue = grad-norm &middot; yellow = logit-max &middot; teal = entropy. A collapse shows up here first.
<b>loader</b>: run-hue = served &middot; red = random-train. Gap &gt; 0.3 nats = recycling tripwire. x = tokens seen (incomparable across eff-batch on steps).</div>
<canvas id="cvA" width="1240" height="280"></canvas>
<canvas id="cvB" width="1240" height="190"></canvas>
<canvas id="cvC" width="1240" height="150"></canvas>
<h2>Arms (registry)</h2>
<div style="max-height:440px;overflow:auto"><table id="arms"></table></div>
<div id="problems"></div>
<h2>Listen to it learn</h2>
<div class="row">
<label>run: <select id="which"></select></label>
<label>step: <select id="step"></select></label>
</div>
<div id="sample">pick a run to hear it learning.</div>
<script>
const DATA = __DATA__;
const RUNS = DATA.runs, ARMS = DATA.arms;
const PAL = ["#f97316","#38bdf8","#a3e635","#e879f9","#facc15","#fb7185","#2dd4bf","#c084fc","#94a3b8","#fda4af"];
let tokAxis = true, onlySig = true;
let fam = 'ALL';
try { fam = localStorage.getItem('fam') || 'ALL'; } catch(e){}

const fams = [...new Set(ARMS.map(a=>a.family))].sort();
const famBox = document.getElementById('famlist');
['ALL', ...fams].forEach(f=>{
  const lab = document.createElement('label');
  lab.innerHTML = `<input type="radio" name="fam" value="${f}" ${f===fam?'checked':''}> ${f}`;
  famBox.appendChild(lab);
});
famBox.addEventListener('change', e=>{ fam = e.target.value;
  try{ localStorage.setItem('fam', fam); }catch(err){}
  drawArms(); drawAll(); });
document.getElementById('xmode').onclick = e => { tokAxis=!tokAxis;
  e.target.textContent = 'axis: '+(tokAxis?'tokens':'steps'); drawAll(); };
document.getElementById('onlysig').onclick = e => { onlySig=!onlySig;
  e.target.textContent = 'show: '+(onlySig?'long arms':'everything'); drawAll(); };

function runColor(name){ let h=0; for(const c of name) h=(h*31+c.charCodeAt(0))>>>0; return PAL[h%PAL.length]; }

function currentRuns(){
  let dirs = null;
  if (fam !== 'ALL'){ dirs = new Set();
    ARMS.filter(a=>a.family===fam).forEach(a=>(a.match_dirs||[]).forEach(d=>dirs.add(d))); }
  let rs = RUNS.filter(r => dirs ? dirs.has(r.name) : true);
  if (onlySig && fam==='ALL') rs = rs.filter(r => (r.stop_step!=null || r.live) && r.rows.length>200);
  return rs;
}
function pts(r, get){ return r.rows.map(e=>[tokAxis ? e.tokens : e.step, get(e,r)]); }
const bpbOf = k => (e,r)=>{ const f=r.bpb_factor, v=e[k]; return (v==null||f==null)?null:v*f; };

function panel(cv, series, label){
  const ctx = cv.getContext('2d'), W=cv.width, H=cv.height, P=52;
  ctx.clearRect(0,0,W,H);
  let lo=Infinity, hi=-Infinity, xmax=1;
  series.forEach(s=>s.pts.forEach(([x,v])=>{ if(v==null||!isFinite(v))return;
    lo=Math.min(lo,v); hi=Math.max(hi,v); xmax=Math.max(xmax,x); }));
  ctx.strokeStyle='#30363d'; ctx.beginPath(); ctx.moveTo(P,8); ctx.lineTo(P,H-22); ctx.lineTo(W-8,H-22); ctx.stroke();
  ctx.fillStyle='#8b949e'; ctx.font='11px system-ui';
  ctx.fillText(label, P+6, 14);
  if(!isFinite(lo)){ ctx.fillText('no data on this axis', P+12, H/2); return; }
  const span=(hi-lo)||Math.abs(hi)||1; lo-=span*0.06; hi+=span*0.06;
  const X=x=>P+(x/xmax)*(W-P-12), Y=v=>H-22-(v-lo)/(hi-lo)*(H-34);
  ctx.fillText(hi.toFixed(2),4,16); ctx.fillText(lo.toFixed(2),4,H-24);
  ctx.fillText(tokAxis? (xmax/1e6).toFixed(0)+'M tok' : xmax+' steps', W-130, H-8);
  for(let g=1;g<4;g++){ const gx=P+(W-P-12)*g/4;
    ctx.strokeStyle='#161b22'; ctx.beginPath(); ctx.moveTo(gx,8); ctx.lineTo(gx,H-22); ctx.stroke(); }
  series.forEach(s=>{
    ctx.strokeStyle=s.color; ctx.lineWidth=s.w||1.6;
    if(s.dash) ctx.setLineDash(s.dash);
    ctx.beginPath(); let pen=false;
    s.pts.forEach(([x,v])=>{ if(v==null||!isFinite(v))return;
      const px=X(x), py=Y(v); pen?ctx.lineTo(px,py):ctx.moveTo(px,py); pen=true; });
    ctx.stroke(); ctx.setLineDash([]);
    if(s.best!=null){ const b=s.pts.find(p=>p[1]===s.best);
      if(b){ ctx.fillStyle=s.color; ctx.beginPath(); ctx.arc(X(b[0]),Y(b[1]),4,0,7); ctx.fill(); } }
    if(s.abort!=null){ ctx.strokeStyle='#fb7185'; ctx.setLineDash([2,3]);
      ctx.beginPath(); ctx.moveTo(X(s.abort),10); ctx.lineTo(X(s.abort),H-22); ctx.stroke(); ctx.setLineDash([]); }
  });
}
const cvA=document.getElementById('cvA'), cvB=document.getElementById('cvB'), cvC=document.getElementById('cvC');
function drawAll(){
  const rs = currentRuns();
  panel(cvA, rs.flatMap(r=>[
    {color:runColor(r.name), pts:pts(r,bpbOf('avg50')), w:1.2},
    {color:'#e6edf3', pts:pts(r,bpbOf('tval')), w:1.3, dash:[5,3]},
    {color:'#f97316', pts:pts(r,e=>e.val_bpb), w:2, best:r.best_bpb, abort:r.stop_step},
  ]), 'loss bpb — train / tval / val (orange dot=best, red dash=guard abort)');
  panel(cvB, rs.flatMap(r=>[
    {color:runColor(r.name), pts:pts(r,e=>e.gnorm), w:1.2},
    {color:'#facc15', pts:pts(r,e=>e.lmax), w:1.2},
    {color:'#2dd4bf', pts:pts(r,e=>e.ent), w:1.2},
  ]), 'stability — gnorm (run hue) / lmax (yellow) / entropy (teal)  [post-2026-09-23 runs only]');
  panel(cvC, rs.flatMap(r=>[
    {color:runColor(r.name), pts:pts(r,e=>e.served), w:1.2},
    {color:'#fb7185', pts:pts(r,e=>e.random_train), w:1.2},
  ]), 'loader — served (run hue) vs random-train (red), nats');
}

function drawArms(){
  const t=document.getElementById('arms');
  const rows = ARMS.filter(a=>fam==='ALL'||a.family===fam)
                   .sort((x,y)=>String(x.family).localeCompare(y.family)||String(x.id).localeCompare(y.id));
  t.innerHTML = '<tr><th>arm</th><th>verdict</th><th>M</th><th>best bpb</th><th>@step</th><th>@tok</th><th>abort</th><th>ctl</th><th>lever</th><th>goal</th></tr>' +
    rows.map(a=>`<tr>
      <td><b>${a.id}</b>${a.live?' <span class="live">LIVE</span>':''}${a.n_dirs>1?' <span class="badge">x'+a.n_dirs+'</span>':''}${a.confounded_flag?' <span class="badge">confounded*</span>':''}</td>
      <td class="v-${a.verdict}">${a.verdict||''}</td>
      <td>${a.params_m?a.params_m.toFixed? a.params_m.toFixed(1):a.params_m:''}</td>
      <td>${a.best_bpb?a.best_bpb.toFixed(4):'—'}</td>
      <td>${a.best_step||'—'}</td>
      <td>${a.best_tokens?(a.best_tokens/1e6).toFixed(0)+'M':'—'}</td>
      <td>${a.stop_step?(''+a.stop_step):''}</td>
      <td>${a.control||''}</td>
      <td style="max-width:280px">${Object.entries(a.lever).filter(([k])=>k!=='note').map(([k,v])=>`<code>${k}</code>: ${v}`).join('<br>')}</td>
      <td style="max-width:340px">${a.goal||''}</td></tr>`).join('');
  document.getElementById('problems').textContent =
    DATA.problems.length ? ('REGISTRY PROBLEMS:\\n' + DATA.problems.join('\\n')) : '';
}

const which=document.getElementById('which'), stepSel=document.getElementById('step'), sample=document.getElementById('sample');
RUNS.forEach((r,i)=>{ if(!r.samples.length)return;
  const o=document.createElement('option'); o.value=i; o.textContent=r.name; which.appendChild(o); });
function fillSteps(){ stepSel.innerHTML=''; const r=RUNS[+which.value]; if(!r)return;
  r.samples.forEach(s=>{const o=document.createElement('option');o.value=s.step;o.textContent='step '+s.step;stepSel.appendChild(o);});
  if(r.samples.length){stepSel.value=r.samples[r.samples.length-1].step;show();} else sample.textContent='no samples'; }
function show(){ const r=RUNS[+which.value]; if(!r)return;
  const s=r.samples.find(s=>String(s.step)===stepSel.value); sample.textContent=s?s.text:'(missing)'; }
which.onchange=fillSteps; stepSel.onchange=show;
if([...which.options].length) fillSteps();
drawArms(); drawAll();
</script></body></html>
"""


def build_once(runs_dir, out):
    runs = load_runs(runs_dir)
    by_name = {r["name"]: r for r in runs}
    meta_arms = json.load(open(EXP, encoding="utf-8"))["arms"] if EXP.exists() else []
    arms, problems = build_arms(meta_arms, by_name)
    payload = json.dumps({"runs": runs, "arms": arms, "problems": problems}
                         ).replace("</", "<\\/")
    page = (HTML.replace("__DATA__", payload)
                .replace("__STAMP__", datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    print(f"dashboard: {len(runs)} runs, {len(arms)} arms, {len(problems)} registry "
          f"problems -> {out} ({out.stat().st_size//1024}KB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(ROOT / "runs"))
    ap.add_argument("--watch", type=int, default=0,
                    help="rebuild every N seconds (0 = once)")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    if a.watch > 0:
        print(f"watching {a.runs} every {a.watch}s -> {a.out} (Ctrl-C stops)")
        while True:
            build_once(a.runs, a.out)
            time.sleep(a.watch)
    else:
        build_once(a.runs, a.out)


if __name__ == "__main__":
    sys.exit(main())
