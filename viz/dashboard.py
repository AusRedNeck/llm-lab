#!/usr/bin/env python3
"""viz/dashboard.py -- One-page training dashboard. Watch it learn.

Reads runs/*/loss.jsonl + samples/step*.txt, inlines everything into
viz/training.html (zero deps, no CDN, works offline on Ronin).

Usage:
    python viz/dashboard.py                  # rebuild once, open viz/training.html
    python viz/dashboard.py --watch 10       # rebuild every 10s while Ronin crunches
    python viz/dashboard.py --runs runs      # point at another runs dir
"""
import argparse
import html
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "viz" / "training.html"

# Distinct hues so overlaid runs stay readable.
PALETTE = ["#f97316", "#38bdf8", "#a3e635", "#e879f9",
           "#facc15", "#fb7185", "#2dd4bf", "#c084fc"]


def load_runs(runs_dir):
    """Harvest every run dir into {name, params_m, steps[], train[], avg50[], val[], samples[]}.

    Skips dirs without loss.jsonl -- partial/corrupt rows are dropped, not fatal.
    """
    runs = []
    for d in sorted(Path(runs_dir).iterdir()):
        f = d / "loss.jsonl"
        if not d.is_dir() or not f.exists():
            continue
        header, steps, train, avg, val = {}, [], [], [], []
        val_bpb, lrs = [], []
        served_bpb, random_train_bpb = [], []
        early_stop = None
        try:
            lines = f.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for i, line in enumerate(lines):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if i == 0 and "args" in row:  # first line is the run header
                header = row
                continue
            if row.get("early_stop"):
                # Deliberate end marker, not a data row. Keep separate
                # so the audit and the UI read it, not the curve.
                early_stop = {"step": row.get("step"),
                              "best_bpb": row.get("best_bpb"),
                              "best_val": row.get("best_val")}
                continue
            if "step" not in row or "train" not in row:
                continue  # junk -- skip
            steps.append(row["step"])
            train.append(row["train"])
            avg.append(row.get("avg50"))
            val.append(row.get("val"))
            val_bpb.append(row.get("val_bpb"))
            lrs.append(row.get("lr"))
            served_bpb.append(row.get("served_bpb"))
            random_train_bpb.append(row.get("random_train_bpb"))
        # Samples: stepN.txt files, oldest first. Text inlined, escaped later.
        samples = []
        sdir = d / "samples"
        if sdir.is_dir():
            def stepkey(p):
                try:
                    return int(p.stem.replace("step", ""))
                except ValueError:
                    return -1
            for p in sorted(sdir.glob("step*.txt"), key=stepkey):
                try:
                    samples.append({"step": stepkey(p),
                                    "text": p.read_text(encoding="utf-8",
                                                        errors="replace")})
                except OSError:
                    continue
        if steps:
            runs.append({"name": d.name, "params_m": header.get("params_m"),
                         "args": header.get("args", {}),
                         "steps": steps, "train": train, "avg50": avg,
                         "val": val, "val_bpb": val_bpb, "lr": lrs,
                         "served_bpb": served_bpb,
                         "random_train_bpb": random_train_bpb,
                         "early_stop": early_stop, "samples": samples})
    return runs


def build(runs_dir, out=OUT):
    """Rebuild the HTML. Data inlined as JSON so file:// just works."""
    runs = load_runs(runs_dir)
    payload = json.dumps(runs).replace("</", "<\\/")
    page = HTML.replace("__DATA__", payload)
    page = page.replace("__STAMP__", time.strftime("%Y-%m-%d %H:%M:%S"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    total_steps = sum(len(r["steps"]) for r in runs)
    print(f"dashboard: {len(runs)} runs, {total_steps} steps -> {out}")
    return runs


HTML = """<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>llm-lab training</title>
<style>
:root{color-scheme:dark}
body{background:#0b0e14;color:#e6edf3;font:14px/1.5 system-ui,sans-serif;margin:0;padding:16px}
h1{font-size:18px;margin:0 0 4px}
.sub{color:#8b949e;font-size:12px;margin-bottom:12px}
#runs label{display:inline-block;margin:0 12px 6px 0;cursor:pointer;font-size:13px}
canvas{background:#11161f;border:1px solid #30363d;border-radius:8px;width:100%;height:340px}
.row{display:flex;gap:12px;margin:10px 0;flex-wrap:wrap;align-items:center}
button,select{background:#1c2330;color:#e6edf3;border:1px solid #30363d;border-radius:6px;padding:4px 10px;cursor:pointer}
#sample{background:#11161f;border:1px solid #30363d;border-radius:8px;padding:12px;white-space:pre-wrap;font:12.5px/1.6 ui-monospace,monospace;max-height:320px;overflow:auto}
.sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px}
</style></head><body>
<h1>llm-lab training dashboard</h1>
<div class="sub">built __STAMP__ &middot; rebuild with <code>python viz/dashboard.py [--watch N]</code></div>
<div id="runs"></div>
<canvas id="cv" width="1200" height="340"></canvas>
<div class="row">
<button id="mode">series: val_bpb</button>
<button id="scale">scale: linear</button>
<button id="lr">lr: on</button>
<button id="threeway">3-way: off</button>
<label>samples: <select id="which"></select></label>
<label>step: <select id="step"></select></label>
</div>
<div id="legend" style="font-size:12px;color:#8b949e;margin-bottom:8px"></div>
<div id="sample">pick a run to hear it learning.</div>
<script>
const RUNS = __DATA__;
const cv = document.getElementById('cv'), ctx = cv.getContext('2d');
const PAL = ["#f97316","#38bdf8","#a3e635","#e879f9","#facc15","#fb7185","#2dd4bf","#c084fc"];
let mode = 'val_bpb', log = false;  // bpb first: fair metric across vocabs
let showLr = true, showThree = false;
// One checkbox per run; all on by default.
const box = document.getElementById('runs');
RUNS.forEach((r, i) => {
  const lab = document.createElement('label');
  lab.innerHTML = `<input type="checkbox" checked data-i="${i}"><span class="sw" style="background:${PAL[i % PAL.length]}"></span>${r.name}${r.params_m ? ` (${r.params_m.toFixed(1)}M)` : ''}`;
  box.appendChild(lab);
});
box.addEventListener('change', draw);
document.getElementById('mode').onclick = e => {
  mode = mode === 'val_bpb' ? 'val' : mode === 'val' ? 'avg50' : mode === 'avg50' ? 'train' : 'val_bpb';
  e.target.textContent = 'series: ' + mode; draw();
};
document.getElementById('scale').onclick = e => {
  log = !log; e.target.textContent = 'scale: ' + (log ? 'log' : 'linear'); draw();
};
document.getElementById('lr').onclick = e => {
  showLr = !showLr; e.target.textContent = 'lr: ' + (showLr ? 'on' : 'off'); draw();
};
document.getElementById('threeway').onclick = e => {
  showThree = !showThree; e.target.textContent = '3-way: ' + (showThree ? 'on' : 'off'); draw();
};
function active() {
  return [...box.querySelectorAll('input:checked')].map(c => RUNS[+c.dataset.i]);
}
function draw() {
  // Min/max over visible series, then plot. Nulls break the pen so
  // eval-death gaps (160M @3901) render as red dashes, not silence.
  const rs = active(), W = cv.width, H = cv.height, P = 44;
  ctx.clearRect(0, 0, W, H);
  let lo = Infinity, hi = -Infinity, xmax = 0;
  rs.forEach(r => r.steps.forEach((s, i) => {
    const v = r[mode][i]; if (v == null) return;
    lo = Math.min(lo, v); hi = Math.max(hi, v); xmax = Math.max(xmax, s);
  }));
  if (!isFinite(lo)) { ctx.fillStyle = '#8b949e'; ctx.fillText('no data for ' + mode, P + 10, 30); return; }
  if (log) { lo = Math.log(Math.max(lo, 1e-6)); hi = Math.log(Math.max(hi, 1e-6)); }
  const X = s => P + (s / Math.max(xmax, 1)) * (W - P - 12);
  const Y = v => { v = log ? Math.log(Math.max(v, 1e-6)) : v;
    return H - 26 - (v - lo) / Math.max(hi - lo, 1e-9) * (H - 26 - 14); };
  // Axes: bare min/max labels, nothing fancy.
  ctx.strokeStyle = '#30363d'; ctx.beginPath();
  ctx.moveTo(P, 8); ctx.lineTo(P, H - 26); ctx.lineTo(W - 8, H - 26); ctx.stroke();
  ctx.fillStyle = '#8b949e'; ctx.font = '11px system-ui';
  ctx.fillText(hi.toFixed(2), 4, 18); ctx.fillText(lo.toFixed(2), 4, H - 28);
  ctx.fillText('step ' + xmax, W - 90, H - 10);
  // LR strip: peak-normalised fill under the top edge. Flat val at high
  // LR reads "schedule, keep going". Flat val at decayed LR reads "knee".
  if (showLr) {
    rs.forEach(r => {
      if (!r.lr || !r.lr.length) return;
      let peak = 0;
      r.lr.forEach(v => { if (v != null && v > peak) peak = v; });
      if (!peak) return;
      ctx.fillStyle = PAL[RUNS.indexOf(r) % PAL.length] + '22';
      ctx.beginPath();
      r.steps.forEach((s, i) => {
        const v = r.lr[i]; if (v == null) return;
        const x = X(s), y = 8 + (1 - v / peak) * 28;
        i === 0 ? ctx.moveTo(x, 8) : ctx.lineTo(x, y);
      });
      const lastS = r.steps[r.steps.length - 1];
      ctx.lineTo(X(lastS), 8); ctx.closePath(); ctx.fill();
    });
  }
  // Guard line: best bpb * (1 + degrade_frac) from the run header.
  // Val crossing it mid-schedule fires the abort. Noise band check by eye.
  rs.forEach(r => {
    const df = r.args && r.args.degrade_frac;
    if (df == null || mode !== 'val_bpb') return;
    let best = Infinity;
    r.val_bpb.forEach(v => { if (v != null && v < best) best = v; });
    if (!isFinite(best)) return;
    const g = best * (1 + df), gy = Y(g);
    if (gy < 8 || gy > H - 26) return;
    ctx.strokeStyle = '#e74c3c'; ctx.lineWidth = 1; ctx.setLineDash([6, 4]);
    ctx.beginPath(); ctx.moveTo(P, gy); ctx.lineTo(W - 8, gy); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#e74c3c'; ctx.font = '10px system-ui';
    ctx.fillText('guard +' + (df * 100).toFixed(0) + '% (' + g.toFixed(3) + ')', P + 4, gy - 3);
  });
  rs.forEach(r => {
    const col = PAL[RUNS.indexOf(r) % PAL.length];
    ctx.strokeStyle = col; ctx.lineWidth = 1.6;
    ctx.beginPath(); let pen = false;
    r.steps.forEach((s, i) => {
      const v = r[mode][i];
      if (v == null) { pen = false; return; }
      const x = X(s), y = Y(v);
      pen ? ctx.lineTo(x, y) : ctx.moveTo(x, y); pen = true;
    });
    ctx.stroke();
    // Red dashed bridge across each null span: eval died here.
    ctx.strokeStyle = '#e74c3c'; ctx.lineWidth = 1.2; ctx.setLineDash([4, 3]);
    let gs = null;
    r.steps.forEach((s, i) => {
      if (r[mode][i] == null && gs == null) gs = s;
      if (r[mode][i] != null && gs != null) {
        ctx.beginPath(); ctx.moveTo(X(gs), H - 26); ctx.lineTo(X(s), H - 26); ctx.stroke();
        ctx.fillStyle = '#e74c3c'; ctx.font = '10px system-ui';
        ctx.fillText('eval gap ' + gs + '-' + s, X(gs) + 2, H - 32);
        gs = null;
      }
    });
    // Trailing gap: line died and never came back (160M signature).
    if (gs != null) {
      ctx.beginPath(); ctx.moveTo(X(gs), H - 26); ctx.lineTo(X(xmax), H - 26); ctx.stroke();
      ctx.fillStyle = '#e74c3c'; ctx.font = '10px system-ui';
      ctx.fillText('eval dead from ' + gs, X(gs) + 2, H - 32);
    }
    ctx.setLineDash([]);
    // Early-stop flag: deliberate end, green tick, not a gap.
    if (r.early_stop && r.early_stop.step != null) {
      const ex = X(r.early_stop.step);
      ctx.strokeStyle = '#2ecc71'; ctx.lineWidth = 1.2;
      ctx.beginPath(); ctx.moveTo(ex, 8); ctx.lineTo(ex, H - 26); ctx.stroke();
      ctx.fillStyle = '#2ecc71'; ctx.font = '10px system-ui';
      const bl = r.early_stop.best_bpb != null ? ' best bpb ' + r.early_stop.best_bpb : '';
      ctx.fillText('stop' + bl, ex + 3, 18);
    }
    // Three-way: served vs random-train bpb. Tight means loader healthy.
    // Served diving below random_train means recycling: kill the run.
    if (showThree && (mode === 'val_bpb' || mode === 'val')) {
      const sKey = mode === 'val_bpb' ? 'served_bpb' : 'served';
      const tKey = mode === 'val_bpb' ? 'random_train_bpb' : 'random_train';
      [[sKey, '#facc15'], [tKey, '#2dd4bf']].forEach(([k, c]) => {
        if (!r[k]) return;
        ctx.strokeStyle = c; ctx.lineWidth = 1.1; ctx.setLineDash([2, 2]);
        ctx.beginPath(); let pen = false;
        r.steps.forEach((s, i) => {
          const v = r[k][i]; if (v == null) { pen = false; return; }
          const x = X(s), y = Y(v);
          pen ? ctx.lineTo(x, y) : ctx.moveTo(x, y); pen = true;
        });
        ctx.stroke(); ctx.setLineDash([]);
      });
    }
  });
  // Legend line under the canvas.
  const lg = document.getElementById('legend');
  lg.textContent = showThree
    ? 'yellow dashed = served bpb (loader replay) / teal dashed = random-train bpb / tight = healthy, served diving = recycling'
    : '';
}
// Sample viewer: run dropdown -> step dropdown -> raw text.
const which = document.getElementById('which'), step = document.getElementById('step'),
      sample = document.getElementById('sample');
RUNS.forEach((r, i) => {
  const o = document.createElement('option'); o.value = i; o.textContent = r.name; which.appendChild(o);
});
function fillSteps() {
  step.innerHTML = '';
  const r = RUNS[+which.value]; if (!r) return;
  r.samples.forEach(s => {
    const o = document.createElement('option'); o.value = s.step; o.textContent = 'step ' + s.step; step.appendChild(o);
  });
  if (r.samples.length) { step.value = r.samples[r.samples.length - 1].step; show(); }
  else sample.textContent = 'no samples for this run yet.';
}
function show() {
  const r = RUNS[+which.value]; if (!r) return;
  const s = r.samples.find(s => String(s.step) === step.value);
  sample.textContent = s ? s.text : '(missing)';
}
which.onchange = fillSteps; step.onchange = show;
if (RUNS.length) fillSteps();
draw();
</script></body></html>
"""


def main():
    """Parse args, build once or keep rebuilding on a timer."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(ROOT / "runs"))
    ap.add_argument("--watch", type=int, default=0,
                    help="rebuild every N seconds (0 = once)")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    if a.watch > 0:
        print(f"watching {a.runs} every {a.watch}s -> {a.out} (Ctrl-C stops)")
        while True:
            build(a.runs, Path(a.out))
            time.sleep(a.watch)
    else:
        build(a.runs, Path(a.out))


if __name__ == "__main__":
    sys.exit(main())
