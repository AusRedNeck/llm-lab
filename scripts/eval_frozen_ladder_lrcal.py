"""Frozen checkpoint ladder for the p160-2k LR-calibration family (arms A/B).

Why this exists: the in-loop val rows are NOT comparable across the two LR
arms (arm A resumed from the step-500 checkpoint, i.e. a re-walked seed), so
the bracket verdict needs ONE external harness that scores every checkpoint of
both arms on byte-identical windows. Method is verbatim from
scripts/eval_frozen_ladder.py, re-parameterized for this family:

  * closed-book 99/1 tail split of the run's own --tok_cache bin (train.py's
    exact recipe, --val_frac 0.01)
  * ONE fixed draw set, seeded, materialized once and reused for every rung
    => rung-to-rung and arm-to-arm deltas are pure model
  * frozen model.eval, no grads
  * bpb = nats * log2(e) / bytes_per_token (per-ckpt cfg wins)

Device defaults to CPU so it can run unattended without touching the GPU that
training owns; pass --device cuda when the GPU is already in use for a run.

Usage:
  python scripts/eval_frozen_ladder_lrcal.py \
      --ckpts checkpoints/<armA>_step6000.pt checkpoints/<armB>_step6000.pt \
      [--batches 40] [--batch 16] [--device cpu] [--train] [--fp32] \
      [--out logs/frozen_ladder_lr_cal.json]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import warnings

import numpy as _np
import torch
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)

BIN = "data/pile_train_full_bpe_pythia_hf.bin"   # --tok_cache of both arms
TXT = "data/pile_train_full.txt"                 # --corpus (for bpt fallback)
META = BIN + ".meta.json"
VAL_FRAC = 0.01                                  # train.py --val_frac (99/1)
SEED = 1234                                      # fixed draws, both arms

ap = argparse.ArgumentParser()
ap.add_argument("--ckpts", nargs="*", default=None,
                help="Explicit checkpoint paths (preferred: arm A + arm B).")
ap.add_argument("--glob", default=None,
                help="Glob alternative, e.g. 'checkpoints/exp002_*_step*.pt'")
ap.add_argument("--batches", type=int, default=40)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
ap.add_argument("--train", action="store_true",
                help="Also score a fixed random-train draw set (gap metric).")
ap.add_argument("--fp32", action="store_true",
                help="Extra fp32 val pass as a precision cross-check.")
ap.add_argument("--out", default="logs/frozen_ladder_lr_cal.json")
args = ap.parse_args()

device = torch.device(args.device)
use_amp = args.device == "cuda"
print(f"device={device}  draws={args.batches} x {args.batch}  seed={SEED}",
      flush=True)

# --- corpus: same memmap loader as train.py ---
n = os.path.getsize(BIN) // 4
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    corpus = torch.from_numpy(_np.memmap(BIN, dtype="<i4", mode="r",
                                          shape=(n,)))
cut = int(n * (1.0 - VAL_FRAC))
train_split, val_split = corpus[:cut], corpus[cut:]
print(f"corpus {n:,} -> train {len(train_split):,} / val {len(val_split):,}",
      flush=True)


def draw(source, n_seq, ctx, g):
    """One fixed batch set: idx drawn once, windows materialized to RAM."""
    idx = torch.randint(0, len(source) - ctx - 1, (n_seq,), generator=g)
    xs = torch.stack([source[int(i):int(i) + ctx] for i in idx]).long()
    ys = torch.stack([source[int(i) + 1:int(i) + 1 + ctx] for i in idx]).long()
    return xs, ys


def bytes_per_token_fallback(cfg_bpt):
    """cfg wins, then the encode manifest, then corpus bytes / token count."""
    if cfg_bpt:
        return float(cfg_bpt), "ckpt cfg"
    if os.path.exists(META):
        try:
            with open(META) as f:
                m = json.load(f)
            for k in ("bytes_per_token", "bpt"):
                if m.get(k):
                    return float(m[k]), f"meta:{k}"
        except Exception:
            pass
    if os.path.exists(TXT):
        return os.path.getsize(TXT) / float(n), "corpus bytes/tokens"
    return 1.0, "assumed 1.0"


files = args.ckpts or (glob.glob(args.glob) if args.glob else [])
if not files:
    raise SystemExit("no checkpoints: pass --ckpts or --glob")
missing = [p for p in files if not os.path.exists(p)]
if missing:
    raise SystemExit(f"missing checkpoint(s): {missing}")

# ctx comes from the run preset (both arms share it) - read it once, up front.
probe = torch.load(files[0], map_location="cpu", weights_only=True)
cfg0 = probe.get("cfg", {})
ctx = cfg0["context_length"]
bpt, bpt_src = bytes_per_token_fallback(cfg0.get("bytes_per_token"))
factor = math.log2(math.e) / bpt
print(f"ctx={ctx}  vocab={cfg0.get('vocab_size')}  bpt={bpt} ({bpt_src})",
      flush=True)
del probe

g = torch.Generator().manual_seed(SEED)
n_seq = args.batches * args.batch
vx, vy = draw(val_split, n_seq, ctx, g)
print(f"fixed val draws materialized: {n_seq} windows "
      f"(every checkpoint sees these exact tensors)", flush=True)
tx = ty = None
if args.train:
    g2 = torch.Generator().manual_seed(SEED + 1)
    tx, ty = draw(train_split, n_seq, ctx, g2)
del corpus, train_split, val_split

from model.transformer import Transformer


def score(model, x, y, amp=False):
    """Mean CE over the fixed windows - mirrors eval_three_way's val arm."""
    model.eval()
    tot, bs = 0.0, args.batch
    nb = math.ceil(x.shape[0] / bs)
    with torch.no_grad():
        for i in range(0, x.shape[0], bs):
            xb, yb = x[i:i + bs].to(device), y[i:i + bs].to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=amp):
                logits = model(xb)
                tot += F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    yb.reshape(-1)).item()
    return tot / nb


rows = []
for path in files:
    st = torch.load(path, map_location="cpu", weights_only=True)
    cfg = st.get("cfg", {})
    if cfg.get("context_length") != ctx:
        raise SystemExit(f"{path}: context_length {cfg.get('context_length')} "
                         f"!= probe {ctx} - arms must share the preset")
    sd = st.get("model") or {k: v for k, v in st.items() if torch.is_tensor(v)}
    ck_bpt = cfg.get("bytes_per_token")
    use_bpt = float(ck_bpt) if ck_bpt else bpt
    fac = math.log2(math.e) / use_bpt
    name = os.path.basename(path)
    stepno = st.get("step")
    live_val = st.get("val")
    live_bpb = st.get("val_bpb")

    model = Transformer.from_config(cfg)
    model.load_state_dict(sd, strict=True)   # fail loudly on arch drift
    model = model.to(device)

    v = score(model, vx, vy, use_amp and args.device == "cuda")
    row = {
        "file": name, "step": stepno, "path": path,
        "live_val_at_save": live_val, "live_bpb_at_save": live_bpb,
        "val_nats": v, "val_bpb": v * fac, "bytes_per_token": use_bpt,
        "draws": n_seq, "batch": args.batch, "seed": SEED,
        "device": args.device, "ctx": ctx,
    }
    if args.train and tx is not None:
        t = score(model, tx, ty, use_amp and args.device == "cuda")
        row.update(train_nats=t, train_bpb=t * fac, gap_nats=t - v)
    if args.fp32:
        vf = score(model, vx, vy, False)
        row.update(val_nats_fp32=vf, val_bpb_fp32=vf * fac)

    lv = f"{live_val:.4f}" if live_val is not None else "n/a"
    extra = f"  train={row['train_nats']:.4f}" if "train_nats" in row else ""
    extra += f"  fp32val={row['val_nats_fp32']:.4f}" if "val_nats_fp32" in row else ""
    print(f"[step {stepno:>6}] live@save={lv}  "
          f"FROZEN val={v:.4f} ({v * fac:.4f} bpb){extra}", flush=True)
    rows.append(row)
    del model, st, sd
    if args.device == "cuda":
        torch.cuda.empty_cache()

rows.sort(key=lambda r: (r["step"] is None, r["step"]))

print("\n=== FROZEN val bpb by step (identical windows for every rung) ===")
for r in rows:
    print(f"  step {r['step']:>6}: {r['val_bpb']:.4f} bpb "
          f"(val {r['val_nats']:.4f} nats)")

os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
with open(args.out, "w") as f:
    json.dump(rows, f, indent=2)
print(f"\nsaved {args.out} ({os.path.getsize(args.out)} B)")

if len(rows) >= 2:
    lo = min(rows, key=lambda r: r["val_bpb"])
    hi = max(rows, key=lambda r: r["val_bpb"])
    print(f"best rung: step {lo['step']} @ {lo['val_bpb']:.4f} bpb | "
          f"worst: step {hi['step']} @ {hi['val_bpb']:.4f} bpb | "
          f"delta {hi['val_bpb'] - lo['val_bpb']:+.4f}")
    print("VERDICT INPUT: same harness + same windows for every arm - add "
          "arm B's step6000 with the identical argv.")
