"""Frozen checkpoint ladder - exp014 (m50m @ 50k vocab, run 20260921_2301).

Question: live val bpb peaked 1.7256 @ step 2700, then slid to 2.0124 @ 5900
until the degrade guard aborted the run. eval_diag (ROADMAP finding #1) says
the live train/val gap is an artifact of live model updates. Was the
post-2700 slide REAL overfit, or artifact?

Method: score every saved checkpoint, frozen (model.eval, no grads), on ONE
fixed set of val + train windows reused verbatim for every rung, using
training's exact recipe: closed-book 99/1 tail split, random crops
(torch.randint over the split), batch 16, ctx 512, bf16 autocast for live
parity (plus an fp32 val pass as a precision cross-check).
bpb = nats * log2(e) / bytes_per_token.

Identical batches for every rung => rung-to-rung deltas are pure model.

Usage: python eval_frozen_ladder.py [--batches 40] [--batch 16]
"""
import argparse, glob, json, math, os, warnings

import numpy as _np
import torch
import torch.nn.functional as F

BIN = "data/openwebtext_combined_bpe_owt50kv2.bin"
PATT = "checkpoints/exp002_pythia_rope_bpe_owt50k_v2_202609221918_*.pt"
VAL_FRAC = 0.01          # train.py --val_frac default (99/1 closed-book)
SEED = 1234              # fixed draws, identical for every checkpoint
DEFAULT_BPT = 4.4130     # exp014 log; per-ckpt cfg["bytes_per_token"] wins

ap = argparse.ArgumentParser()
ap.add_argument("--batches", type=int, default=40)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--out", default="logs/frozen_ladder_exp015.json")
args = ap.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
use_amp = device == "cuda"
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

from model.transformer import Transformer


def draw(source, n_seq, ctx, g):
    """One fixed batch set: idx drawn once, windows materialized to RAM."""
    idx = torch.randint(0, len(source) - ctx - 1, (n_seq,), generator=g)
    xs = torch.stack([source[int(i):int(i) + ctx] for i in idx]).long()
    ys = torch.stack([source[int(i) + 1:int(i) + 1 + ctx] for i in idx]).long()
    return xs, ys


g = torch.Generator().manual_seed(SEED)
n_seq = args.batches * args.batch
vx, vy = draw(val_split, n_seq, 512, g)
tx, ty = draw(train_split, n_seq, 512, g)
print(f"fixed draws materialized: {n_seq} val + {n_seq} train windows "
      f"(every checkpoint sees these exact tensors)", flush=True)
del corpus, train_split, val_split


def score(model, x, y, amp):
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
files = sorted(glob.glob(PATT))
if not files:
    raise SystemExit(f"no checkpoints match {PATT}")
print(f"{len(files)} checkpoints\n", flush=True)

for path in files:
    st = torch.load(path, map_location="cpu", weights_only=True)
    cfg = st.get("cfg", {})
    if "model" in st:
        sd = st["model"]
    else:  # flat state dict
        sd = {k: v for k, v in st.items() if torch.is_tensor(v)}
    bpt = cfg.get("bytes_per_token") or DEFAULT_BPT
    factor = math.log2(math.e) / bpt
    name = os.path.basename(path)
    stepno = st.get("step")
    live_val = st.get("val")
    live_bpb = st.get("val_bpb")

    model = Transformer(
        vocab_size=cfg["vocab_size"],
        context_length=cfg["context_length"],
        embedding_dim=cfg["embedding_dim"],
        num_heads=cfg["num_heads"],
        num_layers=cfg["num_layers"],
        use_rope=bool(cfg.get("use_rope", True)),
        dropout=cfg.get("dropout", 0.0),
    )
    model.load_state_dict(sd, strict=True)   # fail loudly on arch drift
    model = model.to(device)

    v = score(model, vx, vy, use_amp)        # frozen val, bf16 (live parity)
    t = score(model, tx, ty, use_amp)        # frozen random-train
    vf = score(model, vx, vy, False)         # fp32 cross-check, val only

    kind = "best" if name.endswith("best.pt") else "step"
    rows.append({
        "file": name, "kind": kind, "step": stepno,
        "live_val_at_save": live_val, "live_bpb_at_save": live_bpb,
        "val_nats": v, "val_bpb": v * factor,
        "val_nats_fp32": vf, "val_bpb_fp32": vf * factor,
        "train_nats": t, "train_bpb": t * factor,
        "gap_nats": t - v, "bytes_per_token": bpt,
    })
    lv = f"{live_val:.4f}" if live_val is not None else "  n/a"
    print(f"[{kind:>4} step {stepno:>5}] live@save={lv}  "
          f"FROZEN val={v:.4f} ({v * factor:.4f} bpb)  "
          f"train={t:.4f}  gap={t - v:+.4f}  fp32val={vf:.4f}",
          flush=True)
    del model, st, sd
    if use_amp:
        torch.cuda.empty_cache()

rows.sort(key=lambda r: (r["step"] is None, r["step"]))

print("\n=== FROZEN val bpb by step (the ladder) ===")
for r in rows:
    mark = " <- best.pt (guard's champion)" if r["kind"] == "best" else ""
    print(f"  step {r['step']:>5}: {r['val_bpb']:.4f} bpb "
          f"(val {r['val_nats']:.4f} nats){mark}")

with open(args.out, "w") as f:
    json.dump(rows, f, indent=2)
print(f"\nsaved {args.out}")

# The actual verdict input: did frozen val keep improving past best.pt?
non_none = [r for r in rows if r["step"] is not None]
if len(non_none) >= 2:
    first, last = non_none[0], non_none[-1]
    champ = min(non_none, key=lambda r: r["val_bpb"])
    print(f"\nfirst rung  step {first['step']}: {first['val_bpb']:.4f} bpb")
    print(f"last  rung  step {last['step']}: {last['val_bpb']:.4f} bpb")
    print(f"best  rung  step {champ['step']}: {champ['val_bpb']:.4f} bpb")
    delta = last["val_bpb"] - champ["val_bpb"]
    if delta <= 0.005:
        print(f"VERDICT: frozen val did NOT degrade past the champion "
              f"({delta:+.4f} bpb) -> the live slide was ARTIFACT; "
              f"the degrade guard misfired.")
    else:
        print(f"VERDICT: frozen val degraded {delta:+.4f} bpb past the "
              f"champion -> REAL overfit; the guard fired correctly.")
