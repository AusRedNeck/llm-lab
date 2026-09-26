#!/usr/bin/env python3
"""viz/probe.py -- offline X-ray of a checkpoint via the glass-box capture path.

Never runs in the training loop. One 256-token forward per probe window through
model.forward_with_capture() (the dormant viz/hooks.py connector), then:

  layer_norms      per-layer hidden-state norm (collapsing = dead/crushed layer)
  eff_rank         participation ratio of centered hidden-state singular values
                   per layer (representational collapse shows up here first)
  attn_entropy     per-layer mean attention entropy (heads x windows)
  locality         mean attention mass within distance 1 (copy/induction signal)
  topk_gap         top1-prob minus top10-cumulative on final logits (confidence)
  bpb              CE on the probe windows (fixed seed: comparable across ckpts)

Fixed probe windows: seed 1234, val split, same convention as eval_frozen_ladder.
Results -> probes/<ckpt-stem>.json

Usage:
  python viz/probe.py checkpoints/exp002_..._step2000.pt [--rotary-pct 0.25]
  python viz/probe.py checkpoints/exp002_..._step*.pt --out probes/
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BIN = "data/openwebtext_combined_bpe_owt50kv2.bin"
VAL_FRAC = 0.01
SEED = 1234
T = 256
N_WINDOWS = 4


def load_model(ckpt_path, rotary_pct, device):
    from model.transformer import Transformer
    st = torch.load(ckpt_path, map_location=device, weights_only=True)
    cfg = st.get("cfg", {})
    sd = st["model"] if "model" in st else {k: v for k, v in st.items()
                                            if torch.is_tensor(v)}
    model = Transformer(
        vocab_size=cfg["vocab_size"],
        context_length=cfg["context_length"],
        embedding_dim=cfg["embedding_dim"],
        num_heads=cfg["num_heads"],
        num_layers=cfg["num_layers"],
        use_rope=bool(cfg.get("use_rope", True)),
        rotary_pct=rotary_pct,
        dropout=0.0,
    )
    model.load_state_dict(sd, strict=True)
    return model.to(device), cfg, st


def probe_windows(device):
    n = os.path.getsize(BIN) // 4
    with np.errstate(all="ignore"):
        corpus = torch.from_numpy(np.memmap(BIN, dtype="<i4", mode="r", shape=(n,)))
    cut = int(n * (1.0 - VAL_FRAC))
    val = corpus[cut:]
    g = torch.Generator().manual_seed(SEED)
    idx = torch.randint(0, len(val) - T - 1, (N_WINDOWS,), generator=g)
    xs = torch.stack([val[int(i):int(i) + T] for i in idx]).long().to(device)
    ys = torch.stack([val[int(i) + 1:int(i) + 1 + T] for i in idx]).long().to(device)
    return xs, ys


@torch.no_grad()
def probe(model, xs, ys, device):
    from viz.hooks import capture_forward
    model.eval()
    agg = {"layer_norms": [], "eff_rank": [], "attn_entropy": [],
           "locality": []}
    ce = 0.0
    top1s, top10s = [], []
    L = None
    for x, y in zip(xs, ys):
        caps = capture_forward(model, x.unsqueeze(0))  # single dict, not a tuple
        logits = caps["logits"]
        # next-token CE: position i predicts x[i+1]
        logits_last = logits[0]  # [T, V]
        tgt = x[1:]
        ce += F.cross_entropy(logits_last[:-1], tgt).item()
        p = F.softmax(logits_last.float(), dim=-1)
        topv = torch.topk(p, 10, dim=-1).values
        top1s.append(topv[:, 0].mean().item())
        top10s.append(topv.sum(-1).mean().item())
        hid = [caps["embeddings"][0]] + [h[0] for h in caps["layer_hiddens"]]
        attn = caps["layer_attention"]  # [L, 1, heads?, T, T] or [L, heads, T, T]
        if L is None:
            L = len(hid)
            agg["layer_norms"] = [[] for _ in range(L)]
            agg["eff_rank"] = [[] for _ in range(L)]
            agg["attn_entropy"] = [[] for _ in range(L)]
            agg["locality"] = [[] for _ in range(L)]
        for li, h in enumerate(hid):
            hn = h.float()
            agg["layer_norms"][li].append(hn.norm(dim=-1).mean().item())
            z = hn - hn.mean(0, keepdim=True)
            s = torch.linalg.svdvals(z)
            agg["eff_rank"][li].append(((s.sum() ** 2) / (s ** 2).sum()).item())
        a = torch.stack(caps["layer_attention"], 0).detach().float()
        a = a.reshape(a.shape[0], -1, a.shape[-2], a.shape[-1])
        for li in range(len(hid) - 1):  # attn layers align to blocks 1..L-1
            m = a[li]  # [heads, T, T]
            ent = -(m.clamp_min(1e-9) * m.clamp_min(1e-9).log()).sum(-1)
            agg["attn_entropy"][li + 1].append(ent.mean().item())
            # locality: mass on distance<=1 within causal triangle
            Tm = m.shape[-1]
            dist = torch.arange(Tm, device=m.device)
            near = (dist[:, None] - dist[None, :] <= 1) & (dist[:, None] >= dist[None, :])
            agg["locality"][li + 1].append((m * near).sum(-1).mean().item())
    return {
        "ce_nats": ce / N_WINDOWS,
        "top1_p": sum(top1s) / len(top1s),
        "top10_p": sum(top10s) / len(top10s),
        "layer_norms": [sum(v) / len(v) for v in agg["layer_norms"] if v],
        "eff_rank": [sum(v) / len(v) for v in agg["eff_rank"] if v],
        "attn_entropy": [sum(v) / len(v) for v in agg["attn_entropy"] if v],
        "locality": [sum(v) / len(v) for v in agg["locality"] if v],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--rotary-pct", type=float, default=1.0,
                    help="0.25 for fix-era (post e6a526a) checkpoints!")
    ap.add_argument("--out", default="probes")
    ns = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    files = []
    for pat in ns.ckpts:
        files.extend(sorted(glob.glob(pat)) or [pat])
    xs, ys = probe_windows(device)
    os.makedirs(ns.out, exist_ok=True)
    for path in files:
        name = os.path.basename(path).replace(".pt", "")
        model, cfg, st = load_model(path, ns.rotary_pct, device)
        res = probe(model, xs, ys, device)
        res.update({"ckpt": name, "step": st.get("step"),
                    "rotary_pct": ns.rotary_pct,
                    "layers": cfg["num_layers"], "heads": cfg["num_heads"],
                    "ctx": cfg["context_length"], "vocab": cfg["vocab_size"]})
        outp = os.path.join(ns.out, name + ".json")
        json.dump(res, open(outp, "w", encoding="utf-8"), indent=1)
        bpb = res["ce_nats"] * 1.4427 / 4.413
        print(f"{name}: probe bpb~{bpb:.3f} top1={res['top1_p']:.3f} "
              f"rank_end={res['eff_rank'][-1]:.1f} "
              f"ent_mid={res['attn_entropy'][len(res['attn_entropy'])//2]:.2f} -> {outp}")
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
