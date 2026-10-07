#!/usr/bin/env python3
"""Score OUR 70M against pythia-70m at MATCHED TOKENS, not against its asymptote.

WHY THIS EXISTS
    Shane's read: "it's not as far off as that percentage makes it seem and we put 2B
    tokens through it not 300B." He is right, and the +18.7% figure was measured against
    the WRONG DENOMINATOR. We compared our 2.004B-token run against pythia-70m's FINAL
    300B-token checkpoint. That is a 150x token mismatch presented as a quality gap.

    The Pythia suite ships 143 log-spaced checkpoints (step1000 .. step143000). At
    2,097,152 tokens/step, step143000 == 299.89B tokens, confirming that schedule. Our
    2,003,992,003 tokens == pythia step 955.6, i.e. step1000 is a 1.046x match -- within
    5% on tokens. So the honest comparison exists and is cheap to make.

    Also note we are PAST Chinchilla here: 2.004B / 70.74M params = 28.3 tok/param
    against an optimal of 20. We overshot the compute-optimal point, which is a
    legitimate strategy (and explains the late saturation), not a failure.

    CONTEXT MISMATCH, HANDLED EXPLICITLY: pythia-70m was trained at ctx 2048 and we
    train at 512. The RoPE frequencies it learned are baked into its weights, so scoring
    it at 512 is scoring it OUT OF DISTRIBUTION. Two ways to handle that, and this script
    reports BOTH so neither can be quietly cherry-picked:
      - "native": score pythia at ctx 512, same windows as ours. Fair on DATA, unfair to
        pythia -- its rotary positions at 512 are not what it was trained for.
      - "2048-rescore": score pythia at its native 2048 with proper non-overlapping
        2048-token windows, and ours at 512. Fair to pythia, but then the two numbers
        are not measured on identical windows and the comparison is weaker.
    The defensible headline is the 512-vs-512 one (identical ids, identical windows, one
    harness); the 2048 number is context for how much of the residual gap is CONTEXT and
    how much is TRAINING.

Run: .venv/Scripts/python.exe scripts/score_matched_tokens.py
"""
import json
import math
import os
import statistics
import sys

import numpy as np
import torch

LAB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(LAB, "scripts") not in sys.path:
    sys.path.insert(0, os.path.join(LAB, "scripts"))
import val_fixture  # noqa: E402

CKPT = os.path.join(LAB, "checkpoints",
                    "exp002_pythia_tokenizer_202609302055_step13500.pt")
WEIGHTS = os.path.join(LAB, "data", "incoming", "pythia70m_step1000")
# NOTE: the token-MATCHED reference. data/incoming/pythia70m_weights is the
# 300B-token final (step143000) and is the WRONG denominator for a 2B-token
# comparison -- using it produced the misleading +18.7%.
OUT = os.path.join(LAB, "reports", "matched_tokens_70m.json")
VAL_FRAC = 0.01
OURS_TOKENS = 2_003_992_003
PYTHIA_STEP = 1000
PYTHIA_TOK_STEP = 2048 * 1024


def main():
    from model.transformer import Transformer
    from transformers import AutoModelForCausalLM

    cache_bin, _meta_path, meta = val_fixture.resolve()
    ids_np = val_fixture.tail_ids(cache_bin, VAL_FRAC, meta)
    ids = torch.from_numpy(ids_np)
    bpt = val_fixture.trainer_bpt(ids_np, val_fixture.encoder_path(meta))
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"val ids from TOKEN CACHE tail: {ids.shape[0]:,}  bpt {bpt:.4f}  "
          f"device {dev}")
    print(f"  {val_fixture.describe(cache_bin, meta, VAL_FRAC)}")
    print(f"\n=== token matching ===")
    pyt = PYTHIA_STEP * PYTHIA_TOK_STEP
    print(f"  ours            {OURS_TOKENS:>15,} tokens  (28.3 tok/param)")
    print(f"  pythia@{PYTHIA_STEP:<7d} {pyt:>15,} tokens  (1.046x ours)")
    print(f"  pythia@143000   {143000*PYTHIA_TOK_STEP:>15,} tokens  <- what we compared against")

    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    c = ck["cfg"]
    ours = Transformer(vocab_size=c["vocab_size"], context_length=c["context_length"],
                       embedding_dim=c["embedding_dim"], num_heads=c["num_heads"],
                       num_layers=c["num_layers"], use_rope=True,
                       rotary_pct=c.get("rotary_pct", 0.25), dropout=0.0,
                       parallel_residual=c.get("parallel_residual", True)).to(torch.float32)
    ours.load_state_dict(ck["model"])
    ours.to(dev).eval()
    ref = AutoModelForCausalLM.from_pretrained(WEIGHTS, dtype=torch.float32)
    ref.to(dev).eval()
    log2e = math.log2(math.e)

    def nll_of(lg, y):
        return torch.nn.functional.cross_entropy(
            lg.reshape(-1, lg.shape[-1]).float(), y.reshape(-1),
            reduction="mean").item()

    def score(model, is_hf, ctx, n_regions, batch=8, max_batches=30):
        """Non-overlapping windows of `ctx` tokens, stratified over the val tail.

        max_batches caps work per region. Uncapped, the loop wanted ~305 batches per
        region (1.25M tokens of a 20M val tail / 4096-token batches) x 16 regions =
        4,880 forward passes, which is ~16 minutes for a number that was already precise
        to +/-0.01 bpb with 8 regions x 30 batches. Sampling depth is a budget choice,
        not a correctness one: the gap's regional sd was 0.0099, so a 0.010 bpb CI
        needs ~4 regions, and 8 x 30 x 8 x 512 = 9.8M tokens is comfortably past that.
        """
        n = ids.shape[0]
        span = n - ctx - 2
        out = []
        for r in range(n_regions):
            start = r * (span // n_regions)
            off, tot, ntok, nb = start, 0.0, 0, 0
            while off + batch * ctx + 1 <= start + (span // n_regions) and nb < max_batches:
                nb += 1
                xb = torch.stack([ids[off + k * ctx: off + k * ctx + ctx]
                                  for k in range(batch)])
                yb = torch.stack([ids[off + k * ctx + 1: off + k * ctx + ctx + 1]
                                  for k in range(batch)])
                xb, yb = xb.to(dev), yb.to(dev)
                with torch.no_grad():
                    o = model(xb)
                lg = o.logits if is_hf else o
                tot += nll_of(lg, yb) * yb.numel()
                ntok += yb.numel()
                off += batch * ctx
            if ntok:
                out.append(tot / ntok)
        return out

    N_REG = 8
    print(f"\n=== head to head, ctx 512 for BOTH (identical ids, identical windows) ===")
    o512 = [v * log2e / bpt for v in score(ours, False, 512, N_REG)]
    r512 = [v * log2e / bpt for v in score(ref, True, 512, N_REG)]
    print(f"  ours @512        {statistics.mean(o512):.4f}  (sd {statistics.stdev(o512):.4f})")
    print(f"  pythia@{PYTHIA_STEP} @512 {statistics.mean(r512):.4f}  "
          f"(sd {statistics.stdev(r512):.4f})")
    g512 = [a - b for a, b in zip(o512, r512)]
    print(f"  gap              {statistics.mean(g512):+.4f}  "
          f"(sd {statistics.stdev(g512):.4f})  "
          f"= {100*statistics.mean(g512)/statistics.mean(r512):+.1f}%")
    print("  ^ pythia is being scored OUT OF DISTRIBUTION at 512 (trained at 2048),")
    print("    so this UNDERSTATES pythia. It is the fair-on-data number.")

    print(f"\n=== pythia at its NATIVE ctx 2048 (context for the split, 6 regions) ===")
    r2048 = [v * log2e / bpt for v in score(ref, True, 2048, 6)]
    print(f"  pythia@{PYTHIA_STEP} @2048 {statistics.mean(r2048):.4f}  "
          f"(sd {statistics.stdev(r2048):.4f})")
    ctx_penalty = statistics.mean(r512) - statistics.mean(r2048)
    print(f"  ctx-512 penalty on pythia: {ctx_penalty:+.4f} bpb")
    g2048 = statistics.mean(o512) - statistics.mean(r2048)
    print(f"  gap vs native-ctx pythia:  {g2048:+.4f}  "
          f"= {100*g2048/statistics.mean(r2048):+.1f}%")
    print("  ^ NOT apples-to-apples (different window sizes), but it bounds how much")
    print("    of the gap is the context mismatch we impose, not training quality.")

    out = {
        "fixture": {
            "cache": os.path.basename(cache_bin),
            "cache_path": cache_bin,
            "val_desc": val_fixture.describe(cache_bin, meta, VAL_FRAC),
            "bytes_per_token": round(bpt, 6),
            "scored_on": "2026-10-06",
            "supersedes": "the same comparison on the 2.004B cache tail "
                          "(20,039,921 ids, bpt 3.9104): that region sits inside "
                          "the 160M run's training set, so it cannot be the "
                          "shared fixture any more",
        },
        "token_matched": {
            "ours_tokens": OURS_TOKENS,
            "ours_tok_per_param": round(OURS_TOKENS / 70_739_072, 1),
            "pythia_revision": f"step{PYTHIA_STEP}",
            "pythia_tokens": pyt,
            "token_ratio_pythia_over_ours": round(pyt / OURS_TOKENS, 3),
        },
        "ctx512_both": {
            "ours_bpb": round(statistics.mean(o512), 4),
            "pythia_bpb": round(statistics.mean(r512), 4),
            "gap_bpb": round(statistics.mean(g512), 4),
            "gap_pct": round(100 * statistics.mean(g512) / statistics.mean(r512), 1),
            "caveat": "pythia scored out-of-distribution at 512 (trained at 2048)",
        },
        "pythia_native_2048": {
            "bpb": round(statistics.mean(r2048), 4),
            "ctx512_penalty_on_pythia": round(ctx_penalty, 4),
            "gap_vs_native": round(g2048, 4),
            "caveat": "different window size vs ours; bounds the context penalty only",
        },
        "asymptote_comparison_superseded": {
            "pythia_143000_tokens": 143000 * PYTHIA_TOK_STEP,
            "gap_pct": 18.7,
            "why_superseded": "compared 2.004B tokens against 299.9B tokens, a 150x "
                             "token mismatch presented as a quality gap",
        },
    }
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
