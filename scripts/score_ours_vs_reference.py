#!/usr/bin/env python3
"""Compare a lab checkpoint with any local Pythia checkpoint on one shared harness.

The checkpoint's exact training-token budget is derived from its run log (or supplied
with --ours-tokens). The reference token budget comes from the Pythia step in the
snapshot directory name, preventing the old 70M-only denominator from leaking into a
160M comparison.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

LAB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(LAB, "scripts") not in sys.path:
    sys.path.insert(0, os.path.join(LAB, "scripts"))
import val_fixture  # noqa: E402

REF_REPORT = os.path.join(LAB, "reports", "pythia70m_reference.json")
LEGACY_BPT = 3.9104      # fallback only; main always measures bpt on the selected ids
PYTHIA_TOK_STEP = 2048 * 1024


def reference_tokens(ref_dir: str) -> tuple[int, str]:
    """(tokens_seen, label) for a Pythia snapshot dir.

    The suite ships 143 step branches; an unversioned dir like pythia70m_weights
    IS the final 143000-step checkpoint (299.9B tokens). Quoting its gap against
    a 2B-token run as a quality verdict is the exact error that produced the
    superseded "+18.7% behind", so the report states the denominator itself.
    """
    import re
    name = os.path.basename(os.path.normpath(ref_dir))
    # search, not fullmatch: this repo's dirs are pythia70m_step1000 /
    # pythia160m_step1000, and a fullmatch there silently fell through to
    # "final" -- labelling a 2B-token checkpoint as a 299.9B one (caught by
    # tests/test_val_fixture.py the first time it ran).
    m = re.search(r"step(\d+)", name)
    if m:
        step = int(m.group(1))
        return step * PYTHIA_TOK_STEP, f"step{step}"
    return 143000 * PYTHIA_TOK_STEP, "final (step143000, unversioned snapshot)"


def tokens_seen(step: int, train_args: dict, cfg: dict) -> int:
    """Exact optimizer-token budget: completed steps × micro-batch × accum × ctx."""
    return (int(step) * int(train_args["batch"])
            * int(train_args.get("accum", 1)) * int(cfg["context_length"]))


def _logits(model, bt, is_hf):
    """Logits for a batch, from either model, without touching the loss path.

    The two models have DIFFERENT forward signatures: our Transformer.forward(tokens)
    returns raw logits and takes no `labels`, while HF's takes `labels` and returns a loss.
    Calling each model's own loss path would mean scoring them through two different
    implementations, which is exactly the apples-to-oranges error this script exists to
    eliminate. So: take logits from both, and run ONE shared cross-entropy below.
    """
    with torch.no_grad():
        if is_hf:
            return model(bt).logits
        return model(bt)


def _mean_nll(logits, bt):
    """Mean NLL/token with a single shift, identical for both models.

    The shift is logits[..., :-1, :] against tokens[..., 1:]. Slicing the INPUT as well
    (a very easy mistake) double-shifts and makes every token score against its own
    predecessor -- that bug produced a nonsense bpb of 45.6 in the first version of the
    reference scorer. cross_entropy(reduction='mean') over [B, T-1, V] already averages
    per token, so no further normalisation is needed or wanted.
    """
    T = bt.shape[1]
    sl = logits[:, :T - 1, :].reshape(-1, logits.shape[-1])
    tl = bt[:, 1:].reshape(-1)
    return torch.nn.functional.cross_entropy(sl.float(), tl, reduction="mean")


def score_ids(model, ids, batches, batch, ctx, seed, dev, label, is_hf,
              bpt=LEGACY_BPT):
    """Mean NLL/token over non-overlapping windows. Shared by both models, unchanged.

    `bpt` is the bytes-per-token of the ids being scored (val_fixture.trainer_bpt,
    the trainer's own recipe). Dividing by a stale constant does not make two
    numbers comparable -- it makes them comparable-looking.
    """
    n = ids.shape[0]
    g = torch.Generator().manual_seed(seed)
    need = batches * batch * ctx
    if n < need:
        batches = max(1, n // (batch * ctx))
        print(f"  WARNING: only {n:,} tokens for {need:,}; reducing batches to {batches}")
    tot_nll, tot_tok = 0.0, 0
    t0 = time.time()
    for bi in range(batches):
        starts = torch.arange(batch, dtype=torch.long) * ctx + \
            int(torch.randint(0, max(1, n - (batches * batch * ctx)), (1,), generator=g))
        starts = starts.clamp(max=max(0, n - ctx - 1))
        bt = torch.stack([ids[s:s + ctx] for s in starts]).to(dev)
        logits = _logits(model, bt, is_hf)
        nll = _mean_nll(logits, bt)
        tot_nll += nll.item() * (bt.shape[0] * ctx)
        tot_tok += bt.shape[0] * ctx
        if bi % 20 == 0:
            print(f"  [{label}] batch {bi+1}/{batches}  running nll/tok {tot_nll/tot_tok:.4f}")
    nll = tot_nll / tot_tok
    bpb = nll * math.log2(math.e) / bpt
    print(f"  [{label}] nll/token {nll:.4f} nats  bpb {bpb:.4f}  "
          f"({tot_tok:,} tokens in {time.time()-t0:.0f}s)")
    return {"nll_per_token_nats": round(nll, 6), "bpb": round(bpb, 6),
            "tokens_scored": tot_tok}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--batches", type=int, default=60)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--ctx", type=int, default=None,
                    help="scoring context; default: checkpoint training context")
    ap.add_argument("--ours-tokens", type=int, default=None,
                    help="override lab checkpoint's optimizer-token budget; default derives from its run log")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--skip-reference", action="store_true")
    ap.add_argument("--val-frac", type=float, default=0.01)
    ap.add_argument("--cache", default=None,
                    help="token cache to take the val tail from "
                         "(default: val_fixture.resolve = the active run's cache)")
    ap.add_argument("--reference", default=os.path.join(
        LAB, "data", "incoming", "pythia70m_weights"),
        help="HF snapshot dir for the reference model")
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM

    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # Score the TOKEN CACHE TAIL. This is the whole point of the script: the trainer's
    # split_corpus() holds out ids [cut:] of the memmapped cache, so the comparison has
    # to use those exact ids. The historical fixture (pile_val_slice.txt) was produced by
    # decoding the tail to text and re-tokenizing it -- a lossy round trip that matches
    # the cache on only 1.68% of positions and diverges at index 42,843. Scoring both
    # models on decoded text is still "the same text for both", so the gap looks sane,
    # but it is not comparable to the trainer's own val_bpb and the absolute numbers
    # are wrong by ~0.04 bpb. Never route this through text again.
    # Score the TOKEN CACHE TAIL. This is the whole point of the script: the trainer's
    # split_corpus() holds out ids [cut:] of the memmapped cache, so the comparison has
    # to use those exact ids -- from the cache the run in question TRAINED on, which
    # val_fixture resolves (the old 2.004B tail sits inside the 160M training set).
    # The historical fixture (pile_val_slice.txt) was produced by decoding the tail to
    # text and re-tokenizing it: a lossy round trip matching the cache on 1.68% of
    # positions. Never route held-out ids through text again.
    cache_bin, _meta_path, meta = val_fixture.resolve(a.cache)
    ids_np = val_fixture.tail_ids(cache_bin, a.val_frac, meta)
    ids = torch.from_numpy(ids_np)
    bpt = val_fixture.trainer_bpt(ids_np, val_fixture.encoder_path(meta))
    print(f"val ids from TOKEN CACHE tail: {ids.shape[0]:,} tokens  bpt {bpt:.4f}"
          f"  device {dev}")
    print(f"  {val_fixture.describe(cache_bin, meta, a.val_frac)}")
    print("  (not pile_val_slice.txt -- that file is a lossy decode/encode round trip)")

    # ---- our checkpoint, rebuilt as a real HF model so the forward pass is identical
    from model.transformer import Transformer
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg_d = ck["cfg"]
    if a.ours_tokens is not None:
        ours_tokens = a.ours_tokens
        token_budget_source = "--ours-tokens"
    else:
        run_name = ck.get("run_name")
        log_path = os.path.join(LAB, "runs", run_name, "loss.jsonl") if run_name else None
        if not log_path or not os.path.isfile(log_path):
            raise SystemExit("cannot derive this checkpoint's token budget; pass --ours-tokens")
        with open(log_path, encoding="utf-8") as f:
            run_header = json.loads(next(f))
        ours_tokens = tokens_seen(ck["step"], run_header["args"], cfg_d)
        token_budget_source = log_path
    ctx = a.ctx if a.ctx is not None else int(cfg_d["context_length"])
    print(f"checkpoint token budget: {ours_tokens:,} (source: {token_budget_source})")
    print(f"scoring context: {ctx}")
    print(f"\ncheckpoint: step {ck.get('step')} cfg={ {k: cfg_d.get(k) for k in ('embedding_dim','num_layers','num_heads','context_length','vocab_size','use_learned_position_embeddings','lm_head_bias')} }")
    # Rebuild from the checkpoint's actual config so architecture choices (including
    # the optional learned position table) follow the saved weights exactly.
    ours = Transformer.from_config(cfg_d).to(torch.float32)
    ours.load_state_dict(ck["model"])
    ours.to(dev).eval()

    res = {"checkpoint": os.path.basename(a.ckpt), "checkpoint_step": ck.get("step"),
           "cache": cache_bin, "cache_desc": val_fixture.describe(cache_bin, meta, a.val_frac),
           "seed": a.seed, "batches": a.batches, "batch": a.batch,
           "ctx": ctx, "bytes_per_token": round(bpt, 6), "device": dev,
           "ours_tokens": ours_tokens, "token_budget_source": token_budget_source,
           "val_bpb_from_train_log": ck.get("val_bpb")}

    print("\n=== ours (this lab's checkpoint, same harness) ===")
    res["ours"] = score_ids(ours, ids, a.batches, a.batch, ctx, a.seed, dev,
                            "ours", is_hf=False, bpt=bpt)

    # ---- the reference, re-scored in the SAME process for a like-for-like delta
    if not a.skip_reference:
        print(f"\n=== reference weights, re-scored identically: {a.reference} ===")
        ref = AutoModelForCausalLM.from_pretrained(
            a.reference, torch_dtype=torch.float32)
        ref.to(dev).eval()
        res["reference_rerun"] = score_ids(ref, ids, a.batches, a.batch, ctx,
                                           a.seed, dev, "reference", is_hf=True,
                                           bpt=bpt)
        ref_tokens, ref_label = reference_tokens(a.reference)
        res["reference_rerun"]["tokens_seen"] = ref_tokens
        res["reference_rerun"]["revision"] = ref_label
        ratio = ref_tokens / ours_tokens
        res["comparability"] = (
            f"TOKEN-MATCHED ({ratio:.3f}x): ours {ours_tokens:,} vs reference "
            f"{ref_tokens:,} -- the gap is a like-for-like quality gap"
            if abs(ratio - 1.0) <= 0.10 else
            f"NOT TOKEN-MATCHED ({ratio:.3f}x): ours {ours_tokens:,} vs reference "
            f"{ref_tokens:,} ({ref_label}); do not interpret the gap as an equal-budget comparison.")

    stored = None
    if (os.path.basename(os.path.normpath(a.reference)) == "pythia70m_weights"
            and os.path.exists(REF_REPORT)):
        stored = json.load(open(REF_REPORT, encoding="utf-8")).get("bpb")
        res["reference_stored"] = stored

    print("\n=== HEAD TO HEAD ===")
    ours_bpb = res["ours"]["bpb"]
    if "reference_rerun" in res:
        r = res["reference_rerun"]["bpb"]
        print(f"  ours        {ours_bpb:.4f}")
        print(f"  reference   {r:.4f}   ({os.path.basename(os.path.normpath(a.reference))})")
        print(f"  gap         {ours_bpb - r:+.4f} bpb  = {100*(ours_bpb/r - 1):+.1f}%")
        print(f"  comparison  {res['comparability']}")
        print(f"  trainer's own val_bpb for this ckpt: {res['val_bpb_from_train_log']}")
    if stored:
        print(f"  stored reference {stored:.4f} | ours {ours_bpb:.4f} -> {ours_bpb - stored:+.4f}")
        print("  NOTE: the stored score is a different harness measurement; use the re-scored one.")

    ref_tag = os.path.basename(os.path.normpath(a.reference)).replace(".", "_")
    ckpt_tag = os.path.splitext(os.path.basename(a.ckpt))[0]
    out = os.path.join(LAB, "reports", f"ours_vs_{ref_tag}_{ckpt_tag}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w", encoding="utf-8"), indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
