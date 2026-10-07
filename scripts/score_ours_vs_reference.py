#!/usr/bin/env python3
"""Score OUR checkpoint on the SAME slice the pythia-70m reference was scored on.

WHY THIS EXISTS
    reports/pythia70m_reference.json says EleutherAI/pythia-70m scores bpb 1.2727 on
    data/incoming/pile_val_slice.txt. Our full-epoch run reports val bpb 1.40325 on the
    trainer's own held-out tail. Those two numbers were produced by DIFFERENT code paths
    (HF `model(batch, labels=batch)` vs the trainer's in-loop eval), so quoting the
    1.4033-vs-1.2727 gap as "we are 9.3% behind" is an apples-to-oranges claim until
    both are scored the same way.

    This closes the comparison properly: it loads OUR checkpoint into the SAME harness
    that produced the reference, on the SAME slice, with the SAME bpb formula. It also
    re-scores the reference weights in the same process, so any residual difference is
    attributable to the model rather than to the harness.

    VALIDITY RULES (borrowed from score_pythia70m_reference.py, deliberately identical)
    - Same IDS: the val tail of the ACTIVE run's token cache, read from the memmap by
      scripts/val_fixture.py. Never decoded to text. (The 2026-10-01 version of this
      script hardcoded the 70M cache; since the 160M run trains on the full-corpus
      cache, that tail would have been inside its training set.)
    - Same bpb convention: bpb = mean NLL * log2(e) / bytes-per-token, with
      bytes-per-token MEASURED on the scored ids (trainer's recipe), not a constant
      carried over from a previous tail.
    - Fixed seed, non-overlapping windows, HF shifts labels internally (do NOT slice).
    - Same ctx (512) and same number of scored tokens as the stored reference.

    CANNOT SETTLE: whether our trainer's in-loop val_bpb matches this harness. The
    trainer scores its own tail with its own eval path; this is a separate
    measurement of the same ids, so quote the harness number and the logged number
    as two instruments, not one.

Usage:
    .venv/Scripts/python.exe scripts/score_ours_vs_reference.py --ckpt <path> [--batches 60]
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

SLICE = os.path.join(LAB, "data", "incoming", "pile_val_slice.txt")
REF_REPORT = os.path.join(LAB, "reports", "pythia70m_reference.json")
LEGACY_BPT = 3.9104      # the OLD tail's bytes/token; --source slice only
OURS_TOKENS = 2_003_992_003
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
    ap.add_argument("--ctx", type=int, default=512)
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
    print(f"\ncheckpoint: step {ck.get('step')} cfg={ {k: cfg_d.get(k) for k in ('embedding_dim','num_layers','num_heads','context_length','vocab_size')} }")
    # Build exactly the way train/train.py builds it (Transformer + explicit kwargs, NOT a
    # ModelConfig object) so the state_dict keys match one-for-one. Reconstructing a config
    # and calling a different constructor is how you get a silent key mismatch.
    ours = Transformer(
        vocab_size=cfg_d["vocab_size"],
        context_length=cfg_d["context_length"],
        embedding_dim=cfg_d["embedding_dim"],
        num_heads=cfg_d["num_heads"],
        num_layers=cfg_d["num_layers"],
        use_rope=True,
        rotary_pct=cfg_d.get("rotary_pct", 0.25),
        dropout=0.0,
        parallel_residual=cfg_d.get("parallel_residual", True),
    ).to(torch.float32)
    ours.load_state_dict(ck["model"])
    ours.to(dev).eval()

    res = {"checkpoint": os.path.basename(a.ckpt), "checkpoint_step": ck.get("step"),
           "cache": cache_bin, "cache_desc": val_fixture.describe(cache_bin, meta, a.val_frac),
           "seed": a.seed, "batches": a.batches, "batch": a.batch,
           "ctx": a.ctx, "bytes_per_token": round(bpt, 6), "device": dev,
           "ours_tokens": OURS_TOKENS,
           "val_bpb_from_train_log": ck.get("val_bpb")}

    print("\n=== ours (this lab's checkpoint, same harness) ===")
    res["ours"] = score_ids(ours, ids, a.batches, a.batch, a.ctx, a.seed, dev,
                            "ours", is_hf=False, bpt=bpt)

    # ---- the reference, re-scored in the SAME process for a like-for-like delta
    if not a.skip_reference:
        print(f"\n=== reference weights, re-scored identically: {a.reference} ===")
        ref = AutoModelForCausalLM.from_pretrained(
            a.reference, torch_dtype=torch.float32)
        ref.to(dev).eval()
        res["reference_rerun"] = score_ids(ref, ids, a.batches, a.batch, a.ctx,
                                           a.seed, dev, "reference", is_hf=True,
                                           bpt=bpt)
        ref_tokens, ref_label = reference_tokens(a.reference)
        res["reference_rerun"]["tokens_seen"] = ref_tokens
        res["reference_rerun"]["revision"] = ref_label
        ratio = ref_tokens / OURS_TOKENS
        res["comparability"] = (
            f"TOKEN-MATCHED ({ratio:.3f}x): ours {OURS_TOKENS:,} vs reference "
            f"{ref_tokens:,} -- the gap is a like-for-like quality gap"
            if abs(ratio - 1.0) <= 0.10 else
            f"NOT TOKEN-MATCHED ({ratio:.0f}x): ours {OURS_TOKENS:,} vs reference "
            f"{ref_tokens:,} ({ref_label}). The gap measures distance to a FINISHED "
            f"model, not quality at equal training -- never quote it as a verdict. "
            f"Use scripts/score_matched_tokens.py for that.")

    stored = None
    if os.path.exists(REF_REPORT):
        stored = json.load(open(REF_REPORT, encoding="utf-8")).get("bpb")
        res["reference_stored"] = stored

    print("\n=== HEAD TO HEAD ===")
    ours_bpb = res["ours"]["bpb"]
    if "reference_rerun" in res:
        r = res["reference_rerun"]["bpb"]
        print(f"  ours        {ours_bpb:.4f}")
        print(f"  pythia-70m  {r:.4f}   (re-scored here)")
        print(f"  gap         {ours_bpb - r:+.4f} bpb  = {100*(ours_bpb/r - 1):+.1f}% worse")
        print(f"  trainer's own val_bpb for this ckpt: {res['val_bpb_from_train_log']}")
        print("  ^ NOT a verdict: the default reference is pythia-70m's FINAL")
        print("    checkpoint (299.9B tokens, 150x our 2.004B). It answers")
        print("    'how far is a finished model from ours', not 'are we ahead'.")
        print("    For the matched-token verdict run score_matched_tokens.py.")
    if stored:
        print(f"  stored reference {stored:.4f} | ours {ours_bpb:.4f} -> {ours_bpb - stored:+.4f}")
        print("  NOTE: if stored and re-scored differ, the stored number came from a")
        print("        different harness/seed path and the gap is not model quality.")

    out = os.path.join(LAB, "reports", "ours_vs_pythia70m.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w", encoding="utf-8"), indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
