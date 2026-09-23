#!/usr/bin/env python3
"""Run registry checker — the anti-circularity layer.

Reads experiments.json (one row per ARM), and for every run dir:
  1. maps it to <=1 arm or flags it `unregistered`;
  2. re-derives the arm's lever from the ACTUAL loss.jsonl header args of the arm
     and its control (a claimed lever that doesn't match the real arg diff is
     `diff-mismatch`; a real diff of >1 meaningful key is `confounded` — allowed
     only when the arm's lever note declares more than one key);
  3. --check exits 1 on any problem so a bad registry is impossible to miss.

Meaningful keys for the diff: everything in args EXCEPT bookkeeping keys below.
Code-level levers (e.g. parallel-residual fix in model/block.py) cannot appear in
headers; arms that carry one list it in lever with a 'code' key and the checker
treats 'code' as an accepted non-arg lever.

Usage:
  python viz/registry.py            # human report
  python viz/registry.py --check    # machine: exit 1 on problems
  python viz/registry.py --family capacity-50k
"""
import argparse
import glob
import json
import os
import sys

LAB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(LAB, "experiments.json")

# args keys that are bookkeeping/output, not experimental levers
NON_LEVER = {
    "out", "run_dir", "val_frac", "val_every", "resume", "data", "smoke",
    "patience", "min_delta", "patience_frac", "min_steps_frac", "degrade_frac",
    "warmup", "dropout_default", "seed",
    # horizon is arm budget, recorded per-row; not an experimental lever for
    # verdict comparison (verdicts compare on shared axes, see dashboard tokens axis)
    "steps",
}


def norm_args(args):
    """Collapse levers that are mechanically equivalent:
    batch x accum -> eff_batch (micro-batching is a VRAM detail when eff matches).
    tokenizer+tok_cache pair -> 'vocab' (one lever: which vocab/bin stream)."""
    a = dict(args)
    b, ac = a.pop("batch", None), a.pop("accum", None)
    if b is not None:
        a["eff_batch"] = b * (ac if ac else 1)
    a.pop("accum", None)
    if "tokenizer" in a or "tok_cache" in a:
        tok = os.path.basename(str(a.pop("tokenizer", "") or ""))
        a.pop("tok_cache", None)
        a["vocab"] = tok
    return a
# guard args intentionally added at arm time do not count as levers
GUARD_ARGS = {"patience-frac", "min-steps-frac", "degrade-frac", "val_every"}


def header_of(run_dir):
    lj = os.path.join(LAB, "runs", run_dir, "loss.jsonl")
    try:
        with open(lj, encoding="utf-8") as f:
            h = json.loads(f.readline())
        return h.get("args", {}), h.get("cfg", {}), h.get("params_m")
    except Exception:
        return None, None, None


def lever_diff(a, b):
    """keys where two arms' normalized arg dicts differ (bookkeeping excluded)."""
    a, b = norm_args(a), norm_args(b)
    keys = (set(a) | set(b)) - NON_LEVER
    return {k: (a.get(k), b.get(k)) for k in sorted(keys) if a.get(k) != b.get(k)}


def validate(a, ctl_args, args):
    """Compare claimed lever vs real normalized diff. Returns list of problems."""
    problems = []
    real = lever_diff(ctl_args, args)
    claimed = set(a["lever"]) - {"note"}
    meta = "code" in claimed  # code-level levers don't appear in headers
    arg_claims = claimed - {"code"}
    # map claim names onto the normalized diff's vocabulary; a "(NO-OP" claim
    # declares a change that is mechanically inert (e.g. null == preset default)
    claim_norm = set()
    noop_claims = set()
    for k in arg_claims:
        v = str(a["lever"].get(k, ""))
        target = "eff_batch" if k in ("batch", "accum") else k
        if "NO-OP" in v:
            noop_claims.add(target)
        claim_norm.add(target)
    touched = set(real)
    extra = touched - claim_norm
    if extra:
        problems.append(f"undeclared-lever: {sorted(extra)}")
    missing = claim_norm - touched
    if missing:
        problems.append(f"claimed-but-unchanged: {sorted(missing)}")
    if len(touched - noop_claims) > 1 and not meta and not a.get("accepted_confounded"):
        problems.append(f"confounded: {len(touched)} arg levers {sorted(touched)}")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--family")
    ns = ap.parse_args()

    meta = json.load(open(EXP, encoding="utf-8"))
    exp = meta["arms"]
    exp_meta = meta
    by_id = {a["id"]: a for a in exp}
    problems = []

    # run dir -> arm map (dupes are problems)
    owner = {}
    for a in exp:
        for d in a["run_dirs"]:
            if d in owner:
                problems.append(f"dup: {d} claimed by {owner[d]} and {a['id']}")
            owner[d] = a["id"]

    all_dirs = sorted(os.path.basename(p) for p in glob.glob(os.path.join(LAB, "runs", "*"))
                      if os.path.isdir(p))
    cutoff = exp_meta.get("legacy_cutoff", "00000000")
    ignored = set(exp_meta.get("ignore_dirs", []))
    legacy = [d for d in all_dirs if d[:8] < cutoff]
    unregistered = [d for d in all_dirs if d not in owner and d not in legacy and d not in ignored]
    phantom = [d for d in owner if d not in all_dirs]
    for d in phantom:
        problems.append(f"phantom: {d} registered but no run dir")
    if unregistered:
        problems.append(f"unregistered dirs ({len(unregistered)}): " + ", ".join(unregistered))

    rows = []
    for a in exp:
        if ns.family and a["family"] != ns.family:
            continue
        ctl = by_id.get(a["control_id"]) if a.get("control_id") else None
        status = []
        # sample newest run dir of the arm against control's newest
        d_new = a["run_dirs"][-1]
        args, cfg, params = header_of(d_new)
        if args is None:
            problems.append(f"{a['id']}: cannot read header of {d_new}")
            rows.append((a, "-", "header-unreadable"))
            continue
        if ctl:
            cargs, ccfg, cparams = header_of(ctl["run_dirs"][-1])
            if cargs is not None:
                probs = validate(a, cargs, args)
                status.extend(probs)
                rows.append((a, ctl["id"], "; ".join(status) or "ok"))
                continue
        rows.append((a, ctl["id"] if ctl else "-", "; ".join(status) or "no-control"))

    if not ns.check:
        fam = None
        for a, ctlid, verdict in sorted(rows, key=lambda r: (r[0]["family"], r[0]["id"])):
            if a["family"] != fam:
                fam = a["family"]
                print(f"\n== family: {fam} ==")
            print(f"  {a['id']:22s} v={a['verdict']:12s} ctl={ctlid:24s} {verdict}"
                  f"   dirs={len(a['run_dirs'])}")
        print(f"\nunregistered run dirs: {len(unregistered)}")
        for d in unregistered:
            print("  -", d)
        print(f"problems: {len(problems)}")
        for p in problems:
            print("  !", p)
    else:
        bad = [r for r in rows if r[2] not in ("ok", "no-control")] + problems
        for b in bad:
            print("PROBLEM:", b)
        print("CHECK: " + ("FAIL" if bad else "PASS"))
        sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
