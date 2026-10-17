#!/usr/bin/env python3
"""Read an LR-cal bracket verdict JSON and apply the paired decision rule.

The verdict rung scores BOTH arms on ONE shared draw set (seed 1234) in a single
harness invocation:

    .venv/Scripts/python.exe scripts/eval_frozen_ladder_lrcal.py \
        --ckpts <armA step6000> <armB step6000> \
        --batches 40 --batch 16 --device cuda \
        --out logs/frozen_ladder_lr_cal_verdict.json --per-window

This script turns that JSON into a decision. Stdlib only, read-only.

What it computes (arm B relative to arm A, positive = B worse):
  - delta bpb from the harness's own aggregate estimates
  - the PAIRED per-window difference (same windows, so window difficulty
    cancels) -> mean, sd, SE, 95% CI half-width, in nats and in bpb
  - the conservative UNPAIRED floor for comparison (the 0.0138 bpb @640
    number measured on 2026-10-16)
  - the rule readout: |delta| <= ~0.03 bpb within noise (keep uncertainty),
    >= ~0.05 bpb beyond the conservative floor, in between = indeterminate

It REFUSES to render a verdict unless the two rows are a like-for-like pair:
same bytes_per_token, same ctx, same seed, same draw count, same step, and the
per-window arrays present and length-matched. Anything less is plumbing output,
labelled as such.

Usage:
    .venv/Scripts/python.exe scripts/lrcal_verdict_read.py <verdict.json> \
        [--arm-a-stem 202610140911] [--arm-b-stem 202610160941] \
        [--within 0.03] [--beyond 0.05]
"""
import argparse
import json
import math
import os
import sys

LOG2E = math.log2(math.e)


def die(msg, code=2):
    print(f"REFUSE: {msg}", file=sys.stderr)
    sys.exit(code)


def load_rows(path):
    if not os.path.exists(path):
        die(f"no such file: {path}")
    with open(path) as f:
        rows = json.load(f)
    if not isinstance(rows, list) or len(rows) < 2:
        die(f"need >= 2 checkpoint rows in {path}, got {type(rows).__name__} "
            f"with {len(rows) if isinstance(rows, list) else 'n/a'}")
    return rows


def pick(rows, stem, other_stem, label):
    if stem:
        hits = [r for r in rows if stem in os.path.basename(r.get("file", ""))]
        if len(hits) != 1:
            die(f"--{label}-stem {stem!r} matched {len(hits)} rows "
                f"(want exactly 1); files = "
                f"{[os.path.basename(r.get('file', '')) for r in rows]}")
        return hits[0]
    if other_stem:
        hits = [r for r in rows
                if other_stem not in os.path.basename(r.get("file", ""))]
        if not hits:
            die(f"no row left for {label} after excluding the other arm")
        return hits[0]
    die(f"internal: {label} has neither stem")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("verdict_json")
    ap.add_argument("--arm-a-stem", default=None,
                    help="substring of arm A's checkpoint filename (run stamp)")
    ap.add_argument("--arm-b-stem", default=None,
                    help="substring of arm B's checkpoint filename (run stamp)")
    ap.add_argument("--within", type=float, default=0.03,
                    help="|delta| at or below this (bpb) = within noise")
    ap.add_argument("--beyond", type=float, default=0.05,
                    help="|delta| at or above this (bpb) = beyond the floor")
    args = ap.parse_args()

    if args.arm_a_stem and args.arm_a_stem == args.arm_b_stem:
        die("--arm-a-stem and --arm-b-stem are identical")
    if args.within >= args.beyond:
        die(f"--within ({args.within}) must be < --beyond ({args.beyond})")

    rows = load_rows(args.verdict_json)
    if not args.arm_a_stem and not args.arm_b_stem:
        a, b = rows[0], rows[1]
        print("WARNING: no --arm-*-stem given -> ASSUMED row order is "
              "arm A first, arm B second.")
        print("         Confirm the two filenames printed below before "
              "quoting any number.\n")
    else:
        a = pick(rows, args.arm_a_stem, args.arm_b_stem, "arm-a")
        b = pick(rows, args.arm_b_stem, args.arm_a_stem, "arm-b")
    if a is b:
        die("both arms resolved to the same row")

    fa = os.path.basename(a.get("file", "?"))
    fb = os.path.basename(b.get("file", "?"))
    print(f"arm A : {fa}   step={a.get('step')}  val_bpb={a.get('val_bpb')}")
    print(f"arm B : {fb}   step={b.get('step')}  val_bpb={b.get('val_bpb')}")
    print(f"source: {os.path.abspath(args.verdict_json)}")
    print()

    # --- like-for-like gates (a failed gate downgrades this to plumbing) ---
    problems = []
    for key in ("bytes_per_token", "ctx", "seed", "draws", "step"):
        if a.get(key) != b.get(key):
            problems.append(f"{key}: A={a.get(key)!r} B={b.get(key)!r}")
    for name, row in (("A", a), ("B", b)):
        pw = row.get("per_window_val_nats")
        if not pw:
            problems.append(f"arm {name} has no per_window_val_nats "
                            f"(re-run the harness with --per-window)")
        elif not all(isinstance(z, (int, float)) and math.isfinite(z) for z in pw):
            problems.append(f"arm {name} per-window array has non-finite entries")
    pw_a, pw_b = a.get("per_window_val_nats") or [], b.get("per_window_val_nats") or []
    if pw_a and pw_b and len(pw_a) != len(pw_b):
        problems.append(f"per-window lengths differ: A={len(pw_a)} B={len(pw_b)}")

    # self-check the harness wrote arrays consistent with its own aggregates
    for name, row, pw in (("A", a, pw_a), ("B", b, pw_b)):
        if pw:
            m = sum(pw) / len(pw)
            selfcheck = m - row["val_nats"]
            if abs(selfcheck) > 1e-4:
                problems.append(
                    f"arm {name} per-window mean minus reported val_nats = "
                    f"{selfcheck:.3e} (>1e-4) - array and aggregate disagree")
            else:
                print(f"self-check arm {name}: array mean - val_nats = "
                      f"{selfcheck:+.2e} nats  OK")
    if pw_a and pw_b:
        print()

    if problems:
        print("NOT A VERDICT — pair failed like-for-like checks:")
        for p in problems:
            print(f"  - {p}")
        print("\n(aggregate readout below is plumbing only)")
        for name, row in (("A", a), ("B", b)):
            print(f"  arm {name}: val_bpb={row.get('val_bpb')} "
                  f"draws={row.get('draws')} ctx={row.get('ctx')} "
                  f"step={row.get('step')}")
        sys.exit(3)

    bpt = a["bytes_per_token"]
    fac = LOG2E / bpt
    n = len(pw_a)

    # aggregate delta
    d_agg_bpb = b["val_bpb"] - a["val_bpb"]

    # paired per-window differences (identical windows -> difficulty cancels)
    diffs = [y - x for x, y in zip(pw_a, pw_b)]
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1) if n > 1 else 0.0
    sd = math.sqrt(var)
    se_nats = sd / math.sqrt(n) if n > 1 else float("nan")
    mean_bpb = mean * fac
    se_bpb = se_nats * fac
    ci95 = 1.96 * se_bpb if n > 1 else float("nan")

    # cross-check: paired mean must agree with the aggregate delta
    agree = mean_bpb - d_agg_bpb
    agree_ok = abs(agree) < 0.01
    print(f"windows scored : {n}   (seed {a['seed']}, ctx {a['ctx']}, "
          f"bpt {bpt:.6f})")
    print(f"arm A val_bpb  : {a['val_bpb']:.6f}   (in-loop@save "
          f"{a.get('live_bpb_at_save')})")
    print(f"arm B val_bpb  : {b['val_bpb']:.6f}   (in-loop@save "
          f"{b.get('live_bpb_at_save')})")
    print(f"delta (B - A)  : {d_agg_bpb:+.6f} bpb")
    print(f"paired mean    : {mean:+.6f} nats = {mean_bpb:+.6f} bpb   "
          f"(cross-check vs aggregate {agree:+.2e} bpb "
          f"{'OK' if agree_ok else 'MISMATCH'})")
    print(f"paired sd      : {sd:.6f} nats")
    print(f"paired SE      : {se_bpb:.6f} bpb   95% CI "
          f"{mean_bpb - ci95:+.4f} .. {mean_bpb + ci95:+.4f}")
    print(f"unpaired floor : {args.beyond} bpb is the conservative bound "
          f"(paired SE above is the real one)")
    print()

    m = abs(d_agg_bpb)
    if m <= args.within:
        verdict = ("WITHIN NOISE — retain uncertainty; neither LR is "
                   "established as better at this resolution.")
    elif m >= args.beyond:
        if m > ci95:
            verdict = (f"BEYOND the conservative floor — arm B "
                       f"{'wins' if d_agg_bpb < 0 else 'loses'} by "
                       f"{m:.4f} bpb, and the gap also exceeds the paired "
                       f"95% CI ({ci95:.4f} bpb) by {m - ci95:+.4f}.")
        else:
            verdict = (f"BEYOND the conservative floor in magnitude, but "
                       f"INSIDE the paired 95% CI (+/-{ci95:.4f} bpb) — "
                       f"treat as indeterminate and report both numbers.")
    else:
        verdict = ("INDETERMINATE — between the noise band and the "
                   "conservative floor. Report both bpb values and the paired "
                   "CI; do not crown a winner.")

    print("VERDICT READOUT")
    print(f"  |delta| = {m:.4f} bpb vs within<={args.within} / "
          f"beyond>={args.beyond}")
    print(f"  {verdict}")
    print()
    print(f"  rule source: .hermes/plans/2026-10-16_lr-cal-external-harness.md "
          f"(noise floor 2026-10-16 20:55 PDT)")
    print("  NOTE: this reads ONE verdict JSON. The bracket decision rule "
          "still requires")
    print("  the final full-period val envelope under each arm's own cosine "
          "(arm A = step6000);")
    print("  an external-harness delta alone does not close the bracket.")


if __name__ == "__main__":
    main()
