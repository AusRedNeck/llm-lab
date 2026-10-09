"""Candidate repairs for flat_envelope_stop, scored against the three real curves.

V0 current  : stall = no prior round worse than newest+delta   (spike-sensitive)
V1 persist  : V0 must hold for the current AND the previous eval
V2 bestbased: V0 AND no new best (by delta) inside the stall window
V4 windowavg: mean of the window did not improve on the mean of the prior window

Wanted:  EPOCH fires (<= its true best step, >= min_step), PROBE silent, LIVE silent.

RESULT (2026-10-08): only V1 satisfies all three.
  EPOCH  V0 13000 | V1 13200 (true best 13400) | V2 14000 late | V4 15000 late
  PROBE  all four silent
  LIVE   V0 fires at 8900 (the false positive) | V1/V2/V4 silent
V1 shipped: it needs no new statistic, only agreement from the eval before, so a
single sawtooth point can no longer stop a run. V2/V4 also silence LIVE but arrive
too late to be useful as an early-stop guard, and V4 additionally needs 2x budget
history it may never get.

Run: .venv/Scripts/python.exe tests/sweep_flat_guard_persistence.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train.train import envelope  # noqa: E402

LAB = "D:/Projects/llm-lab"
DELTA = 0.005
FRAC = 0.25


def curve(run_dir):
    path = os.path.join(LAB, "runs", run_dir, "loss.jsonl")
    out = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if "step" in r and r.get("val_bpb") is not None:
            out.append((int(r["step"]), float(r["val_bpb"])))
    return out


def budget_rounds(steps, val_every, period, frac=FRAC):
    round_steps = max(1, period * max(1, val_every))
    return max(1, int(round(frac * steps / round_steps)))


def v0(hist, steps, val_every):
    """The shipped test."""
    env, period = envelope(hist)
    if period <= 1 or len(env) < 2:
        if len(hist) < 8:
            return False
        env, period = list(hist), 1
    rb = budget_rounds(steps, val_every, period)
    if rb + 1 > len(env):
        return False
    newest = env[-1]
    prior = env[-(rb + 1):-1]
    if any(p > newest + DELTA for p in prior):
        return False
    return True


def v1(hist, steps, val_every):
    return v0(hist, steps, val_every) and len(hist) >= 9 and v0(hist[:-1], steps, val_every)


def v2(hist, steps, val_every):
    if not v0(hist, steps, val_every):
        return False
    env, period = envelope(hist)
    if period <= 1 or len(env) < 2:
        env, period = list(hist), 1
    rb = budget_rounds(steps, val_every, period)
    if rb + 1 > len(env):
        return False
    win = env[-(rb + 1):]
    before = env[:-(rb + 1)]
    if not before:
        return False
    return (min(before) - min(win)) <= DELTA


def v4(hist, steps, val_every):
    env, period = envelope(hist)
    if period <= 1 or len(env) < 2:
        if len(hist) < 8:
            return False
        env, period = list(hist), 1
    rb = budget_rounds(steps, val_every, period)
    if 2 * rb + 1 > len(env):
        return False
    win = env[-(rb + 1):]
    prev = env[-(2 * rb + 1):-(rb + 1)]
    return (sum(prev) / len(prev)) - (sum(win) / len(win)) <= DELTA


VARIANTS = [("V0 current", v0), ("V1 persist", v1), ("V2 bestbased", v2), ("V4 windowavg", v4)]

CASES = [
    ("EPOCH (must fire)", "20260930_2055_pythia_tokenizer_pile_train_full",
     15136, 200, int(0.5 * 15136)),
    ("PROBE (must stay silent)", "20260930_0907_pythia_tokenizer_pile_train_full",
     5000, 100, int(0.5 * 5000)),
    ("LIVE (ours)", "20261007_2312_pythia160_tokenizer_pile_train_full",
     24694, 100, 0),
]

for name, run, steps_total, val_every, min_step in CASES:
    c = curve(run)
    ev = [b for _, b in c]
    best_step, best_bpb = min(c, key=lambda t: t[1])
    print(f"\n== {name}   best={best_bpb:.4f} @ {best_step}, min_step={min_step}")
    for vname, fn in VARIANTS:
        hit = None
        hist = []
        for step, bpb in c:
            hist.append(bpb)
            if step < min_step:
                continue
            if fn(hist, steps_total, val_every):
                hit = (step, bpb, len(hist))
                break
        if hit is None:
            print(f"   {vname:14s} fire: NO")
        else:
            flag = "OK" if hit[0] <= best_step else "TOO LATE (after best)"
            if name.startswith("PROBE"):
                flag = "FALSE POSITIVE"
            print(f"   {vname:14s} fire: step {hit[0]} bpb {hit[1]:.4f} eval#{hit[2]}  {flag}")
