"""Replay finished runs' curves through the new guard logic, offline.

The saturation guard only earns its place if it fires on the run that actually
saturated AND stays quiet on a run that was still learning. Both assertions are
made against REAL curves on disk, not synthetic ones:

  - the 70M full-epoch run (20260930_2055): saturated from ~step 12400. Must fire.
  - the eff-128k 5k probe (20260930_0907): still descending at its ceiling, no knee.
    Must NOT fire -- a guard that trips a healthy run is worse than no guard.

Run:  .venv/Scripts/python.exe tests/test_saturation_guard.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train.train import envelope, flat_envelope_stop  # noqa: E402

LAB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EPOCH = "20260930_2055_pythia_tokenizer_pile_train_full"
PROBE = "20260930_0907_pythia_tokenizer_pile_train_full"
# The run whose FALSE POSITIVE motivated the persistence rule (2026-10-08):
# stopped at step 8900/24694 after one spike. Only present on this machine.
LIVE = "20261007_2312_pythia160_tokenizer_pile_train_full"


def curve(run_dir):
    """(steps, val_bpb) for every eval in a run, in order."""
    path = os.path.join(LAB, "runs", run_dir, "loss.jsonl")
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
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


def first_flat_step(evals, window, delta, steps, flat_frac, val_every, min_step):
    """Earliest eval index at which the saturation guard would have fired."""
    hist = []
    for i, (step, bpb) in enumerate(evals, start=1):
        hist.append(bpb)
        if flat_envelope_stop(hist, window, delta, step, steps, flat_frac,
                              val_every, min_step=min_step):
            return step, bpb, i
    return None


def main():
    steps_total, val_every, min_steps_frac = 15136, 200, 0.5
    # CALIBRATED 2026-10-01 on the unbiased per-round envelope (see
    # tests/retune_saturation_guard.py, which sweeps these against both controls).
    # The previous values (frac 0.10, delta 0.005) sit in the cell the sweep proved
    # TRIPS the healthy 5k probe, so this test correctly failed against them.
    window, delta, flat_frac = 4, 0.005, 0.25
    min_step = int(min_steps_frac * steps_total)

    print(f"config: steps={steps_total} val_every={val_every} window={window} "
          f"delta={delta} flat_frac={flat_frac} min_step={min_step}")

    # --- positive control: the run that saturated ---
    epoch = curve(EPOCH)
    assert epoch, "no curve found for the epoch run"
    hit = first_flat_step(epoch, window, delta, steps_total, flat_frac,
                          val_every, min_step)
    print(f"\n[epoch run] {len(epoch)} evals, best {min(b for _, b in epoch):.5f}")
    if hit is None:
        print("  FAIL: guard did not fire on a run that saturated")
        return 1
    fire_step, fire_bpb, _ = hit
    best_step, best_bpb = min(epoch, key=lambda sb: sb[1])
    print(f"  fires at step {fire_step} (bpb {fire_bpb:.4f})")
    print(f"  true best was step {best_step} @ {best_bpb:.5f}")
    saved = best_step - fire_step
    # saved_steps / 37 steps-per-min = MINUTES. The old expression also multiplied by
    # val_every, inflating the saving 200x (it printed 2833 min for a 3400-step saving).
    print(f"  -> would have stopped {saved} steps early "
          f"({saved / 37.0:.0f} min at ~37 steps/min)")
    ok_pos = fire_step <= best_step and fire_step >= min_step
    print(f"  fires at or before the true best: {fire_step <= best_step} "
          f"(must be True, else it stops before the peak)")
    print(f"  fires after min_step {min_step}: {fire_step >= min_step}")

    # --- negative control: still descending ---
    probe = curve(PROBE)
    assert probe, "no curve found for the probe"
    probe_steps = 5000
    probe_hit = first_flat_step(probe, window, delta, probe_steps, flat_frac,
                                100, int(min_steps_frac * probe_steps))
    print(f"\n[5k probe] {len(probe)} evals, best {min(b for _, b in probe):.5f}")
    if probe_hit is None:
        print("  PASS: guard stayed silent on a still-descending run")
        ok_neg = True
    else:
        print(f"  FAIL: guard fired at step {probe_hit[0]} on a healthy run")
        ok_neg = False

    # --- the false positive the persistence rule exists to prevent ---
    live = curve(LIVE)
    if not live:
        print("\n[20261007_2312 parity run] absent (runs/ is gitignored) - skipping")
        ok_live = True
    else:
        live_hit = first_flat_step(live, window, delta, 24694, flat_frac, 100, 0)
        print(f"\n[20261007_2312 parity run] {len(live)} evals, "
              f"best {min(b for _, b in live):.5f}")
        if live_hit is None:
            print("  PASS: guard stayed silent - it stopped this run at step 8900 "
                  "before the persistence fix")
            ok_live = True
        else:
            print(f"  FAIL: guard fired at step {live_hit[0]} "
                  f"(bpb {live_hit[1]:.4f}) on a run that was still learning")
            ok_live = False

    # --- the sawtooth the envelope exists to absorb ---
    env, per = envelope([b for _, b in epoch[-16:]])
    raw = [b for _, b in epoch[-16:]]
    print(f"\n[sawtooth] last 16 raw evals span {min(raw):.4f}-{max(raw):.4f} "
          f"(range {max(raw) - min(raw):.4f})")
    print(f"  per-round envelope ({per}-eval cycle): "
          f"{[round(v, 4) for v in env]}")
    print(f"  envelope range {max(env) - min(env):.4f} "
          f"({(max(env) - min(env)) / max(1e-9, max(raw) - min(raw)):.0%} of raw range)")

    print("\n" + ("ALL CHECKS PASS" if (ok_pos and ok_neg and ok_live)
                    else "CHECKS FAILED"))
    return 0 if (ok_pos and ok_neg and ok_live) else 1


if __name__ == "__main__":
    sys.exit(main())
