"""Re-tune the saturation guard on the UNBIASED envelope, using real curves on disk.

The first calibration (2026-10-01, earlier) was fitted against the min-tracked series and
inherited its ~0.0399 bpb low bias, so its thresholds were meaningless. This replays both
control curves through the corrected guard and sweeps (flat_frac, flat_delta) to find
the region where the guard FIRES on the saturated run and STAYS QUIET on the healthy one.

Controls, both real:
  - epoch  20260930_2055...  saturated: the envelope pins from ~step 6200. Must fire.
  - probe  20260930_0907...  still descending at its 5k ceiling. Must NOT fire.

A guard that trips a healthy run is worse than no guard, so the negative control is the
binding constraint, not the positive one.

Run:  .venv/Scripts/python.exe tests/retune_saturation_guard.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train.train import envelope, eval_cycle_length, flat_envelope_stop  # noqa: E402

LAB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EPOCH = "20260930_2055_pythia_tokenizer_pile_train_full"
PROBE = "20260930_0907_pythia_tokenizer_pile_train_full"


def curve(run_dir):
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


def first_fire(evals, steps_total, val_every, flat_frac, flat_delta, min_frac=0.5):
    """Replay: at each eval, would the guard have stopped us? Return the first step."""
    hist = []
    min_step = int(min_frac * steps_total)
    for step, bpb in evals:
        hist.append(bpb)
        if flat_envelope_stop(hist, 4, flat_delta, step, steps_total, flat_frac,
                              val_every, min_step=min_step):
            return step, bpb, len(hist)
    return None


def main():
    epoch = curve(EPOCH)
    probe = curve(PROBE)
    assert epoch and probe, "missing control curves"

    STEPS, VAL_EVERY = 15136, 200
    PROBE_STEPS, PROBE_EVERY = 5000, 100

    print("=== detected eval cycle ===")
    p_epoch = eval_cycle_length([b for _, b in epoch])
    p_probe = eval_cycle_length([b for _, b in probe])
    print(f"  epoch run: period {p_epoch} evals")
    print(f"  probe run: period {p_probe} evals")

    print("\n=== the envelope (per-round mean) on the saturated run ===")
    env, per = envelope([b for _, b in epoch])
    steps = [s for i, (s, _) in enumerate(epoch) if (i + 1) % per == 0]
    for s, e in zip(steps, env):
        if s >= 4000:
            print(f"  step {s:6d}  envelope {e:.5f}")
    late = [e for s, e in zip(steps, env) if s >= 11000]
    if late:
        print(f"\n  late envelope mean {sum(late)/len(late):.5f} "
              f"(min-tracked best was 1.40325 -> bias "
              f"{sum(late)/len(late) - 1.40325:+.5f})")

    # The point where the envelope truly stops improving: last round that beat the
    # best-by-more-than-0.002, scanning forward from the halfway point.
    half = len(env) // 2
    best_seen, last_real = env[0], 0
    for i, e in enumerate(env):
        if i >= half and e < best_seen - 0.002:
            best_seen = e
            last_real = steps[i] if i < len(steps) else 0
    print(f"  last REAL improvement (>0.002) at step ~{last_real}")
    print("  -> anything firing much before this is stopping early; much later wastes time")

    print("\n=== sweep: flat_frac x flat_delta ===")
    print("  fire = step the guard aborts on the SATURATED run (want: >~8000, before 15136)")
    print("  probe = step it would abort on the HEALTHY run (want: none)\n")
    header = f"  {'frac':>6} {'delta':>7} {'fire@':>8} {'saved':>7} {'probe':>8}  verdict"
    print(header)
    print("  " + "-" * (len(header) - 2))
    good = []
    for frac in (0.10, 0.15, 0.20, 0.25, 0.30):
        for delta in (0.001, 0.002, 0.005):
            hit = first_fire(epoch, STEPS, VAL_EVERY, frac, delta)
            phit = first_fire(probe, PROBE_STEPS, PROBE_EVERY, frac, delta)
            fire = hit[0] if hit else None
            pstep = phit[0] if phit else None
            if fire is None:
                verdict = "never fires (USELESS)"
            elif pstep is not None:
                verdict = f"TRIPS HEALTHY RUN (bad)"
            elif fire < last_real * 0.8:
                verdict = "fires too early"
            else:
                verdict = "OK"
                good.append((frac, delta, fire))
            saved = f"{STEPS - fire:,}" if fire else "-"
            print(f"  {frac:6.2f} {delta:7.3f} "
                  f"{(str(fire) if fire else '-'):>8} {saved:>7} "
                  f"{(str(pstep) if pstep else '-'):>8}  {verdict}")

    print("\n=== recommendation ===")
    if not good:
        print("  No (frac, delta) satisfies both controls. Widen the search:")
        print("  the guard may need a longer min_step budget or a larger delta.")
        return 1
    # Pick the config that saves the MOST steps, not the latest-firing one. The sweep
    # sorts by fire step and `max` therefore selected a config that fires 136 steps from
    # the end -- technically valid, operationally useless. Saturation is real at ~11000,
    # so a good guard fires soon after that and must still pass the negative control.
    frac, delta, fire = max(good, key=lambda t: (STEPS - t[2], -t[0]))
    saved = STEPS - fire
    print(f"  flat_frac={frac:.2f}  flat_delta={delta:.3f}")
    print(f"  fires at step {fire:,} (run ends {STEPS:,})")
    # 37 steps/min was the 70M throughput. saved_steps / 37 = MINUTES. The old
    # expression multiplied by val_every as well, inflating the saving 200x (it
    # reported 1780 min for a 2136-step saving on a ~7h run).
    mins = saved / 37.0
    print(f"  saves {saved:,} steps ~= {mins:.0f} min ({mins/60:.1f} h) "
          f"at ~37 steps/min")
    print(f"  true saturation point ~{last_real:,} -> fires "
          f"{fire - last_real:+,} steps relative to it")
    print(f"  probe (healthy) unaffected: confirmed by the sweep")
    print(f"\n  flags: --flat-frac {frac} --flat-delta {delta}")
    print("\n  NOTE the sensitive cell: delta=0.002/0.005 at frac<=0.20 TRIPS THE HEALTHY")
    print("  PROBE. The margin here is one delta step, not a wide basin -- the probe only")
    print("  has 50 evals, so its 'true' endpoint is thin evidence. Treat delta=0.001 as")
    print("  the safe choice and re-verify on a longer healthy run before trusting it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
