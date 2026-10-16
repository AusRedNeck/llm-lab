#!/usr/bin/env python3
"""Fire-ready launcher for LR-cal bracket ARM B (2e-4) — ONE command, zero traps.

WHY THIS EXISTS
---------------
Arm A (`p160-2k-lr-cal-3e-4`) finished its chunk at step 6000 and is parked at
`paused_at_chunk` awaiting Shane's explicit go. The go for arm B was reduced to a
markdown procedure (`.hermes/plans/2026-10-15_lr-cal-arm-b-fire-ready.md`) that a
human or agent has to hand-transcribe into a shell. That procedure carries three
documented traps, all reproduced and fixed on 2026-10-16:

  1. `--lever lr=3e-4->2e-4` unquoted in bash/PowerShell is a REDIRECTION: the
     shell reads `lr=3e-4-` and writes stdout into a file named `2e-4`. The
     registry then records a lever of `3e-4-` and the single-lever claim of the
     whole calibration is silently destroyed. (Reproduced 10-16: a 0-byte `2e-4`
     appeared in the repo root.)
  2. `--dry-run` placed AFTER `--` becomes a *trainer* arg (arm.py uses
     argparse.REMAINDER) and trips the active-job guard with exit 1.
  3. `.venv/Scripts/python.exe train_watchdog.py` — that path moved to
     `scripts/` (commit 68c058d); the root path no longer exists.

This script removes all three structurally: it builds the argv as a Python list
and hands it to `subprocess` with `shell=False`, so no shell ever sees `>`.

SAFETY MODEL
------------
Default invocation is a DRY RUN. It runs every precondition against the live
tree and prints exactly what `--execute` would do, writing NOTHING.
`--execute` is required to touch `train_job.json`, `experiments.json`, or the
archive copies — that is a GPU-training process-control action and is Shane's
call, not this script's.

USAGE
    python scripts/arm_b_fire.py             # dry run: verify + print, write nothing
    python scripts/arm_b_fire.py --execute   # archive arm A, arm arm B
    python scripts/arm_b_fire.py --execute --launch-now
                                             # ... and kick scripts/train_watchdog.py
                                             # once (default: let cron b7b9bb50fd4f
                                             # pick the new spec up in <=10 min)
"""
import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime

LAB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPEC = os.path.join(LAB, "train_job.json")
STATE = os.path.join(LAB, "train_job.state.json")
EXP = os.path.join(LAB, "experiments.json")

CONTROL = "p160-2k-lr-cal-3e-4"
ARM_B = "p160-2k-lr-cal-2e-4"
FAMILY = "pythia160-2k-lr-calibration"
TARGET = 6000
SCHEDULE = 16000

# Single lever vs arm A: --lr 3e-4 -> 2e-4. Nothing else differs, not even order.
GOAL = ("LR calibration B: compare LR=2e-4 against 3e-4 under the same 16000-step "
        "global cosine, ending at global step 6000; identical Pythia160_2k, Pile "
        "cache, effective batch, warmup, eval cadence and seed as p160-2k-lr-cal-3e-4 "
        "(single lever: lr); choose via final full-period envelope and retain uncertainty")

TRAINER_ARGS = [
    "--preset", "pythia160_2k",
    "--tokenizer", "data/incoming/pythia70m_hf/tokenizer.json",
    "--tok_cache", "data/pile_train_full_bpe_pythia_hf.bin",
    "--corpus", "data/pile_train_full.txt",
    "--lr", "2e-4",
    "--steps", "16000",
    "--stop-at-step", "6000",
    "--batch", "1", "--accum", "64", "--warmup", "160",
    "--val_every", "100", "--val_frac", "0.01", "--out", "checkpoints",
    "--patience-frac", "0", "--degrade-frac", "0.30", "--flat-frac", "0",
    "--flat-delta", "0.005", "--recycle-margin", "0",
]

ARM_ARGV_PREFIX = [
    os.path.join(LAB, ".venv", "Scripts", "python.exe"),
    os.path.join(LAB, "viz", "arm.py"),
    "--id", ARM_B,
    "--family", FAMILY,
    "--control", CONTROL,
    "--lever", "lr=3e-4->2e-4",          # a single list element: never a redirection
    "--max-restarts", "6",
    "--goal", GOAL,
    "--",                                 # trainer args follow
]

checks = []   # (ok:bool, label, detail)


def check(ok, label, detail=""):
    checks.append((bool(ok), label, detail))
    return bool(ok)


def jload(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def live_train_procs():
    """python processes whose command line looks like this lab's trainer/watchdog."""
    ps = ("Get-CimInstance Win32_Process | Where-Object { $_.Name -like 'python*' } | "
          "Select-Object -ExpandProperty CommandLine")
    try:
        out = subprocess.run(["powershell.exe", "-NoProfile", "-Command", ps],
                             capture_output=True, text=True, timeout=60).stdout
    except Exception as exc:                       # pragma: no cover
        return [f"<probe failed: {exc}>"]
    hits = []
    for line in out.splitlines():
        low = line.lower()
        if "llm-lab" in low and ("train" in low or "watchdog" in low):
            hits.append(line.strip())
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true",
                    help="actually archive arm A and arm arm B (default: dry run)")
    ap.add_argument("--launch-now", action="store_true",
                    help="after arming, run scripts/train_watchdog.py once")
    ns = ap.parse_args()
    stamp = f"{datetime.now():%Y-%m-%d %H:%M:%S %Z}"
    print(f"arm_b_fire — {stamp}")
    print(f"mode: {'EXECUTE' if ns.execute else 'DRY RUN (writes nothing)'}\n")

    # ---- 1. arm A's live spec -------------------------------------------------
    spec = jload(SPEC)
    check(spec.get("name") == CONTROL, "train_job.json is arm A",
          f"name={spec.get('name')!r}")
    check(spec.get("completed") is False, "arm A spec still open (completed=false)")
    check(spec.get("target_steps") == TARGET and spec.get("schedule_steps") == SCHEDULE,
          "arm A chunk is 6000 of a 16000 cosine",
          f"target={spec.get('target_steps')} schedule={spec.get('schedule_steps')}")
    check(spec.get("next_job") in (None, ""), "arm A has no auto-chain next_job",
          f"next_job={spec.get('next_job')!r}")

    # ---- 2. arm A is genuinely parked ----------------------------------------
    st = jload(STATE)
    check(st.get("status") == "paused_at_chunk", "arm A status is paused_at_chunk",
          f"status={st.get('status')!r}")
    check(st.get("paused_step") == TARGET, "arm A paused at exactly step 6000",
          f"paused_step={st.get('paused_step')}")
    check(st.get("pause_warning") is None, "chunk marker present (no pause_warning)",
          f"pause_warning={st.get('pause_warning')!r}")

    # ---- 3. arm A's step-6000 evidence is on disk ----------------------------
    run_name = st.get("run_name") or ""
    m = re.match(r"(\d{8})_(\d{4})_", run_name)
    arm_stem = None
    if m:
        arm_stem = f"*{m.group(1)}{m.group(2)}*_step{TARGET}.pt"
    ckpt = sorted(glob.glob(os.path.join(LAB, "checkpoints", arm_stem or f"*_step{TARGET}.pt"))) if arm_stem else []
    check(bool(ckpt), "arm A step-6000 checkpoint on disk",
          "; ".join(os.path.basename(c) for c in ckpt) or "NOT FOUND")

    runs = os.path.join(LAB, "runs", run_name)
    lossf = os.path.join(runs, "loss.jsonl")
    val_row = None
    marker_row = False
    if os.path.exists(lossf):
        with open(lossf, encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                if row.get("chunk_complete") and row.get("step") == TARGET:
                    marker_row = True
                if row.get("step") == TARGET and row.get("val_bpb") is not None:
                    val_row = row
    check(val_row is not None, "arm A step-6000 full-period val row on disk",
          f"val_bpb={val_row.get('val_bpb') if val_row else None} "
          f"served_bpb={val_row.get('served_bpb') if val_row else None}")

    # The chunk-complete marker lives in TWO places; require both. The trainer's
    # prose line goes to the log file, the structured row to loss.jsonl.
    logf = os.path.join(LAB, (spec.get("log") or "").replace("\\", "/"))
    marker_log = False
    if os.path.exists(logf):
        with open(logf, encoding="utf-8", errors="replace") as f:
            for line in f:
                if f"chunk complete at step {TARGET}/{SCHEDULE}" in line:
                    marker_log = True
    check(marker_log, "chunk-complete prose line in the trainer log",
          spec.get("log") or "<no log path in spec>")
    check(marker_row, "chunk_complete:true row in loss.jsonl")

    # ---- 4. nothing is currently training ------------------------------------
    procs = live_train_procs()
    check(not procs, "no llm-lab train/watchdog process alive",
          "; ".join(procs) if procs else "none found")

    # ---- 5. arm B is not already registered ----------------------------------
    meta = jload(EXP)
    ids = {a["id"] for a in meta["arms"]}
    fam = [a for a in meta["arms"] if a.get("family") == FAMILY]
    check(ARM_B not in ids, "arm B id is free in the registry", f"arms={len(ids)}")
    check(CONTROL in ids, "arm A is registered as the control")
    check(len(fam) >= 1, f"family {FAMILY} present", f"arms in family={len(fam)}")

    # ---- 6. inputs -----------------------------------------------------------
    for rel in ("data/incoming/pythia70m_hf/tokenizer.json",
                "data/pile_train_full_bpe_pythia_hf.bin",
                "data/pile_train_full.txt"):
        p = os.path.join(LAB, rel)
        check(os.path.exists(p), f"input present: {rel}",
              f"{os.path.getsize(p)/1e9:.2f} GB" if os.path.exists(p) else "MISSING")

    # ---- 7. single-lever proof: diff arm A's train_args vs arm B's -----------
    a_args = spec.get("train_args") or []
    diffs = [(i, x, y) for i, (x, y) in enumerate(zip(a_args, TRAINER_ARGS)) if x != y]
    check(len(a_args) == len(TRAINER_ARGS), "arm B argv is the same length as arm A's",
          f"armA={len(a_args)} armB={len(TRAINER_ARGS)}")
    check(len(diffs) == 1 and a_args[diffs[0][0]] == "3e-4"
          and TRAINER_ARGS[diffs[0][0]] == "2e-4",
          "EXACTLY ONE token differs, and it is --lr 3e-4 -> 2e-4",
          "; ".join(f"idx {i}: {x!r} -> {y!r}" for i, x, y in diffs) or "no diffs")

    # ---- report --------------------------------------------------------------
    bad = [c for c in checks if not c[0]]
    print("PRECONDITIONS")
    for ok, label, detail in checks:
        print(f"  [{'OK ' if ok else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))
    print(f"\n  {len(checks) - len(bad)}/{len(checks)} passed")
    if bad:
        print("\nABORT: preconditions failed. Nothing was written.")
        for _, label, detail in bad:
            print(f"  - {label}: {detail}")
        return 2

    argv = ARM_ARGV_PREFIX + TRAINER_ARGS
    print("\nWHAT --execute WOULD DO (in order)")
    print("  1. cp train_job.json       -> train_job.p160-2k-lr-cal-3e-4.done.json")
    print("  2. cp train_job.state.json -> train_job.p160-2k-lr-cal-3e-4.state.done.json")
    print("  3. set train_job.json 'completed': true  (so arm.py's guard passes;")
    print("     with next_job=None the watchdog then no-ops on arm A: no auto-chain)")
    print("  4. run viz/arm.py with this argv (shell=False, so '>' can never redirect):")
    print("       " + " ".join(a if " " not in a else repr(a) for a in argv))
    print("  5. verify: train_job.json name == " + ARM_B + ", registry arms == "
          f"{len(ids) + 1}")
    print("  6. launch: " + ("scripts/train_watchdog.py once (--launch-now)"
                            if ns.launch_now else
                            "nothing — cron b7b9bb50fd4f (every 10 m) picks the new "
                            "spec up and launches it; its 'DEAD at step None/6000 - "
                            "relaunching' line is NORMAL wording for a fresh arm"))
    print("\n  Arm B runtime: 6000 steps x ~9.2 s/step ~= 15.3 h.")
    if not ns.execute:
        print("\nDRY RUN — nothing written, nothing launched. Re-run with --execute.")
        return 0

    # ================= EXECUTE =================
    print("\nEXECUTING")
    ts = f"{datetime.now():%Y%m%d_%H%M%S}"
    a_done = os.path.join(LAB, f"train_job.{CONTROL}.done.json")
    s_done = os.path.join(LAB, f"train_job.{CONTROL}.state.done.json")
    shutil.copy2(SPEC, a_done)
    shutil.copy2(STATE, s_done)
    print(f"  archived -> {os.path.basename(a_done)}, {os.path.basename(s_done)}")

    spec["completed"] = True
    spec["notes"] = (spec.get("notes", "") + f" | completed by arm_b_fire {ts} "
                                             f"(chunk paused at 6000; archived)").strip(" |")
    with open(SPEC, "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2)
    print("  train_job.json marked completed=true (arm A archived first)")

    res = subprocess.run(argv, cwd=LAB, shell=False, capture_output=True, text=True)
    sys.stdout.write(res.stdout)
    sys.stderr.write(res.stderr)
    if res.returncode != 0:
        print(f"\nFAILED: arm.py exited {res.returncode}. Arm A is archived and marked "
              f"completed; re-run arm.py by hand or restore from {os.path.basename(a_done)}.")
        return res.returncode

    new_spec = jload(SPEC)
    new_ids = {a["id"] for a in jload(EXP)["arms"]}
    ok1 = new_spec.get("name") == ARM_B and new_spec.get("train_args") == TRAINER_ARGS
    ok2 = ARM_B in new_ids
    print(f"  verified: train_job.json name={new_spec.get('name')!r} args-match={ok1}; "
          f"registry has {ARM_B}={ok2} (arms={len(new_ids)})")
    if not (ok1 and ok2):
        print("FAILED VERIFICATION — inspect before launching anything.")
        return 3

    if ns.launch_now:
        wd = os.path.join(LAB, "scripts", "train_watchdog.py")
        r2 = subprocess.run([os.path.join(LAB, ".venv", "Scripts", "python.exe"), wd],
                            cwd=LAB, shell=False)
        print(f"  train_watchdog.py -> exit {r2.returncode}")
    else:
        print("  not launching: cron b7b9bb50fd4f fires within <=10 min and will "
              "launch it. Use --launch-now to skip the wait.")

    print(f"\nARMED {ARM_B} at {datetime.now():%Y-%m-%d %H:%M:%S}. "
          f"Verify from live records: train_job.state.json, "
          f"logs/{ARM_B}_chunk01.log advancing past step 1, nvidia-smi busy.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
