#!/usr/bin/env python3
"""Resume-aware launcher for a supervised training job.

Reads train_job.json, finds the NEWEST checkpoint of that job's identity, and runs the
trainer with --resume against it. This is the piece that turns "checkpoints exist" into
"the run comes back by itself": the watchdog calls the scheduled task, the task runs this,
and this decides where to pick up.

Why the step-ckpt is preferred over _best.pt: _best.pt is by definition an EARLIER step
(whatever step last improved val), so resuming from it would throw away progress. The
best ckpt still matters - it is the keeper - but it is not the resume point.

--dry-run prints the decision and launches nothing.
"""
import glob
import json
import os
import re
import subprocess
import sys
from datetime import datetime

LAB = os.path.dirname(os.path.abspath(__file__))
SPEC = os.path.join(LAB, "train_job.json")
STATE = os.path.join(LAB, "train_job.state.json")


def load_spec(path=SPEC):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def newest_ckpt(spec):
    """(step, path) of the highest-step periodic ckpt for this job, else (None, best|None)."""
    step_pat = os.path.join(LAB, spec["ckpt_dir"], f"*_{spec['stamp']}_step*.pt")
    best = None
    for p in glob.glob(step_pat):
        m = re.search(r"_step(\d+)\.pt$", p)
        if not m:
            continue
        st = int(m.group(1))
        if best is None or st > best[0]:
            best = (st, p)
    if best:
        return best
    g = glob.glob(os.path.join(LAB, spec["ckpt_dir"], f"*_{spec['stamp']}_best.pt"))
    return (None, g[0]) if g else (None, None)


def build_cmd(spec, ckpt):
    cmd = [spec["python"], "-u", "-m", "train.train"] + list(spec["train_args"])
    if ckpt:
        cmd += ["--resume", os.path.relpath(ckpt, LAB).replace("\\", "/")]
    return cmd


def main():
    dry = "--dry-run" in sys.argv
    spec = load_spec()
    step, ckpt = newest_ckpt(spec)
    cmd = build_cmd(spec, ckpt)
    print(f"job={spec['name']} target={spec['target_steps']} steps")
    print(f"resume point: step={step} ckpt={os.path.basename(ckpt) if ckpt else 'NONE (fresh start)'}")
    print("cmd:", " ".join(cmd))
    if dry:
        return 0
    if step is not None and step >= spec["target_steps"]:
        print(f"already at target ({step} >= {spec['target_steps']}) - not launching")
        return 0

    log_path = os.path.join(LAB, spec["log"])
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = ""            # never let a foreign site-packages shadow torch
    env.pop("CUDA_VISIBLE_DEVICES", None)

    with open(log_path, "a", encoding="utf-8", buffering=1) as lf:
        lf.write(f"\n=== launch {datetime.now():%Y-%m-%d %H:%M:%S} "
                 f"resumed_from_step={step} ckpt={os.path.basename(ckpt) if ckpt else 'fresh'} ===\n")
        proc = subprocess.run(cmd, cwd=LAB, stdout=lf, stderr=subprocess.STDOUT, env=env)
        lf.write(f"=== trainer exited rc={proc.returncode} at {datetime.now():%H:%M:%S} ===\n")

    # State file is written by the WATCHDOG (it owns the restart ledger), but the launcher
    # records the outcome so a session can see how the last attempt ended.
    try:
        st = json.load(open(STATE, encoding="utf-8")) if os.path.exists(STATE) else {}
    except Exception:
        st = {}
    st.update({"last_launch": datetime.now().isoformat(timespec="seconds"),
               "last_resume_step": step, "last_exit_code": proc.returncode})
    with open(STATE, "w", encoding="utf-8") as f:
        json.dump(st, f, indent=2)
    print(f"trainer exited rc={proc.returncode}")
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
