#!/usr/bin/env python3
"""Quiet watcher for whichever llm-lab arm is currently armed.

WHY THIS SHAPE
    Runs under Hermes cron in `no-agent` mode: empty stdout delivers NOTHING,
    so silence is the healthy default and every line costs Shane an attention
    hit. This emits only when there is news:
      (a) a milestone step crossing,
      (b) a terminal event (early_stop / done. final) with final best bpb,
      (c) dead silence - the curve frozen past the freeze window with no
          terminal event, which catches the SUPERVISOR being broken rather
          than the run.

    It reads the ACTIVE arm out of train_job.json + train_job.state.json
    instead of hardcoding a run dir. A hardcoded watcher silently misfires the
    moment a different arm is armed, and editing cron per experiment is churn.
    Verdicts latch per job name in watch_active_arm.state.json so a finished
    run cannot re-alarm forever (that failure mode produced 121 consecutive
    failed cron ticks on a run that ended cleanly).

    Deliver to the CHAT (session-attach), not `local`: a local job's output is
    saved and nobody reads it.

Usage:  python watch_active_arm.py [--force] [--lab <llm-lab path>]   (--force ignores the latch)

Cron requires cron scripts to live in the Hermes scripts dir, which is NOT the
lab, so the lab root cannot be assumed to be the script's parent. Resolve it in
this order: --lab flag, LLM_LAB env var, then the script's parent (the normal
in-repo invocation). Without this the copied cron copy silently watches a
directory that has no train_job.json and reports nothing forever.
"""
import json
import os
import sys
import time


def _resolve_lab():
    """Find the llm-lab root, wherever this script was invoked from.

    Cron requires cron scripts to live in the Hermes scripts dir, which is NOT
    the lab, so the script's parent is not a usable signal for that deployment.
    Order: --lab flag, LLM_LAB env var, the script's parent (in-repo), the
    process CWD (cron's workdir), then the known install path. Without the
    final fallback the cron copy resolves to the scripts dir, finds no
    train_job.json, and reports nothing forever - silently, because empty
    stdout is the healthy case.
    """
    if "--lab" in sys.argv:
        return os.path.abspath(sys.argv[sys.argv.index("--lab") + 1])
    env = os.environ.get("LLM_LAB")
    if env:
        return os.path.abspath(env)
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if os.path.exists(os.path.join(here, "train_job.json")):
        return here
    cwd = os.path.abspath(os.getcwd())
    if os.path.exists(os.path.join(cwd, "train_job.json")):
        return cwd
    known = r"D:\Projects\llm-lab"
    if os.path.exists(os.path.join(known, "train_job.json")):
        return known
    return here


LAB = _resolve_lab()
SPEC = os.path.join(LAB, "train_job.json")
STATE = os.path.join(LAB, "train_job.state.json")
LATCH = os.path.join(LAB, "watch_active_arm.state.json")

# Pace: ~0.26 s/step measured on b16x2, so a healthy run adds steps constantly.
# 30 min of frozen curve with a live trainer pid means the loop is wedged, not
# merely between evals (val runs every 1000 steps ~= 4.5 min at this pace).
FREEZE_SECONDS = 30 * 60

# Milestones chosen around the failure zones of the arms this run is compared
# against. Crossing one IS the verdict, so they are worth interrupting for.
DEFAULT_MILESTONES = [1000, 2000, 5000, 10000, 20000, 30000, 40000, 60000,
                      80000, 100000, 120000]


def load_json(path, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def say(msg):
    print(msg, flush=True)


def read_curve(run_dir):
    """Return (step, mtime) from a run's loss.jsonl.

    Deliberately does NOT try to read val/bpb from here: the trainer writes the
    per-step row to loss.jsonl but prints val + bpb to the LOG only (val_bpb
    lands in checkpoints/_best.pt and the log, never in the curve). A watcher
    that looks for bpb in loss.jsonl reports "n/a" at every milestone forever.
    read_val() below is the authority for the number.
    """
    p = os.path.join(LAB, "runs", run_dir, "loss.jsonl")
    if not os.path.exists(p):
        return 0, 0
    step, mtime = 0, 0
    try:
        mtime = os.stat(p).st_mtime
        with open(p, "rb") as f:
            for line in f:
                s = line.decode("utf-8", "replace").strip()
                if not s:
                    continue
                try:
                    d = json.loads(s)
                except Exception:
                    continue          # tolerate a torn line; the curve is append-only
                st = d.get("step")
                if isinstance(st, int) and st > step:
                    step = st
    except Exception:
        pass
    return step, mtime


def read_val(log_path):
    """(best_bpb, last_bpb, last_val_step) parsed from the trainer LOG.

    Lines look like:  step  1000/121091 loss=5.7412 avg50=5.8914 val=5.7868 bpb=2.1350 lr=5.0e-04
    """
    best, last, last_step = None, None, 0
    if not log_path or not os.path.exists(log_path):
        return best, last, last_step
    try:
        with open(log_path, encoding="utf-8", errors="replace") as f:
            for line in f:
                if "bpb=" not in line or "val=" not in line or not line.startswith("step"):
                    continue
                try:
                    step = int(line.split()[1].split("/")[0])
                    b = float(line.split("bpb=")[1].split()[0])
                except Exception:
                    continue
                if b > 0:
                    best = b if best is None else min(best, b)
                    last, last_step = b, step
    except Exception:
        pass
    return best, last, last_step


def terminal_from_state(st):
    return st.get("status") in (
        "stopped-by-rule", "complete", "exhausted", "stalled", "quarantined")


def main():
    force = "--force" in sys.argv
    spec = load_json(SPEC, {})
    st = load_json(STATE, {})
    job = spec.get("name")
    if not job:
        return 0
    target = spec.get("target_steps", 0)

    latch = load_json(LATCH, {})
    # A different job owns a different latch: never inherit its silence.
    if latch.get("job") != job:
        latch = {"job": job, "milestones": [], "terminal": False}
    done_milestones = set(latch.get("milestones", []))

    run_dir = spec.get("run_name") or ""
    step, mtime = read_curve(run_dir) if run_dir else (0, 0)
    best, last_bpb, val_step = read_val(spec.get("log"))
    status = st.get("status")

    def bpb_txt():
        return f"{best:.4f}" if best is not None else "n/a (no val yet)"

    # (b) terminal event - report once with the number it must be judged on.
    if terminal_from_state(st) or spec.get("completed"):
        if not latch.get("terminal") or force:
            latch["terminal"] = True
            with open(LATCH, "w", encoding="utf-8") as f:
                json.dump(latch, f, indent=2)
            say(f"[llm-lab] {job}: {status or 'completed'} — "
                f"final best val bpb {bpb_txt()} "
                f"(last val @ step {val_step or 'n/a'}, curve reached {step}/{target}).")
        return 0

    # (a) milestone crossing
    for m in DEFAULT_MILESTONES:
        if step >= m and m not in done_milestones:
            done_milestones.add(m)
            say(f"[llm-lab] {job}: step {m}/{target} — best val bpb {bpb_txt()}.")
    if done_milestones != set(latch.get("milestones", [])):
        latch["milestones"] = sorted(done_milestones)
        with open(LATCH, "w", encoding="utf-8") as f:
            json.dump(latch, f, indent=2)

    # (c) dead silence: curve frozen with no terminal event. This is the
    # supervisor-broken detector, and it must stay quiet while step advances.
    if step > 0 and mtime and (time.time() - mtime) > FREEZE_SECONDS:
        if not latch.get("frozen") or force:
            latch["frozen"] = True
            with open(LATCH, "w", encoding="utf-8") as f:
                json.dump(latch, f, indent=2)
            say(f"[llm-lab] ALERT {job}: curve FROZEN at step {step}/{target} for "
                f"{int((time.time() - mtime) / 60)} min with no terminal event. "
                f"Trainer or watchdog is wedged — check it.")
    else:
        latch["frozen"] = False

    return 0


if __name__ == "__main__":
    sys.exit(main())
