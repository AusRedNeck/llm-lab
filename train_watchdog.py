#!/usr/bin/env python3
"""Training watchdog: keep a long run alive without a human noticing it died.

Run on a schedule (Hermes cron, every 10 min). Decides ONE of four things and exits:

  running   - the job's process is alive (matched by spec["match"] substrings in a python
              cmdline). Quiet, exit 0.
  complete  - the loss curve reached target_steps. Marked in the state file, exit 0.
  relaunch  - the job is dead and incomplete -> `schtasks /run` the parked training task
              (detached: parented to Task Scheduler, so an app restart cannot kill it).
              One line on stdout; exit 0.
  stalled   - it keeps dying WITHOUT making progress (a deterministic crash: OOM, shape
              error, bad corpus). Stops relaunching and exits 1 so the failure surfaces
              instead of looping forever.

Why progress is read from the loss curve and not from checkpoints: checkpoints land only on
500-step boundaries, so a run that trains 6500->6900 and then dies looks "unmoved" if you
compare checkpoints, and the guard would strike a healthy run. loss.jsonl advances every
step, so it is the honest progress signal. The ckpt step is the fallback.

Usage:
  python train_watchdog.py                 # normal (mutates state, may relaunch)
  python train_watchdog.py --dry-run       # decide + print, never touch anything
  python train_watchdog.py --spec <path>   # use an alternate job spec (testing)
"""
import glob
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta

LAB = os.path.dirname(os.path.abspath(__file__))
SPEC = os.path.join(LAB, "train_job.json")
TASK = "Hermes_TrainRun"          # parked scheduled task that launches the trainer detached
GRACE_MINUTES = 5                 # never relaunch within this window of the last attempt
STRIKE_LIMIT = 2                  # consecutive no-progress deaths before giving up


def state_path_for(spec_path):
    """State lives beside the spec it belongs to, so a test spec cannot clobber the real
    job's restart ledger."""
    return re.sub(r"\.json$", ".state.json", spec_path)


def load_json(path, default=None):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default if default is not None else {}


def save_state(st, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(st, f, indent=2)


def running_process(spec):
    """The first PYTHON process whose cmdline contains every spec['match'] substring.

    Two extra guards, both earned: (1) the name must be a python interpreter -- a shell,
    editor or agent process can have the job's own command text on its command line (a test
    harness that merely MENTIONS the needles matched itself); (2) the cmdline must contain
    the literal '-m train.train', which is how the trainer is always invoked."""
    try:
        import psutil
    except ImportError:
        return None
    needles = [n.lower() for n in spec.get("match", ["train.train"])]
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        name = (p.info.get("name") or "").lower()
        if not name.startswith("python"):
            continue
        try:
            cl = " ".join(p.info.get("cmdline") or []).lower()
        except Exception:
            continue
        if cl and "-m train.train" in cl and all(n.lower() in cl for n in needles):
            return p
    return None


def stuck_note(spec, st, step, now):
    """Warn when a process is alive but the curve has not moved for a long time.

    ~1.2 s/step here, so 30 minutes without one new step is not slow, it is hung (driver
    TDR, lost CUDA context, blocked I/O). We WARN rather than auto-kill: killing a live
    run is destructive and a human should decide."""
    if step is None:
        return None
    last_step, last_at = st.get("watchdog_last_step"), st.get("watchdog_last_at")
    if last_step != step or not last_at:
        return None
    try:
        idle = now - datetime.fromisoformat(last_at)
    except Exception:
        return None
    if idle >= timedelta(minutes=30):
        return (f"[watchdog] WARNING: pid alive but no new step for {int(idle.total_seconds() // 60)}m "
                f"(stuck at step {step}) - likely hung (driver/context), not slow. "
                f"Inspect {spec['log']}.")
    return None


def progress_step(spec):
    """Newest step the run has actually reached: loss.jsonl first, ckpt step second."""
    run_dir = os.path.join(LAB, "runs", spec["run_name"])
    lj = os.path.join(run_dir, "loss.jsonl")
    last = None
    if os.path.exists(lj):
        with open(lj, "rb") as f:
            try:
                f.seek(max(0, os.path.getsize(lj) - 65536))
                tail = f.read().decode("utf-8", "replace")
            except Exception:
                tail = ""
        for line in tail.splitlines():
            m = re.search(r'"step"\s*:\s*(\d+)', line)
            if m:
                last = int(m.group(1))
    step_ck = newest_ckpt_step(spec)
    return max([x for x in (last, step_ck) if x is not None], default=None)


def newest_ckpt_step(spec):
    best = None
    for p in glob.glob(os.path.join(LAB, spec["ckpt_dir"], f"*_{spec['stamp']}_step*.pt")):
        m = re.search(r"_step(\d+)\.pt$", p)
        if m:
            st = int(m.group(1))
            best = st if best is None else max(best, st)
    return best


def task_exists(task=TASK):
    r = subprocess.run(["schtasks", "/query", "/tn", task], capture_output=True, text=True)
    return r.returncode == 0


def main():
    dry = "--dry-run" in sys.argv
    quiet = "--quiet" in sys.argv        # cron passes this: healthy runs print nothing
    spec_path = SPEC
    if "--spec" in sys.argv:
        spec_path = sys.argv[sys.argv.index("--spec") + 1]
    spec = load_json(spec_path)
    STATE = state_path_for(spec_path)
    st = load_json(STATE, {})
    now = datetime.now()

    proc = running_process(spec)
    if proc is not None:
        step_now = progress_step(spec)
        note = stuck_note(spec, st, step_now, now)
        if not dry:
            st.update({"status": "running", "pid": proc.info["pid"],
                       "checked_at": now.isoformat(timespec="seconds")})
            if st.get("watchdog_last_step") != step_now:
                st["watchdog_last_step"] = step_now
                st["watchdog_last_at"] = now.isoformat(timespec="seconds")
            save_state(st, STATE)
        if not quiet:
            print(f"[watchdog] running: pid={proc.info['pid']} (job={spec['name']}) "
                  f"step={step_now}")
        if note:
            print(note)
        return 0

    step = progress_step(spec)
    target = spec["target_steps"]
    if step is not None and step >= target:
        if not dry:
            st.update({"status": "complete", "final_step": step, "completed": True,
                       "checked_at": now.isoformat(timespec="seconds")})
            save_state(st, STATE)
            spec["completed"] = True
            with open(spec_path, "w", encoding="utf-8") as f:
                json.dump(spec, f, indent=2)
        print(f"[watchdog] COMPLETE: {spec['name']} reached step {step}/{target} - no relaunch")
        return 0

    if spec.get("completed"):
        print(f"[watchdog] job marked complete - nothing to do")
        return 0

    # ---- not running, not complete: decide whether to relaunch -------------------------
    last_launch = st.get("last_relaunch")
    if last_launch:
        try:
            age = now - datetime.fromisoformat(last_launch)
            if age < timedelta(minutes=GRACE_MINUTES):
                if not quiet:
                    print(f"[watchdog] within {GRACE_MINUTES}m grace of last relaunch "
                          f"({age.seconds}s ago) - waiting")
                return 0
        except Exception:
            pass

    prev_step = st.get("last_resume_step")
    min_prog = spec.get("min_progress_steps", 25)
    if prev_step is not None and step is not None and step <= prev_step + min_prog:
        strikes = int(st.get("strikes", 0)) + 1
    else:
        strikes = 0

    if strikes >= STRIKE_LIMIT:
        if not dry:
            st.update({"status": "stalled", "strikes": strikes, "last_step": step,
                       "checked_at": now.isoformat(timespec="seconds")})
            save_state(st, STATE)
        print(f"[watchdog] STALLED: {spec['name']} died {strikes}x without advancing past "
              f"step {step} - NOT relaunching. Inspect {spec['log']} (likely OOM/shape/corpus).")
        return 1

    restarts = int(st.get("relaunch_count", 0)) + 1
    if restarts > spec.get("max_restarts", 6):
        if not dry:
            st.update({"status": "exhausted", "relaunch_count": restarts - 1,
                       "checked_at": now.isoformat(timespec="seconds")})
            save_state(st, STATE)
        print(f"[watchdog] EXHAUSTED: {spec['name']} hit max_restarts="
              f"{spec.get('max_restarts', 6)} - NOT relaunching.")
        return 1

    print(f"[watchdog] DEAD at step {step}/{target} - relaunching from newest checkpoint "
          f"(attempt {restarts}, strikes={strikes})"
          + ("  [dry-run: not actually launching]" if dry else ""))
    if dry:
        return 0

    if not task_exists():
        print(f"[watchdog] scheduled task {TASK} missing - cannot launch detached; "
              f"create it first (see llm-lab skill).")
        return 1
    r = subprocess.run(["schtasks", "/run", "/tn", TASK], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[watchdog] schtasks /run failed: {r.stderr.strip() or r.stdout.strip()}")
        return 1
    st.update({"status": "relaunching", "relaunch_count": restarts, "strikes": strikes,
               "last_relaunch": now.isoformat(timespec="seconds"),
               "last_resume_step": step if step is not None else st.get("last_resume_step"),
               "checked_at": now.isoformat(timespec="seconds")})
    save_state(st, STATE)
    print(f"[watchdog] relaunch dispatched via {TASK}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
