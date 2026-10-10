"""Health check for the 160M Pile parity run (p160-pile-parity-full).

Prints ONLY when something is wrong; empty stdout means healthy, and the cron
delivers nothing. This exists because the watchdog is correct to stay silent --
it latches a rule-stopped run as "final, latched - nothing to do" -- which is
exactly how this run sat dead for 8.5h, then 22h, unnoticed.

Alerts on:
  COMPLETE   the arm reached target_steps (retire this monitor)
  STOPPED    status latched terminal while step < target (with the reason)
  STALLED    status says running but loss.jsonl has not moved in 15 min
  STALE      the watchdog has not ticked in 40 min while the job is live

Run: .venv/Scripts/python.exe scripts/check_run_health.py   (exit 0 always;
     stdout is the signal)
"""
import datetime as dt
import json
import os
import re
import sys

LAB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPEC = os.path.join(LAB, "train_job.json")
STATE = os.path.join(LAB, "train_job.state.json")
CURVE = os.path.join(LAB, "runs", "20261007_2312_pythia160_tokenizer_pile_train_full",
                     "loss.jsonl")

TERMINAL = {"stopped-by-rule", "stalled", "exhausted", "quarantined", "complete"}
LIVE = {"running", "relaunching"}
CURVE_QUIET_MIN = 15          # a healthy trainer writes every ~100 s
WATCHDOG_QUIET_MIN = 40       # cron ticks every 10 min


def load(path, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def curve_state(path):
    """(last step, mtime) from the tail only -- curves are large."""
    try:
        with open(path, "rb") as f:
            f.seek(max(0, os.path.getsize(path) - 65536))
            tail = f.read().decode("utf-8", "replace")
    except Exception:
        return None, None
    last = None
    for line in tail.splitlines():
        m = re.search(r'"step"\s*:\s*(\d+)', line)
        if m:
            last = int(m.group(1))
    return last, os.path.getmtime(path)


def main():
    spec, st = load(SPEC, {}), load(STATE, {})
    target = int(spec.get("target_steps") or 0)
    step, mtime = curve_state(CURVE)
    status = str(st.get("status", "?"))
    now = dt.datetime.now()
    problems = []

    if step is not None and target and step >= target:
        print(f"COMPLETE: step {step}/{target} - the parity run finished; retire this monitor")
        return 0

    where = f"step {step if step is not None else '?'}/{target}"

    if status in TERMINAL:
        print(f"STOPPED: {where}, status={status}, relaunches={st.get('relaunch_count')}, "
              f"reason={st.get('reason', 'n/a')} - the run is NOT advancing")
        return 0

    if status in LIVE and mtime:
        quiet = (now.timestamp() - mtime) / 60
        if quiet > CURVE_QUIET_MIN:
            problems.append(f"STALLED: status={status} but loss.jsonl untouched for "
                            f"{int(quiet)} min ({where})")

    checked = st.get("checked_at")
    if checked and status in LIVE:
        try:
            age = (now - dt.datetime.fromisoformat(checked)).total_seconds() / 60
            if age > WATCHDOG_QUIET_MIN:
                problems.append(f"WATCHDOG STALE: last tick {int(age)} min ago "
                                f"(status={status}, {where})")
        except Exception:
            pass

    if not status in LIVE and not status in TERMINAL:
        problems.append(f"UNEXPECTED STATUS: {status!r} ({where})")

    for p in problems:
        print(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
