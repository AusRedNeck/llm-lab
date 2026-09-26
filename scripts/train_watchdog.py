#!/usr/bin/env python3
"""Training watchdog: keep a long run alive without a human noticing it died.

Run on a schedule (Hermes cron, every 10 min). Decides ONE of four things and exits:

  running   - the job's process is alive (matched by spec["match"] substrings in a python
              cmdline). Quiet, exit 0.
  complete  - the loss curve reached target_steps. Marked in the state file, exit 0.
  relaunch  - the job is dead and incomplete -> launches directly via subprocess.Popen
              with DETACHED_PROCESS (survives app restart). Falls back to schtasks/VBS
              if direct launch fails.
  stalled   - it keeps dying WITHOUT making progress (a deterministic crash: OOM, shape
              error, bad corpus). Stops relaunching and exits 1 so the failure surfaces
              instead of looping forever.

Why progress is read from the loss curve and not from checkpoints: checkpoints land only on
500-step boundaries, so a run that trains 6500->6900 and then dies looks "unmoved" if you
compare checkpoints, and the guard would strike a healthy run. loss.jsonl advances every
step, so it is the honest progress signal. The ckpt step is the fallback.

Two rules that keep this honest, both paid for in production:

  * Identify a run by its FAMILY, not by the exact run_name/log the spec claims. The trainer
    self-stamps its own run dir when the spec carries no stamp, so the spec drifts from what
    was actually written. Matching on spec fields made the finish test blind to a completed
    run, which then got relaunched until the restart budget ran out: 121 consecutive failed
    cron ticks on a run that had ended cleanly by its own early-stop rule.
  * LATCH a terminal verdict. Once a run is finished/complete/stalled/exhausted that is the
    answer for that job, so later ticks exit 0 and stay silent. Re-deriving the same verdict
    every 10 minutes turned one finished run into a permanent red alarm -- and because the
    alert itself failed to deliver, nobody saw it.

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

_here = os.path.dirname(os.path.abspath(__file__))
# The watchdog lives in scripts/ but every path it touches (train_job.json, checkpoints/,
# logs/, data/) hangs off the repo root. Resolve the root from the script's own location so
# moving the script between the root and scripts/ cannot silently point SPEC at nothing
# (2026-09-23: the move made SPEC=scripts/train_job.json, load_json returned {}, and every
# 10-minute tick died on KeyError 'target_steps' -- the watchdog was off for ~25h).
LAB = _here if os.path.exists(os.path.join(_here, "train_job.json")) else os.path.dirname(_here)
SPEC = os.path.join(LAB, "train_job.json")
WATCHDOG_LOCK_PATH = os.path.join(LAB, ".train-watchdog.lock")
TASK = "Hermes_TrainRun"          # parked scheduled task (kept for fallback; direct launch preferred)
GRACE_MINUTES = 5                 # never relaunch within this window of the last attempt
STRIKE_LIMIT = 2                  # consecutive no-progress deaths before giving up
# Verdicts that are FINAL for one job. Once recorded they latch: later ticks stay silent
# instead of re-deriving the same conclusion and failing the cron run forever.
TERMINAL = ("stopped-by-rule", "complete", "exhausted", "stalled")
HERMES_PYTHON = "C:/Users/shane/AppData/Local/hermes/hermes-agent/venv/Scripts/python.exe"
DETACHED = 0x00000008             # Windows DETACHED_PROCESS flag


def launch_direct(spec, spec_path=None):
    """Launch the trainer directly via subprocess.Popen (detached, no schtasks/VBS/bat).

    Returns True on success, False on failure. The process is parented to nothing
    (DETACHED_PROCESS), so an app restart cannot kill it.

    Relaunches RESUME from the job's newest step ckpt (fresh when none is newer than the
    spec) -- without this a crash burned the run's whole compute history on restart.
    """
    trainers, can_tell = running_trainers()
    if not can_tell:
        print("[watchdog] GLOBAL PREFLIGHT BLIND: cannot inspect trainers; refusing launch")
        return False
    if trainers:
        pids = [getattr(p, "pid", p) for p in trainers]
        print(f"[watchdog] GLOBAL PREFLIGHT BLOCKED: trainer already alive "
              f"(pid={pids}); refusing launch")
        return False
    cmd = [spec["python"], "-u", "-m", "train.train"] + list(spec["train_args"])
    resume = newest_step_ckpt(spec, spec_path)
    if resume:
        cmd += ["--resume", os.path.relpath(resume[1], LAB).replace("\\", "/")]
    log_path = os.path.join(LAB, spec.get("log", "logs/trainer.log"))
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    env = dict(os.environ)
    env["PYTHONPATH"] = ""
    env.pop("CUDA_VISIBLE_DEVICES", None)

    try:
        lf = open(log_path, "a", encoding="utf-8", buffering=1)
        proc = subprocess.Popen(
            cmd, cwd=LAB, stdout=lf, stderr=subprocess.STDOUT,
            env=env, creationflags=DETACHED,
            close_fds=True,
        )
        print(f"[watchdog] DIRECT LAUNCH: pid={proc.pid} "
              f"resume={('step' + str(resume[0])) if resume else 'fresh'} "
              f"cmd={' '.join(cmd[:6])}...")
        return True
    except Exception as e:
        print(f"[watchdog] DIRECT LAUNCH FAILED: {e}")
        return False


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


def _iter_psutil_processes():
    import psutil
    return psutil.process_iter(["pid", "name", "cmdline"])


class TrainerProbeError(RuntimeError):
    pass


def _is_trainer(process):
    info = getattr(process, "info", None) or {}
    name = info.get("name")
    if not isinstance(name, str):
        candidate = getattr(process, "name", "")
        try:
            name = candidate() if callable(candidate) else candidate
        except Exception as exc:
            raise TrainerProbeError("cannot read process name") from exc
    if not isinstance(name, str):
        return False
    name = name.lower()
    if not name.startswith("python"):
        return False
    args = info.get("cmdline")
    if args is None:
        candidate = getattr(process, "cmdline", None)
        try:
            args = candidate() if callable(candidate) else candidate
        except Exception as exc:
            raise TrainerProbeError("cannot read process command line") from exc
    if not isinstance(args, (list, tuple)):
        raise TrainerProbeError("invalid process command line")
    args = [str(arg).lower() for arg in args]
    return any(args[i] == "-m" and args[i + 1] == "train.train"
               for i in range(len(args) - 1))


def _running_trainers_wmic():
    query = ("Get-CimInstance Win32_Process -Filter \"Name LIKE 'python%'\" | "
             "ForEach-Object { \"$($_.ProcessId)`t$($_.CommandLine)\" }")
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", query],
            capture_output=True, text=True, timeout=60)
    except Exception:
        return [], False
    if result.returncode != 0:
        return [], False
    found = []
    for line in result.stdout.splitlines():
        pid_text, _, cmdline = line.partition("\t")
        if "-m train.train" not in cmdline.lower():
            continue
        try:
            found.append(int(pid_text.strip()))
        except ValueError:
            continue
    return found, True


def running_trainers():
    """Return every live llm-lab trainer, independent of the active job match."""
    try:
        processes = list(_iter_psutil_processes())
    except ImportError:
        return _running_trainers_wmic()
    except Exception:
        return [], False
    found = []
    for process in processes:
        try:
            if _is_trainer(process):
                found.append(process)
        except TrainerProbeError:
            return [], False
    return found, True

def running_process(spec):
    """The first PYTHON process whose cmdline contains every spec['match'] substring.

    Two extra guards, both earned: (1) the name must be a python interpreter -- a shell,
    editor or agent process can have the job's own command text on its command line (a test
    harness that merely MENTIONS the needles matched itself); (2) the cmdline must contain
    the literal '-m train.train', which is how the trainer is always invoked.

    Returns (proc_or_None, can_tell). can_tell=False means the probe itself was blind
    (no psutil under this interpreter -- observed live 2026-09-23: cron ran the watchdog
    under a python without psutil, running_process returned None, and the tick launched a
    SECOND trainer beside a healthy one). Blind must NEVER be read as dead: caller treats
    can_tell=False as no-op."""
    needles = [n.lower() for n in spec.get("match", ["train.train"])]
    try:
        import psutil
    except ImportError:
        # Cron may run us under an interpreter without psutil (observed 2026-09-23).
        # Fall back to a stdlib-only probe instead of going blind.
        return _running_process_wmic(needles)
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        name = (p.info.get("name") or "").lower()
        if not name.startswith("python"):
            continue
        try:
            cl = " ".join(p.info.get("cmdline") or []).lower()
        except Exception:
            continue
        if cl and "-m train.train" in cl and all(n.lower() in cl for n in needles):
            return p, True
    return None, True


def _running_process_wmic(needles):
    """Stdlib-only process probe via WMI, for interpreters without psutil.

    Returns (match, can_tell). can_tell=False only if WMI itself fails — same
    blind-is-not-dead rule. Match rules mirror the psutil path: python-named
    process whose cmdline contains '-m train.train' + every needle."""
    import subprocess
    q = ("Get-CimInstance Win32_Process -Filter \"Name LIKE 'python%'\" | "
         "ForEach-Object { \"$($_.ProcessId)`t$($_.CommandLine)\" }")
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-Command", q],
                           capture_output=True, text=True, timeout=60)
    except Exception:
        return None, False
    if r.returncode != 0:
        return None, False
    for line in r.stdout.splitlines():
        pid_s, _, cl = line.partition("\t")
        cl = (cl or "").lower()
        if "-m train.train" in cl and all(n in cl for n in needles):
            try:
                return int(pid_s.strip()), True
            except ValueError:
                continue
    return None, True


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


def run_family(spec):
    """The run-dir family a spec belongs to: run_name minus its leading YYYYMMDD_HHMM_ stamp.

    The trainer SELF-STAMPS its own run dir whenever the spec carries no stamp, so a spec
    that says run_name 1332 can end up with a curve in the 1344 dir (observed live
    2026-09-19: train_job.json said 20260919_1332, the trainer wrote 20260919_1344). Keying
    the finish test off the exact run_name made the watchdog blind to a completed run and
    relaunch it until the restart budget ran out -- 121 consecutive failed cron ticks.
    Matching on the family (everything after the stamp) makes that drift harmless.
    """
    rn = spec.get("run_name") or ""
    return re.sub(r"^\d{8}_\d{4}_", "", rn)


def curve_step(lj):
    """Newest step recorded in a loss curve, or None. Reads the tail only (curves get big)."""
    try:
        with open(lj, "rb") as f:
            f.seek(max(0, os.path.getsize(lj) - 65536))
            tail = f.read().decode("utf-8", "replace")
    except Exception:
        return None
    last = None
    for line in tail.splitlines():
        m = re.search(r'"step"\s*:\s*(\d+)', line)
        if m:
            last = int(m.group(1))
    return last


def find_run_dir(spec):
    """The run dir this job actually writes to: exact run_name when its curve exists, else the
    NEWEST dir of the same family (ranked by curve step, then mtime). None if neither exists."""
    named = spec.get("run_name")
    if named:
        exact = os.path.join(LAB, "runs", named)
        if os.path.exists(os.path.join(exact, "loss.jsonl")):
            return exact
    fam = run_family(spec)
    if not fam:
        return None
    cands = [d for d in glob.glob(os.path.join(LAB, "runs", "*" + fam)) if os.path.isdir(d)]
    if not cands:
        return None

    def score(d):
        lj = os.path.join(d, "loss.jsonl")
        step = curve_step(lj) if os.path.exists(lj) else None
        return (step if step is not None else -1, os.path.getmtime(d))

    return max(cands, key=score)


def finished_log(spec, st, spec_path=None):
    """A trainer log written since this job started that carries the completion line. Written as
    a search rather than spec['log'] because a launcher can hand the same job a different log
    name than the spec claims (spec said exp012b_supervised.log, the run wrote exp013_ctx1024.log).

    The floor matters: with no floor the search would happily return the completion line of a
    PREVIOUS job and declare the current one finished before it ran a step. So a candidate must
    be newer than the last launch, or (first run, nothing launched yet) newer than the spec."""
    stamps = [st.get("last_launch"), st.get("last_relaunch")]
    floor = None
    for v in stamps:
        if not v:
            continue
        try:
            ts = datetime.fromisoformat(v).timestamp()
            floor = ts if floor is None else max(floor, ts)
        except Exception:
            pass
    if floor is None and spec_path and os.path.exists(spec_path):
        floor = os.path.getmtime(spec_path)
    for p in glob.glob(os.path.join(LAB, "logs", "*.log")):
        try:
            if floor is not None and os.path.getmtime(p) < floor:
                continue
            with open(p, "rb") as f:
                f.seek(max(0, os.path.getsize(p) - 8192))
                if "done. final" in f.read().decode("utf-8", "replace"):
                    return p
        except Exception:
            continue
    return None


def run_finished(spec, st, spec_path=None):
    """True when the RUN ITSELF ended deliberately: its early-stop guard fired, it reached
    target_steps, or the trainer printed its completion line. This is a legitimate end state,
    NOT a crash -- treating it as 'dead' makes the watchdog relaunch a finished run, which then
    aborts again (observed live 2026-09-18: the degrade guard fired at step 8200/20000)."""
    rd = find_run_dir(spec)
    if rd:
        lj = os.path.join(rd, "loss.jsonl")
        if os.path.exists(lj):
            try:
                with open(lj, "rb") as f:
                    f.seek(max(0, os.path.getsize(lj) - 65536))
                    tail = f.read().decode("utf-8", "replace")
                if '"early_stop": true' in tail:
                    return True, "early-stop guard fired"
            except Exception:
                pass
        step = curve_step(lj) if os.path.exists(lj) else None
        if step is not None and step >= spec["target_steps"]:
            return True, f"reached target_steps ({spec['target_steps']})"
    log = finished_log(spec, st, spec_path)
    if log:
        return True, f"trainer printed 'done.' in {os.path.basename(log)}"
    return False, ""


def progress_step(spec):
    """Newest step the run has actually reached: loss.jsonl first, ckpt step second."""
    rd = find_run_dir(spec)
    last = None
    if rd:
        lj = os.path.join(rd, "loss.jsonl")
        if os.path.exists(lj):
            last = curve_step(lj)
    step_ck = newest_ckpt_step(spec)
    return max([x for x in (last, step_ck) if x is not None], default=None)


def ckpt_patterns(spec):
    """Glob patterns for THIS job's step checkpoints, using the trainer's real naming convention
    (exp002_<preset>_<tokenizer-stem>_<stamp>_step<N>.pt). The run-dir family is NOT used: ckpt
    names carry the preset+tokenizer stem and omit the run dir's corpus suffix, so family
    matching finds nothing (exp002_pythia_rope_bpe_owt16k_202609191655_step500.pt has no
    'pythia_rope_bpe_owt16k_openwebtext_combined' anywhere in it)."""
    pats = []
    if spec.get("stamp"):
        pats.append(f"*_{spec['stamp']}_step*.pt")
    args = spec.get("train_args") or []

    def val(flag):
        try:
            return args[args.index(flag) + 1]
        except Exception:
            return None

    preset, tok = val("--preset"), val("--tokenizer")
    if preset and tok:
        stem = os.path.splitext(os.path.basename(tok))[0]
        pats.append(f"*{preset}*{stem}*_step*.pt")
    return pats


def newest_ckpt_step(spec):
    """Newest step checkpoint belonging to this job, or None. Deliberately returns None rather
    than guessing: a step from the WRONG run would either fake completion or suppress a needed
    relaunch, and the loss curve is the authoritative progress signal anyway."""
    best = None
    for pat in ckpt_patterns(spec):
        for p in glob.glob(os.path.join(LAB, spec["ckpt_dir"], pat)):
            m = re.search(r"_step(\d+)\.pt$", p)
            if m:
                st = int(m.group(1))
                best = st if best is None else max(best, st)
    return best


def newest_step_ckpt(spec, spec_path=None):
    """Highest-step ckpt for a (re)launch to --resume from. STEP ckpts only: _best.pt is an
    earlier step by definition, so resuming from it throws away progress (train_launch.py's
    rule, kept).

    Cross-run bleed guard: the loose preset+tokenizer glob matches EVERY run of that pair --
    a chained probe would resume its predecessor's weights and measure the wrong LR -- so
    candidates must be newer than the active spec file. Arming and chain-swapping both
    rewrite train_job.json, so its mtime marks this job's era: its own ckpts are newer,
    earlier runs' ckpts are not."""
    floor = None
    if spec_path:
        try:
            floor = os.path.getmtime(spec_path)
        except OSError:
            pass
    best = None
    for pat in ckpt_patterns(spec):
        for p in glob.glob(os.path.join(LAB, spec["ckpt_dir"], pat)):
            m = re.search(r"_step(\d+)\.pt$", p)
            if not m:
                continue
            if floor is not None and os.path.getmtime(p) < floor:
                continue
            st = int(m.group(1))
            if best is None or st > best[0]:
                best = (st, p)
    return best


def task_exists(task=TASK):
    r = subprocess.run(["schtasks", "/query", "/tn", task], capture_output=True, text=True)
    return r.returncode == 0


def main():
    """Single watchdog tick, serialized against overlapping ticks.

    Cron fires every 10 minutes and a slow tick can overlap the next one; two
    watchdogs judging the same dead job both relaunch it. The first tick holds
    the watchdog lock, the second exits 0 without doing anything.
    """
    if LAB not in sys.path:
        sys.path.insert(0, LAB)
    from train.runtime_lock import acquire_watchdog_lock
    try:
        lock = acquire_watchdog_lock(WATCHDOG_LOCK_PATH)
    except Exception:
        return 0
    try:
        return _main_locked()
    finally:
        lock.release()


def _main_locked():
    dry = "--dry-run" in sys.argv
    quiet = "--quiet" in sys.argv        # cron passes this: healthy runs print nothing
    spec_path = SPEC
    if "--spec" in sys.argv:
        spec_path = sys.argv[sys.argv.index("--spec") + 1]
    spec = load_json(spec_path)
    STATE = state_path_for(spec_path)
    st = load_json(STATE, {})
    now = datetime.now()

    # The state file is shared by every job that ever lives in this spec path, so it must not
    # leak one job's verdict into the next. Anything recorded for a DIFFERENT job name is
    # stale by definition: drop the restart ledger with it (otherwise a fresh run inherits
    # 'exhausted' and the watchdog refuses to supervise it at all).
    if st.get("job") not in (None, spec.get("name")):
        st = {}
    st["job"] = spec.get("name")
    st["run_name"] = spec.get("run_name")

    proc, can_tell = running_process(spec)
    proc_pid = (proc.info["pid"] if hasattr(proc, "info") else proc) if proc else None
    if not can_tell:
        # Blind probe (no psutil under THIS interpreter). Do NOT relaunch: 2026-09-23,
        # cron's python lacked psutil, this returned None, and the tick double-launched
        # a healthy 160M run onto a 16GB GPU. Blind != dead. Alert loudly instead.
        print(f"[watchdog] ERROR: cannot probe processes (psutil missing under "
              f"{sys.executable}) — refusing to judge job '{spec.get('name')}'. "
              f"Install psutil for this interpreter or repoint cron. No-op this tick.")
        return 1
    if proc is not None:
        step_now = progress_step(spec)
        note = stuck_note(spec, st, step_now, now)
        if not dry:
            st.update({"status": "running", "pid": proc_pid,
                       "checked_at": now.isoformat(timespec="seconds")})
            if st.get("watchdog_last_step") != step_now:
                st["watchdog_last_step"] = step_now
                st["watchdog_last_at"] = now.isoformat(timespec="seconds")
            save_state(st, STATE)
        if not quiet:
            print(f"[watchdog] running: pid={proc_pid} (job={spec['name']}) "
                  f"step={step_now}")
        if note:
            print(note)
        return 0

    # A latched verdict means we already said the true thing about this run. Re-deriving it
    # every 10 minutes is how one finished run produced 121 failed cron ticks and 121
    # undeliverable alerts; a finished run is not a recurring failure.
    if st.get("status") in TERMINAL or st.get("status") == "quarantined":
        if not spec.get("completed") and not dry:
            spec["completed"] = True
            with open(spec_path, "w", encoding="utf-8") as f:
                json.dump(spec, f, indent=2)
        if not quiet:
            print(f"[watchdog] {spec['name']}: {st['status']} (final, latched - nothing to do)")
        return 0

    step = progress_step(spec)
    target = spec["target_steps"]

    # A run that ended BY ITS OWN RULE (early-stop / degrade guard) is finished, not dead.
    # Without this the watchdog relaunches it, the guard fires again, and it burns the
    # restart budget on a completed run -- observed live 2026-09-18, step 8200/20000.
    finished, why = run_finished(spec, st, spec_path)
    if finished:
        first_time = not spec.get("completed")
        if not dry and first_time:
            st.update({"status": "stopped-by-rule", "final_step": step, "reason": why,
                       "checked_at": now.isoformat(timespec="seconds")})
            save_state(st, STATE)
            spec["completed"] = True
            with open(spec_path, "w", encoding="utf-8") as f:
                json.dump(spec, f, indent=2)
        if first_time and not quiet:
            print(f"[watchdog] {spec['name']} ENDED BY ITS OWN RULE at step {step}/{target} "
                  f"({why}) - no relaunch")
        # ---- chain: swap in next_job spec and launch (even on early-stop) -----
        if spec.get("completed"):
            next_job = spec.get("next_job")
            if next_job and not dry:
                next_path = os.path.join(LAB, next_job)
                if os.path.exists(next_path):
                    with open(next_path, encoding="utf-8") as f:
                        nspec = json.load(f)
                    if not nspec.get("completed"):
                        print(f"[watchdog] CHAIN: swapping in {nspec['name']} from {next_job}")
                        with open(spec_path, "w", encoding="utf-8") as f:
                            json.dump(nspec, f, indent=2)
                        launch_direct(nspec, spec_path)
                        return 0
        return 0

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
        # ---- chain: swap in next_job spec and launch --------------------------
        next_job = spec.get("next_job")
        if next_job and not dry:
            next_path = os.path.join(LAB, next_job)
            if os.path.exists(next_path):
                with open(next_path, encoding="utf-8") as f:
                    nspec = json.load(f)
                if not nspec.get("completed"):
                    print(f"[watchdog] CHAIN: swapping in {nspec['name']} from {next_job}")
                    # Overwrite train_job.json with the next spec
                    with open(spec_path, "w", encoding="utf-8") as f:
                        json.dump(nspec, f, indent=2)
                    launch_direct(nspec, spec_path)
                    return 0
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
        print(f"[watchdog] launching directly (no scheduled task needed)")
    if not launch_direct(spec, spec_path):
        print(f"[watchdog] direct launch failed - cannot relaunch")
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
