"""Regression tests for the three cross-run contamination bugs in train_watchdog.py.

All three had the same failure mode: the watchdog reached a COMPLETE verdict for a
job that had never launched, because it adopted evidence belonging to a DIFFERENT,
already-dead run. They only bit when a spec was armed but not yet started
(run_name == ""), which is the normal state between `viz/arm.py` and first launch.

1. find_run_dir: an empty run family is not a wildcard. glob("*") matched every run
   dir in the lab, so the finish test read a dead run's step count.
2. finished_log: with no state and no spec_path there is no "newer than launch" floor,
   so the staleness guard disabled itself and matched a log weeks old.
3. newest_ckpt_step: all pythia runs share ckpt_dir and a pattern prefix, so a job
   targeting 5000 steps read _step8000.pt from a dead run. spec["stamp"] is "" until
   launch, so the stamp must be derived from run_name and used as a hard filter.

Observed live 2026-09-30 08:57 while arming pile-eff128k-5k, which reported
"COMPLETE: reached step 8000/5000 - no relaunch" three times in a row.
"""

import os
import sys

import pytest

LAB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(LAB, "scripts"))

import train_watchdog as w  # noqa: E402


def spec_with(**over):
    """A spec shaped like the real train_job.json, including stamp="".

    stamp is deliberately empty: that is its value when a spec is armed but has not
    yet launched, which is exactly the state these bugs exploited.

    train_args carries the real --preset/--tokenizer because ckpt_patterns() builds
    its glob from those two flags. An empty train_args yields NO patterns, which made
    every ckpt lookup return None and the scoping assertions below pass vacuously.
    """
    base = {
        "name": "test_job",
        "run_name": "",
        "stamp": "",
        "ckpt_dir": "checkpoints",
        "target_steps": 5000,
        "train_args": [
            "--preset", "pythia",
            "--tokenizer", "data/incoming/pythia70m_hf/tokenizer.json",
        ],
    }
    base.update(over)
    return base


# ---------------------------------------------------------------- bug 1: family

def test_empty_family_is_not_a_wildcard():
    """An armed-but-unlaunched spec must report no run dir, not the lab's newest."""
    assert w.find_run_dir(spec_with()) is None


def test_pinned_run_name_resolves_to_that_exact_dir():
    rd = w.find_run_dir(
        spec_with(run_name="20260929_2159_pythia_tokenizer_pile_train_full")
    )
    assert rd is not None
    assert rd.endswith("20260929_2159_pythia_tokenizer_pile_train_full")


# ------------------------------------------------------------------ bug 2: log

def test_no_floor_means_no_stale_log_match():
    """No launch floor => cannot prove any log belongs to this job => None."""
    assert w.finished_log(spec_with(), {}, None) is None


def test_log_newer_than_floor_is_still_found(tmp_path):
    """The guard must not be so strict it stops detecting real completion."""
    logs = os.path.join(LAB, "logs")
    if not os.path.isdir(logs):
        pytest.skip("no logs dir in this checkout")
    probe = os.path.join(logs, "_test_floor_probe.log")
    with open(probe, "w", encoding="utf-8") as f:
        f.write("done. final avg50 loss=1.0\n")
    try:
        import datetime as _dt
        floor = _dt.datetime.fromtimestamp(
            os.path.getmtime(probe) - 60).isoformat()
        got = w.finished_log(spec_with(), {"last_launch": floor}, None)
        assert got is not None and "_test_floor_probe.log" in got
        # ...and a log OLDER than the floor is still rejected.
        future = _dt.datetime.fromtimestamp(
            os.path.getmtime(probe) + 60).isoformat()
        assert w.finished_log(spec_with(), {"last_launch": future}, None) != probe
    finally:
        os.remove(probe)


# ----------------------------------------------------------------- bug 3: ckpt

@pytest.mark.parametrize("run_name,expected", [
    ("20260929_2159_pythia_tokenizer_pile_train_full", 8000),
    ("20260929_2133_pythia_tokenizer_pile_train_full", 6000),
    ("20260929_2250_pythia_tokenizer_pile_train_full", 200),
])
def test_ckpt_lookup_is_scoped_to_this_jobs_stamp(run_name, expected):
    if not os.path.isdir(os.path.join(LAB, "checkpoints")):
        pytest.skip("no checkpoints in this checkout")
    got = w.newest_ckpt_step(spec_with(run_name=run_name))
    assert got == expected, (
        f"{run_name}: expected its own newest ckpt {expected}, got {got} - "
        "cross-run contamination is back"
    )


def test_armed_job_reports_no_ckpt_progress():
    """The bug's exact symptom: an armed 5000-step job reading 8000."""
    assert w.newest_ckpt_step(spec_with()) is None


@pytest.mark.parametrize("run_name", [
    "20991231_9999_future_run",
    "no_digits_here",
    "",
])
def test_unstampable_run_names_yield_none_not_a_guess(run_name):
    assert w.newest_ckpt_step(spec_with(run_name=run_name)) is None


# ------------------------------------------------------------- end to end

def test_unlaunched_job_is_never_reported_finished(tmp_path):
    """The composite guard: no dir, no ckpt, no floor => not finished."""
    specp = str(tmp_path / "train_job.json")
    spec = spec_with(target_steps=5000)
    import json as _json
    with open(specp, "w", encoding="utf-8") as f:
        _json.dump(spec, f)
    assert w.run_finished(spec, {}, specp)[0] is False


def test_progress_step_is_none_not_a_foreign_step():
    assert w.progress_step(spec_with()) is None

# --------------------------------- stale completion evidence after a resume
# (the same shape of bug, opposite direction: evidence that WAS terminal and
#  stopped being terminal the moment the operator deliberately continued)
#
# Observed 2026-10-08 on p160-pile-parity-full: stopped at step 8900 by the
# saturation guard's false positive, resumed at 8500 after the guard was fixed.
# The stale "early_stop": true row and the stale "done. final" log line were both
# still on disk, so run_finished() would have re-latched the job on the first tick
# after its own relaunch -- and every later crash would have been reported as
# "final, latched - nothing to do". A marker is terminal only while it is the
# last thing that happened.


def test_completion_line_stops_being_terminal_once_training_resumes():
    import datetime as _dt
    logs = os.path.join(LAB, "logs")
    if not os.path.isdir(logs):
        pytest.skip("no logs dir in this checkout")
    probe = os.path.join(logs, "_test_stale_done.log")
    with open(probe, "w", encoding="utf-8") as f:
        f.write("done. final avg50 loss=1.0\n")
    try:
        floor = _dt.datetime.fromtimestamp(os.path.getmtime(probe) - 60).isoformat()
        st = {"last_launch": floor}
        assert w.finished_log(spec_with(), st, None) == probe, \
            "a log whose last line is completion must still count as finished"
        # A relaunch appends a banner and step lines after it.
        with open(probe, "a", encoding="utf-8") as f:
            f.write("=== launch 2026-10-08 21:00:00 ===\nstep 8501/24694 loss=2.7\n")
        assert w.finished_log(spec_with(), st, None) != probe, \
            "completion is no longer the last thing in this log"
    finally:
        os.remove(probe)


def test_stale_early_stop_marker_does_not_veto_a_resume(tmp_path, monkeypatch):
    """Rows appended after the marker mean the run was continued on purpose."""
    lj = tmp_path / "loss.jsonl"
    lj.write_text(
        '{"step": 8800, "val_bpb": 1.49}\n'
        '{"early_stop": true, "step": 8900, "best_bpb": 1.348}\n'
        '{"step": 8901, "val_bpb": 1.50}\n',
        encoding="utf-8")
    monkeypatch.setattr(w, "find_run_dir", lambda spec: str(tmp_path))
    # a launch floor far in the future keeps any real log out of the verdict
    st = {"last_launch": "2100-01-01T00:00:00"}
    assert w.run_finished(spec_with(target_steps=24694), st, None)[0] is False


def test_current_early_stop_marker_is_still_terminal(tmp_path, monkeypatch):
    """The guard firing for real must still end the run -- that is its job."""
    lj = tmp_path / "loss.jsonl"
    lj.write_text(
        '{"step": 8800, "val_bpb": 1.49}\n'
        '{"early_stop": true, "step": 8900, "best_bpb": 1.348}\n',
        encoding="utf-8")
    monkeypatch.setattr(w, "find_run_dir", lambda spec: str(tmp_path))
    st = {"last_launch": "2100-01-01T00:00:00"}
    finished, why = w.run_finished(spec_with(target_steps=24694), st, None)
    assert finished, "a marker with nothing after it must still be terminal"
    assert "early-stop" in why

def test_operator_resume_override_suspends_the_verdict(tmp_path, monkeypatch):
    """The marker that stopped a run cannot veto the relaunch proving it stale.

    On the FIRST resume after a stop-by-rule there are no rows after the marker yet
    (the resumed trainer is what appends them), so without an explicit override the
    verdict is unfalsifiable: "finished" blocks the very relaunch that would falsify
    it. state["resume_requested"] is that override, and main() drops it once a live
    trainer is observed so it cannot outlive the resume it unblocked.
    """
    lj = tmp_path / "loss.jsonl"
    lj.write_text(
        '{"step": 8899, "val_bpb": 1.56}\n'
        '{"step": 8900, "val_bpb": 1.58}\n'
        '{"early_stop": true, "step": 8900, "best_bpb": 1.348}\n',
        encoding="utf-8")
    monkeypatch.setattr(w, "find_run_dir", lambda spec: str(tmp_path))
    st = {"last_launch": "2100-01-01T00:00:00"}
    spec = spec_with(target_steps=24694)
    assert w.run_finished(spec, st, None)[0] is True, "marker with nothing after it is terminal"
    st["resume_requested"] = True
    finished, why = w.run_finished(spec, st, None)
    assert finished is False, "an operator resume must suspend the verdict"
    assert "resume" in why
