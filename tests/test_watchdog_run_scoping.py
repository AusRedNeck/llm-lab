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

Every test here is SELF-CONTAINED: it builds its own runs/, checkpoints/ and logs/
under tmp_path and points the watchdog's LAB there. No test reads a historical
runtime artifact, so the suite passes on a fresh checkout with an empty (or absent)
lab. The checkpoints/ dir is deliberately created EMPTY in the family fixture: the
real-world trap was a ckpt_dir that exists but holds nothing for this job, which the
old tests only handled by skipping when the whole directory was missing.
"""

import datetime
import json
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


@pytest.fixture
def lab(tmp_path, monkeypatch):
    """A private lab root with the three dirs the watchdog reads, pointed at by LAB.

    checkpoints/ exists but starts EMPTY -- the exact condition the live bug hit
    (a ckpt_dir present with no files for this job), which an isdir-skip test never
    exercises.
    """
    for d in ("runs", "checkpoints", "logs"):
        (tmp_path / d).mkdir()
    monkeypatch.setattr(w, "LAB", str(tmp_path))
    return tmp_path


def _run_dir(lab, run_name, step=None):
    d = lab / "runs" / run_name
    d.mkdir(parents=True, exist_ok=True)
    if step is not None:
        (d / "loss.jsonl").write_text(
            json.dumps({"step": step}) + "\n", encoding="utf-8")
    return d


def _ckpt(lab, stamp, step):
    """A step checkpoint named the trainer's way, in this job's ckpt_dir."""
    p = lab / "checkpoints" / f"exp002_pythia_tokenizer_{stamp}_step{step}.pt"
    p.write_bytes(b"")
    return p


# ---------------------------------------------------------------- bug 1: family

def test_empty_family_is_not_a_wildcard(lab):
    """An armed-but-unlaunched spec must report no run dir, not the lab's newest.

    The lab holds a foreign run with a huge step count; glob("*") on an empty
    family used to adopt it and declare the new job complete before it launched.
    """
    _run_dir(lab, "20991231_0000_something_else", step=999999)
    assert w.find_run_dir(spec_with()) is None


def test_pinned_run_name_resolves_to_that_exact_dir(lab):
    """A pinned run_name must resolve to that exact dir, not a family sibling."""
    exact = "20260929_2159_pythia_tokenizer_pile_train_full"
    _run_dir(lab, exact, step=8000)
    _run_dir(lab, "20260929_2133_pythia_tokenizer_pile_train_full", step=9999)
    rd = w.find_run_dir(spec_with(run_name=exact))
    assert rd == str(lab / "runs" / exact)


def test_missing_pinned_dir_falls_back_to_newest_family_member(lab):
    """An exact run_name with no curve yet must still find its family's newest run.

    The trainer self-stamps its dir, so the spec's run_name can lag the real one;
    family fallback is what makes that drift harmless.
    """
    _run_dir(lab, "20260929_2133_pythia_tokenizer_pile_train_full", step=6000)
    newest = _run_dir(lab, "20260929_2159_pythia_tokenizer_pile_train_full", step=8000)
    rd = w.find_run_dir(
        spec_with(run_name="20260929_2999_pythia_tokenizer_pile_train_full"))
    assert rd == str(newest)


# ------------------------------------------------------------------ bug 2: log

def test_no_floor_means_no_stale_log_match(lab):
    """No launch floor => cannot prove any log belongs to this job => None.

    A stale completion log is sitting right there; with no floor to exclude it the
    guard must still refuse to adopt it.
    """
    (lab / "logs" / "exp011b_train.log").write_text(
        "done. final avg50 loss=1.0\n", encoding="utf-8")
    assert w.finished_log(spec_with(), {}, None) is None


def test_log_newer_than_floor_is_still_found(lab):
    """The guard must not be so strict it stops detecting real completion."""
    probe = lab / "logs" / "_test_floor_probe.log"
    probe.write_text("done. final avg50 loss=1.0\n", encoding="utf-8")
    floor = datetime.datetime.fromtimestamp(
        os.path.getmtime(probe) - 60).isoformat()
    got = w.finished_log(spec_with(), {"last_launch": floor}, None)
    assert got == str(probe)
    # ...and a log OLDER than the floor is still rejected.
    future = datetime.datetime.fromtimestamp(
        os.path.getmtime(probe) + 60).isoformat()
    assert w.finished_log(spec_with(), {"last_launch": future}, None) is None


def test_spec_file_mtime_is_the_floor_when_nothing_launched(lab):
    """First run: no state, but the spec file's own mtime is a usable floor."""
    specp = lab / "train_job.json"
    specp.write_text(json.dumps(spec_with()), encoding="utf-8")
    old = lab / "logs" / "old.log"
    old.write_text("done. final avg50 loss=1.0\n", encoding="utf-8")
    # make old.log clearly older than the spec
    past = os.path.getmtime(specp) - 3600
    os.utime(old, (past, past))
    assert w.finished_log(spec_with(), {}, str(specp)) is None
    fresh = lab / "logs" / "fresh.log"
    fresh.write_text("done. final avg50 loss=1.0\n", encoding="utf-8")
    assert w.finished_log(spec_with(), {}, str(specp)) == str(fresh)


# ----------------------------------------------------------------- bug 3: ckpt

# Three real runs of the same family, each with its own ckpt stamp, plus a foreign
# run sharing the preset+tokenizer prefix. A correct lookup returns the job's OWN
# newest step and never another run's.
CKPT_CASES = [
    ("20260929_2159_pythia_tokenizer_pile_train_full", "202609292159", 8000),
    ("20260929_2133_pythia_tokenizer_pile_train_full", "202609292133", 6000),
    ("20260929_2250_pythia_tokenizer_pile_train_full", "202609292250", 200),
]


@pytest.fixture
def ckpt_lab(lab):
    """The ckpt_lab with all three runs' checkpoints, each stamped to its own run."""
    for run_name, stamp, newest in CKPT_CASES:
        for step in (200, newest):
            _ckpt(lab, stamp, step)
    # foreign run, same preset+tokenizer, must never leak in
    _ckpt(lab, "202610010000", 5000)
    return lab


@pytest.mark.parametrize("run_name,stamp,expected", CKPT_CASES)
def test_ckpt_lookup_is_scoped_to_this_jobs_stamp(ckpt_lab, run_name, stamp, expected):
    got = w.newest_ckpt_step(spec_with(run_name=run_name))
    assert got == expected, (
        f"{run_name}: expected its own newest ckpt {expected}, got {got} - "
        "cross-run contamination is back"
    )


def test_empty_ckpt_dir_is_not_a_wildcard(lab):
    """A ckpt_dir that EXISTS but holds nothing for this job must yield None.

    This is the live condition: the local checkpoints/ dir was present and empty of
    this run's files, and the old test skipped whenever the dir was missing -- so it
    could not catch a lookup that reached into a populated dir.
    """
    assert (lab / "checkpoints").is_dir()
    assert w.newest_ckpt_step(
        spec_with(run_name="20260929_2159_pythia_tokenizer_pile_train_full")) is None


def test_armed_job_reports_no_ckpt_progress(ckpt_lab):
    """The bug's exact symptom: an armed 5000-step job reading 8000."""
    assert w.newest_ckpt_step(spec_with()) is None


@pytest.mark.parametrize("run_name", [
    "20991231_9999_future_run",
    "no_digits_here",
    "",
])
def test_unstampable_run_names_yield_none_not_a_guess(ckpt_lab, run_name):
    assert w.newest_ckpt_step(spec_with(run_name=run_name)) is None


# ------------------------------------------------------------- end to end

def test_unlaunched_job_is_never_reported_finished(lab):
    """The composite guard: no dir, no ckpt, no floor => not finished."""
    specp = lab / "train_job.json"
    specp.write_text(json.dumps(spec_with(target_steps=5000)), encoding="utf-8")
    assert w.run_finished(spec_with(target_steps=5000), {}, str(specp))[0] is False


def test_progress_step_is_none_not_a_foreign_step(lab):
    _run_dir(lab, "20991231_0000_something_else", step=999999)
    _ckpt(lab, "209912310000", 999999)
    assert w.progress_step(spec_with()) is None


def test_progress_step_prefers_curve_then_ckpt_for_a_real_run(lab):
    """A real run's progress is its curve step, falling back to its own ckpt step."""
    run_name = "20260929_2159_pythia_tokenizer_pile_train_full"
    _run_dir(lab, run_name, step=8000)
    _ckpt(lab, "202609292159", 9000)
    assert w.progress_step(spec_with(run_name=run_name)) == 9000
    # and with no curve, the ckpt step is the honest fallback
    (lab / "runs" / run_name / "loss.jsonl").unlink()
    assert w.progress_step(spec_with(run_name=run_name)) == 9000


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


def test_completion_line_stops_being_terminal_once_training_resumes(lab):
    probe = lab / "logs" / "_test_stale_done.log"
    probe.write_text("done. final avg50 loss=1.0\n", encoding="utf-8")
    floor = datetime.datetime.fromtimestamp(
        os.path.getmtime(probe) - 60).isoformat()
    st = {"last_launch": floor}
    assert w.finished_log(spec_with(), st, None) == str(probe), \
        "a log whose last line is completion must still count as finished"
    # A relaunch appends a banner and step lines after it.
    with open(probe, "a", encoding="utf-8") as f:
        f.write("=== launch 2026-10-08 21:00:00 ===\nstep 8501/24694 loss=2.7\n")
    assert w.finished_log(spec_with(), st, None) != str(probe), \
        "completion is no longer the last thing in this log"


def test_stale_early_stop_marker_does_not_veto_a_resume(lab, monkeypatch):
    """Rows appended after the marker mean the run was continued on purpose."""
    rd = _run_dir(lab, "20260929_2159_pythia_tokenizer_pile_train_full")
    (rd / "loss.jsonl").write_text(
        '{"step": 8800, "val_bpb": 1.49}\n'
        '{"early_stop": true, "step": 8900, "best_bpb": 1.348}\n'
        '{"step": 8901, "val_bpb": 1.50}\n',
        encoding="utf-8")
    # a launch floor far in the future keeps any real log out of the verdict
    st = {"last_launch": "2100-01-01T00:00:00"}
    assert w.run_finished(
        spec_with(run_name="20260929_2159_pythia_tokenizer_pile_train_full",
                  target_steps=24694), st, None)[0] is False


def test_current_early_stop_marker_is_still_terminal(lab):
    """The guard firing for real must still end the run -- that is its job."""
    rd = _run_dir(lab, "20260929_2159_pythia_tokenizer_pile_train_full")
    (rd / "loss.jsonl").write_text(
        '{"step": 8800, "val_bpb": 1.49}\n'
        '{"early_stop": true, "step": 8900, "best_bpb": 1.348}\n',
        encoding="utf-8")
    st = {"last_launch": "2100-01-01T00:00:00"}
    finished, why = w.run_finished(
        spec_with(run_name="20260929_2159_pythia_tokenizer_pile_train_full",
                  target_steps=24694), st, None)
    assert finished, "a marker with nothing after it must still be terminal"
    assert "early-stop" in why


def test_operator_resume_override_suspends_the_verdict(lab):
    """The marker that stopped a run cannot veto the relaunch proving it stale.

    On the FIRST resume after a stop-by-rule there are no rows after the marker yet
    (the resumed trainer is what appends them), so without an explicit override the
    verdict is unfalsifiable: "finished" blocks the very relaunch that would falsify
    it. state["resume_requested"] is that override, and main() drops it once a live
    trainer is observed so it cannot outlive the resume it unblocked.
    """
    rd = _run_dir(lab, "20260929_2159_pythia_tokenizer_pile_train_full")
    (rd / "loss.jsonl").write_text(
        '{"step": 8899, "val_bpb": 1.56}\n'
        '{"step": 8900, "val_bpb": 1.58}\n'
        '{"early_stop": true, "step": 8900, "best_bpb": 1.348}\n',
        encoding="utf-8")
    st = {"last_launch": "2100-01-01T00:00:00"}
    spec = spec_with(run_name="20260929_2159_pythia_tokenizer_pile_train_full",
                     target_steps=24694)
    assert w.run_finished(spec, st, None)[0] is True, \
        "marker with nothing after it is terminal"
    st["resume_requested"] = True
    finished, why = w.run_finished(spec, st, None)
    assert finished is False, "an operator resume must suspend the verdict"
    assert "resume" in why
