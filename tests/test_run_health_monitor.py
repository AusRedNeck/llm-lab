"""The run monitor must be silent when healthy and loud when the run is not.

Why this exists: the watchdog's silence is correct (it latches a rule-stopped run
as "final, latched - nothing to do"), which is how p160-pile-parity-full sat dead
8.5h on 2026-10-08 and another 22h on 2026-10-09 with nobody looking. The monitor
is the thing that speaks up.

Run: pytest tests/test_run_health_monitor.py -q
"""
import json
import os
import sys
import time

import pytest

LAB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(LAB, "scripts"))

import check_run_health as h  # noqa: E402


@pytest.fixture
def lab(tmp_path, monkeypatch):
    """Point the monitor at a throwaway lab tree."""
    curve = tmp_path / "loss.jsonl"
    spec = tmp_path / "train_job.json"
    state = tmp_path / "train_job.state.json"
    curve.write_text('{"step": 9700, "val_bpb": 1.35}\n', encoding="utf-8")
    spec.write_text(json.dumps({"target_steps": 24694}), encoding="utf-8")
    state.write_text(json.dumps({"status": "running", "relaunch_count": 7,
                                 "checked_at": "2999-01-01T00:00:00"}), encoding="utf-8")
    monkeypatch.setattr(h, "CURVE", str(curve))
    monkeypatch.setattr(h, "SPEC", str(spec))
    monkeypatch.setattr(h, "STATE", str(state))
    return {"curve": curve, "spec": spec, "state": state}


def set_state(path, **over):
    d = json.loads(path.read_text(encoding="utf-8"))
    d.update(over)
    path.write_text(json.dumps(d), encoding="utf-8")


def test_healthy_run_is_silent(lab, capsys):
    h.main()
    assert capsys.readouterr().out == "", "a healthy run must produce no output"


def test_rule_stopped_run_is_reported(lab, capsys):
    set_state(lab["state"], status="stopped-by-rule",
              reason="trainer printed 'done.' in p160-pile-parity-full.log")
    h.main()
    out = capsys.readouterr().out
    assert out.startswith("STOPPED:") and "9700/24694" in out and "done." in out


def test_stopped_run_wins_over_other_checks(lab, capsys):
    """A latched verdict is the story; a stale curve beside it is not news."""
    set_state(lab["state"], status="exhausted", checked_at="2000-01-01T00:00:00")
    h.main()
    out = capsys.readouterr().out
    assert out.startswith("STOPPED:") and "STALE" not in out


def test_running_but_curve_frozen_is_reported(lab, capsys):
    old = time.time() - 3600
    os.utime(lab["curve"], (old, old))
    h.main()
    assert capsys.readouterr().out.startswith("STALLED:")


def test_reaching_target_reports_complete(lab, capsys):
    lab["curve"].write_text('{"step": 24694, "val_bpb": 1.40}\n', encoding="utf-8")
    h.main()
    out = capsys.readouterr().out
    assert out.startswith("COMPLETE:") and "retire this monitor" in out


def test_complete_wins_over_terminal_status(lab, capsys):
    """A finished arm must not be reported as a stop."""
    lab["curve"].write_text('{"step": 24694, "val_bpb": 1.40}\n', encoding="utf-8")
    set_state(lab["state"], status="stopped-by-rule")
    h.main()
    out = capsys.readouterr().out
    assert out.startswith("COMPLETE:")
