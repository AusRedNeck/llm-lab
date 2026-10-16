import json
import os
import sys

import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import train_watchdog as w


def make_spec(path, target, horizon):
    spec = {"name": "chunk-test", "target_steps": target,
            "schedule_steps": horizon, "completed": False,
            "next_job": None, "run_name": "20261011_1000_t1m_synthetic",
            "stamp": "202610111000", "ckpt_dir": "checkpoints", "train_args": []}
    path.write_text(json.dumps(spec), encoding="utf-8")
    return spec


def test_intermediate_boundary_pauses_and_waits_for_explicit_go(tmp_path, monkeypatch, capsys):
    spec_path = tmp_path / "train_job.json"
    make_spec(spec_path, 4_000, 16_000)
    monkeypatch.setattr(w, "SPEC", str(spec_path))
    monkeypatch.setattr(w, "LAB", str(tmp_path))
    monkeypatch.setattr(w, "running_process", lambda spec: (None, True))
    monkeypatch.setattr(w, "progress_step", lambda spec: spec["target_steps"])
    monkeypatch.setattr(w, "run_finished", lambda *a: (False, ""))
    monkeypatch.setattr(w, "newest_step_ckpt", lambda *a: (4_000, "step4000.pt"))
    monkeypatch.setattr(w, "chunk_marker_complete", lambda *a: True)
    monkeypatch.setattr(w, "launch_direct", lambda *a: pytest.fail("must remain paused"))
    monkeypatch.setattr(sys, "argv", ["train_watchdog.py", "--quiet"])
    assert w._main_locked() == 0
    assert "PAUSED AT CHUNK" in capsys.readouterr().out
    state = json.loads((tmp_path / "train_job.state.json").read_text())
    spec = json.loads(spec_path.read_text())
    assert state["status"] == "paused_at_chunk"
    assert spec["completed"] is False
    assert w._main_locked() == 0
    assert capsys.readouterr().out == ""
    assert json.loads((tmp_path / "train_job.state.json").read_text())["status"] == "paused_at_chunk"


def test_global_horizon_is_final_not_paused(tmp_path, monkeypatch):
    spec_path = tmp_path / "train_job.json"
    make_spec(spec_path, 16_000, 16_000)
    monkeypatch.setattr(w, "SPEC", str(spec_path))
    monkeypatch.setattr(w, "LAB", str(tmp_path))
    monkeypatch.setattr(w, "running_process", lambda spec: (None, True))
    monkeypatch.setattr(w, "progress_step", lambda spec: spec["target_steps"])
    monkeypatch.setattr(w, "run_finished", lambda *a: (False, ""))
    monkeypatch.setattr(sys, "argv", ["train_watchdog.py"])
    assert w._main_locked() == 0
    assert json.loads((tmp_path / "train_job.state.json").read_text())["status"] == "complete"
    assert json.loads(spec_path.read_text())["completed"] is True


def test_intermediate_target_is_not_a_final_run(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "loss.jsonl").write_text('{"step": 4000}' + chr(10), encoding="utf-8")
    spec = make_spec(tmp_path / "unused.json", 4_000, 16_000)
    monkeypatch.setattr(w, "find_run_dir", lambda _spec: str(run_dir))
    monkeypatch.setattr(w, "finished_log", lambda *a: None)
    assert w.run_finished(spec, {}, str(tmp_path / "train_job.json"))[0] is False


def test_first_live_tick_pins_run_identity_and_registry(tmp_path, monkeypatch):
    spec_path = tmp_path / "train_job.json"
    spec = make_spec(spec_path, 4_000, 16_000)
    spec.update({"run_name": "", "stamp": "", "log": "logs/chunk01.log"})
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    floor = spec_path.stat().st_mtime - 60
    os.utime(spec_path, (floor, floor))
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    run_name = "20261011_1000_pythia160_2k_tokenizer_pile_train_full"
    (log_dir / "chunk01.log").write_text(
        f"  run dir: runs\\\\{run_name}" + chr(10), encoding="utf-8")
    registry = tmp_path / "experiments.json"
    registry.write_text(json.dumps({"arms": [{"id": "chunk-test", "run_dirs": []}]}), encoding="utf-8")
    monkeypatch.setattr(w, "LAB", str(tmp_path))
    assert w.pin_run_identity(spec, str(spec_path)) is True
    assert spec["run_name"] == run_name and spec["stamp"] == "202610111000"
    assert abs(spec_path.stat().st_mtime - floor) < 1
    assert json.loads(registry.read_text())["arms"][0]["run_dirs"] == [run_name]


def test_active_watchdog_tick_pins_before_reading_progress(tmp_path, monkeypatch):
    spec_path = tmp_path / "train_job.json"
    spec = make_spec(spec_path, 4_000, 16_000)
    spec.update({"run_name": "", "stamp": "", "log": "logs/chunk01.log"})
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    (tmp_path / "logs").mkdir()
    run_name = "20261011_1000_pythia160_2k_tokenizer_pile_train_full"
    (tmp_path / "logs/chunk01.log").write_text(f"run dir: runs/{run_name}" + chr(10))
    (tmp_path / "experiments.json").write_text(
        json.dumps({"arms": [{"id": "chunk-test", "run_dirs": []}]}))
    monkeypatch.setattr(w, "SPEC", str(spec_path))
    monkeypatch.setattr(w, "LAB", str(tmp_path))
    monkeypatch.setattr(w, "running_process", lambda _spec: (123, True))
    seen = []
    monkeypatch.setattr(w, "progress_step", lambda s: seen.append(s["run_name"]) or 12)
    monkeypatch.setattr(w, "stuck_note", lambda *a: None)
    monkeypatch.setattr(sys, "argv", ["train_watchdog.py"])
    assert w._main_locked() == 0
    assert seen == [run_name]
    state = json.loads((tmp_path / "train_job.state.json").read_text())
    assert state["run_name"] == run_name


def test_final_done_marker_is_complete_not_rule_stopped(tmp_path, monkeypatch):
    spec_path = tmp_path / "train_job.json"
    make_spec(spec_path, 16_000, 16_000)
    monkeypatch.setattr(w, "SPEC", str(spec_path))
    monkeypatch.setattr(w, "LAB", str(tmp_path))
    monkeypatch.setattr(w, "running_process", lambda _spec: (None, True))
    monkeypatch.setattr(w, "progress_step", lambda _spec: 16_000)
    monkeypatch.setattr(w, "run_finished", lambda *a: (True, "trainer printed 'done.' in chunk04.log"))
    monkeypatch.setattr(sys, "argv", ["train_watchdog.py"])
    assert w._main_locked() == 0
    state = json.loads((tmp_path / "train_job.state.json").read_text())
    assert state["status"] == "complete" and state["final_step"] == 16_000


def test_boundary_checkpoint_without_chunk_marker_pauses_with_warning(tmp_path, monkeypatch, capsys):
    spec_path = tmp_path / "train_job.json"
    make_spec(spec_path, 4_000, 16_000)
    monkeypatch.setattr(w, "SPEC", str(spec_path))
    monkeypatch.setattr(w, "LAB", str(tmp_path))
    monkeypatch.setattr(w, "running_process", lambda _spec: (None, True))
    monkeypatch.setattr(w, "progress_step", lambda _spec: 4_000)
    monkeypatch.setattr(w, "run_finished", lambda *a: (False, ""))
    monkeypatch.setattr(w, "newest_step_ckpt", lambda *a: (4_000, "step4000.pt"))
    monkeypatch.setattr(w, "chunk_marker_complete", lambda *a: False, raising=False)
    monkeypatch.setattr(w, "launch_direct", lambda *a: pytest.fail("must not resume at the target step"))
    monkeypatch.setattr(sys, "argv", ["train_watchdog.py", "--quiet"])
    assert w._main_locked() == 0
    state = json.loads((tmp_path / "train_job.state.json").read_text())
    assert state["status"] == "paused_at_chunk"
    assert state["pause_warning"] == "chunk marker missing"
    assert "WARNING" in capsys.readouterr().out


def test_chunk_marker_must_be_last_row_for_this_run(tmp_path, monkeypatch):
    spec_path = tmp_path / "unused.json"
    spec = make_spec(spec_path, 4_000, 16_000)
    run_dir = tmp_path / "runs" / spec["run_name"]
    run_dir.mkdir(parents=True)
    curve = run_dir / "loss.jsonl"
    marker = {"chunk_complete": True, "step": 4_000, "schedule_steps": 16_000}
    curve.write_text(json.dumps(marker) + chr(10), encoding="utf-8")
    monkeypatch.setattr(w, "LAB", str(tmp_path))
    assert w.chunk_marker_complete(spec, 4_000) is True
    with curve.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"step": 3_999, "train": 1.0}) + chr(10))
    assert w.chunk_marker_complete(spec, 4_000) is False
