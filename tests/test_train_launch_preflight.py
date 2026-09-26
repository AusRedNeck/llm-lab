import importlib.util
import sys
from pathlib import Path

import pytest

LAB = Path(__file__).resolve().parents[1]


def load_launcher():
    path = LAB / "scripts" / "train_launch.py"
    spec = importlib.util.spec_from_file_location("train_launch_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_launcher_resolves_repo_root_not_scripts_directory():
    launcher = load_launcher()
    assert Path(launcher.LAB) == LAB


def test_launcher_refuses_when_any_trainer_is_alive(tmp_path, monkeypatch):
    launcher = load_launcher()
    state_path = tmp_path / "train_job.state.json"
    state_path.write_text('{"job": "job", "status": "running"}', encoding="utf-8")
    monkeypatch.setattr(
        launcher, "running_trainers", lambda: ([object()], True)
    )
    assert launcher.preflight_reason(
        {"name": "job", "completed": False}, state_path
    ) == "trainer already alive"


def test_launcher_refuses_blind_process_probe(tmp_path, monkeypatch):
    launcher = load_launcher()
    state_path = tmp_path / "train_job.state.json"
    state_path.write_text('{"job": "job", "status": "running"}', encoding="utf-8")
    monkeypatch.setattr(launcher, "running_trainers", lambda: ([], False))
    assert launcher.preflight_reason(
        {"name": "job", "completed": False}, state_path
    ) == "cannot inspect trainers"


def test_launcher_refuses_quarantined_state(tmp_path, monkeypatch):
    launcher = load_launcher()
    state_path = tmp_path / "train_job.state.json"
    state_path.write_text(
        '{"job": "job", "status": "quarantined"}', encoding="utf-8"
    )
    monkeypatch.setattr(
        launcher, "running_trainers", lambda: pytest.fail("probe must not run")
    )
    assert launcher.preflight_reason(
        {"name": "job", "completed": False}, state_path
    ) == "job quarantined"
