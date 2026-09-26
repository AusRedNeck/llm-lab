import importlib.util
import os
import sys
from pathlib import Path

import pytest

LAB = Path(__file__).resolve().parents[1]


def load_watchdog():
    path = LAB / "scripts" / "train_watchdog.py"
    spec = importlib.util.spec_from_file_location("watchdog_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeProcess:
    def __init__(self, pid, name, cmdline):
        self.pid = pid
        self.name = name
        self.cmdline = cmdline
        self.info = {"pid": pid, "name": name, "cmdline": cmdline}


def test_global_probe_finds_trainer_but_not_mentions(monkeypatch):
    wd = load_watchdog()
    fake = [
        FakeProcess(10, "python.exe", [os.sys.executable, "-u", "-m", "train.train"]),
        FakeProcess(11, "python.exe", [os.sys.executable, "-c", "mentions train.train"]),
        FakeProcess(12, "bash.exe", [os.sys.executable, "-m", "train.train"]),
    ]
    monkeypatch.setattr(wd, "_iter_psutil_processes", lambda: iter(fake))
    found, can_tell = wd.running_trainers()
    assert can_tell is True
    assert [p.pid for p in found] == [10]


def test_global_probe_is_independent_of_job_match(monkeypatch):
    wd = load_watchdog()
    fake = [FakeProcess(20, "python.exe", [os.sys.executable, "-m", "train.train", "--lr", "1e-3"])]
    monkeypatch.setattr(wd, "_iter_psutil_processes", lambda: iter(fake))
    found, can_tell = wd.running_trainers()
    assert can_tell is True
    assert len(found) == 1


def test_direct_launch_refuses_when_any_trainer_is_alive(monkeypatch):
    wd = load_watchdog()
    fake = [FakeProcess(30, "python.exe", [os.sys.executable, "-m", "train.train"])]
    monkeypatch.setattr(wd, "running_trainers", lambda: (fake, True))
    monkeypatch.setattr(wd.subprocess, "Popen", lambda *a, **k: pytest.fail("must not Popen"))
    spec = {"python": sys.executable, "train_args": ["--preset", "t1m", "--steps", "1"]}
    assert wd.launch_direct(spec) is False


def test_quarantined_state_never_relaunches(tmp_path, monkeypatch):
    wd = load_watchdog()
    spec_path = tmp_path / "train_job.json"
    spec_path.write_text(
        '{"name": "collision", "completed": false}', encoding="utf-8"
    )
    state_path = tmp_path / "train_job.state.json"
    state_path.write_text(
        '{"job": "collision", "status": "quarantined"}', encoding="utf-8"
    )
    monkeypatch.setattr(wd, "SPEC", str(spec_path))
    monkeypatch.setattr(wd, "running_process", lambda spec: (None, True))
    monkeypatch.setattr(
        wd, "launch_direct", lambda *a, **k: pytest.fail("must not launch")
    )
    monkeypatch.setattr(sys, "argv", ["train_watchdog.py"])

    assert wd.main() == 0
    assert '"completed": true' in spec_path.read_text(encoding="utf-8")


def test_probe_tolerates_psutil_name_method(monkeypatch):
    wd = load_watchdog()

    class NameMethodProcess:
        pid = 40
        info = {"pid": 40, "cmdline": [os.sys.executable, "-m", "train.train"]}

        def name(self):
            return "python.exe"

    monkeypatch.setattr(
        wd, "_iter_psutil_processes", lambda: iter([NameMethodProcess()])
    )
    found, can_tell = wd.running_trainers()
    assert can_tell is True
    assert [p.pid for p in found] == [40]


def test_probe_fails_closed_when_cached_cmdline_is_unreadable(monkeypatch):
    wd = load_watchdog()

    class UnreadableProcess:
        pid = 50
        info = {"pid": 50, "name": "python.exe", "cmdline": None}

        def name(self):
            return "python.exe"

        def cmdline(self):
            raise (RuntimeError("process exited during probe"))

    monkeypatch.setattr(
        wd, "_iter_psutil_processes", lambda: iter([UnreadableProcess()])
    )
    found, can_tell = wd.running_trainers()
    assert found == []
    assert can_tell is False


def test_second_watchdog_instance_is_a_quiet_noop(tmp_path, monkeypatch):
    wd = load_watchdog()
    from train.runtime_lock import acquire_watchdog_lock

    lock_path = tmp_path / "watchdog.lock"
    monkeypatch.setattr(wd, "WATCHDOG_LOCK_PATH", lock_path)
    monkeypatch.setattr(
        wd, "_main_locked", lambda *a, **k: pytest.fail("second watchdog ran")
    )
    first = acquire_watchdog_lock(lock_path)
    try:
        assert wd.main() == 0
    finally:
        first.release()

