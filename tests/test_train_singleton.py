import importlib
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

LAB = Path(__file__).resolve().parents[1]


def runtime_lock_module():
    assert importlib.util.find_spec("train.runtime_lock") is not None, (
        "train.runtime_lock must exist before CUDA initialization"
    )
    return importlib.import_module("train.runtime_lock")


def test_lock_rejects_second_owner_and_release_allows_next(tmp_path):
    mod = runtime_lock_module()
    path = tmp_path / "train.lock"

    first = mod.acquire_train_lock(path, owner={"pid": os.getpid(), "note": "first"})
    try:
        with pytest.raises(mod.TrainLockError) as exc:
            mod.acquire_train_lock(path, owner={"pid": 4242, "note": "second"})
        assert "train.train singleton lock" in str(exc.value)
        assert str(path) in str(exc.value)
    finally:
        first.release()

    second = mod.acquire_train_lock(path, owner={"pid": os.getpid(), "note": "next"})
    second.release()


def test_train_entrypoint_refuses_before_cuda_allocation(monkeypatch, tmp_path):
    mod = runtime_lock_module()
    # Use a TEST-ONLY lock path via the env override. These tests deliberately
    # contend for a real lock, so pointing them at the production
    # DEFAULT_LOCK_PATH made them FAIL whenever a run was actually in flight:
    # the live trainer holds that lock, the probe could never acquire it, and
    # the test could not tell its own rejection from the trainer's. Same
    # assertion, isolated from live state.
    lock = tmp_path / "train.lock"
    monkeypatch.setenv("LLM_TRAIN_LOCK", str(lock))
    importlib.reload(mod)
    held = mod.acquire_train_lock(
        lock, owner={"pid": os.getpid(), "note": "pytest owner"}
    )
    probe_lines = [
        "import builtins, runpy, sys",
        "real = builtins.__import__",
        "def guarded(name, *a, **k):",
        "    if name == 'torch' or name.startswith('torch.'):",
        "        print('TORCH_IMPORTED', file=sys.stderr)",
        "        raise AssertionError('torch imported before lock rejection')",
        "    return real(name, *a, **k)",
        "builtins.__import__ = guarded",
        "runpy.run_module('train.train', run_name='__main__')",
    ]
    probe = "\n".join(probe_lines)
    env = {**os.environ, "LLM_TRAIN_LOCK": str(lock), "PYTHONPATH": str(LAB)}
    try:
        proc = subprocess.run(
            [sys.executable, "-u", "-c", probe],
            cwd=LAB,
            text=True,
            capture_output=True,
            check=False,
            env=env,
            timeout=60,
        )
    finally:
        held.release()

    output = proc.stdout + proc.stderr
    assert proc.returncode != 0
    assert "train.train singleton lock" in output
    assert str(lock) in output
    assert "TORCH_IMPORTED" not in output
    assert "device=" not in output


def test_train_wrapper_releases_lock_when_training_raises(monkeypatch, tmp_path):
    trainer = importlib.import_module("train.train")
    # Same isolation as above. train.train imports acquire_train_lock INSIDE
    # main(), so there is no module attribute to patch — reloading
    # train.runtime_lock with the env override set is what makes main() resolve
    # the test lock path.
    lock = tmp_path / "train.lock"
    monkeypatch.setenv("LLM_TRAIN_LOCK", str(lock))
    mod = importlib.reload(runtime_lock_module())
    assert mod.DEFAULT_LOCK_PATH == lock

    def fail(_args):
        raise RuntimeError("synthetic training failure")

    monkeypatch.setattr(trainer, "_train", fail)
    monkeypatch.setattr(
        sys, "argv", ["train.train", "--preset", "t1m", "--steps", "1"]
    )
    with pytest.raises(RuntimeError, match="synthetic training failure"):
        trainer.main()

    next_owner = mod.acquire_train_lock(lock)
    next_owner.release()
