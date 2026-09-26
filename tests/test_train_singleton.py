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


def test_train_entrypoint_refuses_before_cuda_allocation():
    mod = runtime_lock_module()
    lock = mod.acquire_train_lock(
        mod.DEFAULT_LOCK_PATH, owner={"pid": os.getpid(), "note": "pytest owner"}
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
    try:
        proc = subprocess.run(
            [sys.executable, "-u", "-c", probe],
            cwd=LAB,
            text=True,
            capture_output=True,
            check=False,
            timeout=60,
        )
    finally:
        lock.release()

    output = proc.stdout + proc.stderr
    assert proc.returncode != 0
    assert "train.train singleton lock" in output
    assert str(mod.DEFAULT_LOCK_PATH) in output
    assert "TORCH_IMPORTED" not in output
    assert "device=" not in output


def test_train_wrapper_releases_lock_when_training_raises(monkeypatch):
    trainer = importlib.import_module("train.train")
    mod = runtime_lock_module()

    def fail(_args):
        raise RuntimeError("synthetic training failure")

    monkeypatch.setattr(trainer, "_train", fail)
    monkeypatch.setattr(
        sys, "argv", ["train.train", "--preset", "t1m", "--steps", "1"]
    )
    with pytest.raises(RuntimeError, match="synthetic training failure"):
        trainer.main()

    next_owner = mod.acquire_train_lock(mod.DEFAULT_LOCK_PATH)
    next_owner.release()
