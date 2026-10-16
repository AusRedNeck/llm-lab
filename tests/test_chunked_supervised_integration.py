import json
import os
import sys
import subprocess
from pathlib import Path
from types import SimpleNamespace

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / "scripts"))
import train_watchdog as w
import continue_train_chunk as c

REAL_RUN = subprocess.run


def run_trainer(root, spec):
    log = root / spec["log"]
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [spec["python"], "-u", "-m", "train.train"] + spec["train_args"]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    with log.open("w", encoding="utf-8") as stream:
        result = REAL_RUN(cmd, cwd=LAB, env=env, stdout=stream,
                          stderr=subprocess.STDOUT, text=True)
    if result.returncode:
        raise AssertionError(log.read_text(encoding="utf-8"))
    return log


def test_supervised_chunk_pauses_then_continues_on_explicit_go(tmp_path, monkeypatch):
    (tmp_path / "logs").mkdir()
    runs, ckpts = tmp_path / "runs", tmp_path / "checkpoints"
    args = ["--preset", "t1m", "--steps", "4", "--stop-at-step", "2",
            "--batch", "1", "--accum", "1", "--warmup", "1", "--val_every", "0",
            "--run_dir", str(runs), "--out", str(ckpts)]
    spec = {"name": "supervised-smoke", "target_steps": 2, "schedule_steps": 4,
            "chunk_steps": 2, "chunk_index": 1, "completed": False,
            "next_job": None, "run_name": "", "stamp": "", "ckpt_dir": str(ckpts),
            "python": sys.executable, "train_args": args, "log": "logs/chunk01.log",
            "match": ["train.train", "--preset t1m"], "max_restarts": 3}
    state = {"job": spec["name"], "run_name": "", "status": "running"}
    spec_path = tmp_path / "train_job.json"
    state_path = tmp_path / "train_job.state.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    state_path.write_text(json.dumps(state), encoding="utf-8")
    row = {"id": spec["name"], "run_dirs": [], "verdict": "running"}
    (tmp_path / "experiments.json").write_text(json.dumps({"arms": [row]}))

    monkeypatch.setattr(w, "LAB", str(tmp_path))
    run_trainer(tmp_path, spec)
    assert w.pin_run_identity(spec, str(spec_path)) is True
    state["run_name"] = spec["run_name"]
    state_path.write_text(json.dumps(state), encoding="utf-8")
    monkeypatch.setattr(w, "LAB", str(tmp_path))
    monkeypatch.setattr(w, "SPEC", str(spec_path))
    monkeypatch.setattr(w, "running_process", lambda _spec: (None, True))
    monkeypatch.setattr(sys, "argv", ["train_watchdog.py", "--spec", str(spec_path)])
    assert w._main_locked() == 0
    paused = json.loads(state_path.read_text())
    assert paused["status"] == "paused_at_chunk" and paused["paused_step"] == 2

    monkeypatch.setattr(c, "_running_trainers", lambda: ([], True))
    def dispatch(_cmd, **_kwargs):
        next_spec = json.loads(spec_path.read_text())
        run_trainer(tmp_path, next_spec)
        monkeypatch.setattr(sys, "argv", ["train_watchdog.py", "--spec", str(spec_path)])
        assert w._main_locked() == 0
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(c.subprocess, "run", dispatch)
    assert c.main(["--spec", str(spec_path), "--stop-at-step", "4"]) == 0
    final = json.loads(state_path.read_text())
    assert final["status"] == "complete" and final["final_step"] == 4
    assert json.loads(spec_path.read_text())["completed"] is True
    reg = json.loads((tmp_path / "experiments.json").read_text())
    assert len(reg["arms"]) == 1 and len(reg["arms"][0]["run_dirs"]) == 1
