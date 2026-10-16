import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import continue_train_chunk as c


def test_dry_run_previews_chunk_without_mutating_files(tmp_path, monkeypatch):
    run = "20261011_1000_pythia160_2k"
    ckpts = tmp_path / "checkpoints"
    ckpts.mkdir()
    ckpt = ckpts / f"exp002_pythia160_2k_{run.split('_')[0]}{run.split('_')[1]}_step4000.pt"
    torch.save({"step": 4000, "schedule_steps": 16000, "run_name": run,
               "rng_state": {"cpu": torch.get_rng_state(), "cuda": None}}, ckpt)
    spec = {"name": "arm", "target_steps": 4000, "schedule_steps": 16000,
            "chunk_steps": 4000, "chunk_index": 1, "completed": False,
            "run_name": run, "stamp": "202610111000", "ckpt_dir": str(ckpts),
            "python": sys.executable, "train_args": ["--steps", "16000",
            "--stop-at-step", "4000", "--preset", "pythia160_2k"]}
    state = {"job": "arm", "run_name": run, "status": "paused_at_chunk"}
    sp, st = tmp_path / "train_job.json", tmp_path / "train_job.state.json"
    sp.write_text(json.dumps(spec), encoding="utf-8")
    st.write_text(json.dumps(state), encoding="utf-8")
    before = (sp.read_text(), st.read_text())
    monkeypatch.setattr(c, "_running_trainers", lambda: ([], True))
    assert c.main(["--spec", str(sp), "--stop-at-step", "8000", "--dry-run"]) == 0
    assert (sp.read_text(), st.read_text()) == before
    calls = []
    monkeypatch.setattr(c.subprocess, "run", lambda cmd, **kw: (calls.append(cmd) or SimpleNamespace(returncode=0)))
    assert c.main(["--spec", str(sp), "--stop-at-step", "8000"]) == 0
    updated = json.loads(sp.read_text())
    resumed = json.loads(st.read_text())
    assert updated["target_steps"] == 8000 and updated["schedule_steps"] == 16000
    assert updated["train_args"][-2] == "--resume"
    assert updated["train_args"][-1].endswith("step4000.pt")
    assert resumed["status"] == "armed" and resumed["chunk_start_step"] == 4000
    assert calls and calls[0][1].lower().endswith("train_watchdog.py")
