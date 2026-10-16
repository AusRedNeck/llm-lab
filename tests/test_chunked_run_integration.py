import json
import os
import subprocess
import sys
from pathlib import Path
import torch

LAB = Path(__file__).resolve().parents[1]


def run_segment(tmp, name, end, start=None):
    out, runs = tmp / f"{name}-ckpt", tmp / f"{name}-runs"
    cmd = [sys.executable, "-u", "-m", "train.train", "--preset", "t1m",
           "--steps", "4", "--stop-at-step", str(end), "--batch", "1",
           "--accum", "1", "--warmup", "1", "--val_every", "0",
           "--run_dir", str(runs), "--out", str(out)]
    if start:
        cmd += ["--resume", str(start)]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    return subprocess.run(cmd, cwd=LAB, env=env, text=True,
                          capture_output=True, check=True), out, runs


def test_two_chunks_match_uninterrupted_cpu_run(tmp_path):
    _, chunk_out, chunk_runs = run_segment(tmp_path, "chunked", 2)
    checkpoint = next(chunk_out.glob("*_step2.pt"))
    first = next(chunk_runs.glob("*/loss.jsonl"))
    assert any(json.loads(x).get("chunk_complete") for x in first.read_text().splitlines())
    run_segment(tmp_path, "chunked", 4, checkpoint)
    _, full_out, full_runs = run_segment(tmp_path, "full", 4)
    c_rows = [json.loads(s) for s in next(chunk_runs.glob("*/loss.jsonl")).read_text().splitlines()]
    f_rows = [json.loads(s) for s in next(full_runs.glob("*/loss.jsonl")).read_text().splitlines()]
    c_metrics = [r for r in c_rows if "train" in r]
    f_metrics = [r for r in f_rows if "train" in r]
    assert [(r["step"], r["avg50"]) for r in c_metrics] == [
        (r["step"], r["avg50"]) for r in f_metrics]
    a = torch.load(next(chunk_out.glob("*_step4.pt")), map_location="cpu", weights_only=False)
    b = torch.load(next(full_out.glob("*_step4.pt")), map_location="cpu", weights_only=False)
    assert all(torch.equal(a["model"][k], b["model"][k]) for k in a["model"])
    assert a["optimizer"]["state"].keys() == b["optimizer"]["state"].keys()
    assert "rng_state" in a and a["schedule_steps"] == 4
