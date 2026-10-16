import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from continue_train_chunk import prepare_next_chunk


def fixture_data():
    spec = {"name": "arm", "target_steps": 4000, "schedule_steps": 16000,
            "chunk_steps": 4000, "chunk_index": 1, "completed": False,
            "run_name": "20261011_1000_pythia160_2k", "stamp": "202610111000",
            "log": "logs/arm_chunk01.log", "next_job": None,
            "train_args": ["--steps", "16000", "--stop-at-step", "4000",
                           "--resume", "old.pt"]}
    state = {"job": "arm", "run_name": spec["run_name"],
             "status": "paused_at_chunk", "relaunch_count": 0}
    ckpt = {"step": 4000, "schedule_steps": 16000,
            "run_name": spec["run_name"], "rng_state": {"cpu": "saved"}}
    return spec, state, ckpt


def test_continue_advances_same_run_to_next_boundary():
    spec, state, ckpt = fixture_data()
    new, ns = prepare_next_chunk(spec, state, 8000, "checkpoints/step4000.pt", ckpt)
    assert new["name"] == "arm" and new["run_name"] == spec["run_name"]
    assert (new["target_steps"], new["schedule_steps"], new["chunk_index"]) == (8000, 16000, 2)
    assert new["train_args"][-2:] == ["--resume", "checkpoints/step4000.pt"]
    assert ns["status"] == "armed" and ns["relaunch_count"] == 0
    assert ns["watchdog_last_step"] is None and "watchdog_last_at" not in ns


def test_continue_rejects_nonpaused_or_wrong_checkpoint():
    spec, state, ckpt = fixture_data()
    with pytest.raises(ValueError, match="paused"):
        prepare_next_chunk(spec, {**state, "status": "running"}, 8000, "x.pt", ckpt)
    with pytest.raises(ValueError, match="paused step"):
        prepare_next_chunk(spec, {**state, "paused_step": 3000}, 8000, "x.pt", ckpt)
    with pytest.raises(ValueError, match="checkpoint"):
        prepare_next_chunk(spec, state, 8000, "x.pt", {**ckpt, "step": 3500})
    with pytest.raises(ValueError, match="RNG"):
        prepare_next_chunk(spec, state, 8000, "x.pt", {**ckpt, "rng_state": None})


def test_continue_rejects_nonsequential_or_changed_horizon():
    spec, state, ckpt = fixture_data()
    with pytest.raises(ValueError, match="next chunk"):
        prepare_next_chunk(spec, state, 9000, "x.pt", ckpt)
    with pytest.raises(ValueError, match="schedule"):
        prepare_next_chunk(spec, state, 8000, "x.pt", {**ckpt, "schedule_steps": 8000})


def test_continue_requires_pinned_identity_and_matching_job():
    spec, state, ckpt = fixture_data()
    with pytest.raises(ValueError, match="identity"):
        prepare_next_chunk({**spec, "stamp": ""}, state, 8000, "x.pt", ckpt)
    with pytest.raises(ValueError, match="job"):
        prepare_next_chunk(spec, {**state, "job": "other"}, 8000, "x.pt", ckpt)
