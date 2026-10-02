"""Tests for scripts/readme_state.py — the README's generated state block.

The whole point of the generator is that it can't lie. Two ways it could:

  1. RANKING bpb ACROSS VAL SLICES. A Pile run and an OpenWebText run score
     different text; putting them in one leaderboard is the same
     mismatched-denominator mistake as the superseded "+18.7% behind pythia"
     figure (commit a3a3b8b). They must land in separate groups.

  2. PRINTING A VERDICT THE DISK DISAGREES WITH. An arm marked `running` whose
     curve stopped days ago should raise a drift warning, not be printed as
     current fact.

Also covered: the splice only touches the fenced region, and the fence is
mandatory (a README without it must fail loudly, not silently no-op).
"""
import json
import os
import time

import pytest

from scripts.readme_state import (
    FENCE_BEGIN,
    FENCE_END,
    build_state,
    splice,
)
from viz import dashboard

# Grab the real loader BEFORE any patching, or the patch calls itself.
_real_load_runs = dashboard.load_runs


def _write_run(runs_dir, name, corpus, best_bpb, best_step, *, mtime=None):
    d = runs_dir / name
    d.mkdir(parents=True)
    header = {"args": {"preset": "pythia", "batch": 16, "accum": 16,
                       "corpus": corpus},
              "cfg": {"context_length": 512},
              "params_m": 70.7,
              "train_tokens": 1_983_952_082}
    rows = [{"step": 1, "train": 6.0, "avg50": 6.0, "val": None},
            {"step": best_step, "train": 3.0, "avg50": 3.0,
             "val": 3.1, "val_bpb": best_bpb}]
    with open(d / "loss.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps(header) + "\n")
        for r in rows:
            f.write(json.dumps(r) + "\n")
    if mtime is not None:
        os.utime(d / "loss.jsonl", (mtime, mtime))
    return d


def _wire(tmp_path, monkeypatch, runs_dir, arms=None):
    """Point the generator at a fake lab: our runs dir, optional registry."""
    monkeypatch.setattr("viz.dashboard.load_runs",
                        lambda *a, **k: _real_load_runs(str(runs_dir)))
    monkeypatch.setattr("scripts.readme_state.ROOT", tmp_path)
    exp = tmp_path / "experiments.json"
    exp.write_text(json.dumps({"arms": arms or []}), encoding="utf-8")
    monkeypatch.setattr("scripts.readme_state.EXPERIMENTS", exp)


def _runs_dir(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    return runs


def test_slices_never_merge_into_one_leaderboard(tmp_path, monkeypatch):
    """A better bpb on Pile must NOT be ranked against an OWT run."""
    runs = _runs_dir(tmp_path)
    # Pile scores BETTER (1.40) than OWT (1.47). If these merge, one table
    # implies cross-corpus comparability -- the a3a3b8b mistake.
    _write_run(runs, "pile_run", "data/pile_train_full.txt", 1.4033, 13400)
    _write_run(runs, "owt_run", "data/openwebtext_combined.txt", 1.4699, 12000)
    _wire(tmp_path, monkeypatch, runs)

    block = build_state()

    pile_section = block.split("### Best on Pile")[1].split("###")[0]
    owt_section = block.split("### Best on OpenWebText")[1].split("###")[0]
    assert "pile_run" in pile_section
    assert "owt_run" not in pile_section
    assert "owt_run" in owt_section
    assert "pile_run" not in owt_section
    assert "do not rank it against a run from another slice" in pile_section


def test_short_runs_are_excluded(tmp_path, monkeypatch):
    """A 200-step smoke is a launch receipt, not a scoreboard entry.

    MIN_STEP gates the RANKING only. A short run still belongs in the live
    table -- it is actively training and the operator needs to see it.
    """
    runs = _runs_dir(tmp_path)
    _write_run(runs, "smoke", "data/pile_train_full.txt", 1.88, 200)
    _wire(tmp_path, monkeypatch, runs)

    block = build_state()
    scoreboard = block.split("### In flight")[0]
    assert "smoke" not in scoreboard
    assert "No run has scored a usable best-bpb yet" in scoreboard


def test_stale_running_verdict_raises_drift_warning(tmp_path, monkeypatch):
    """`running` + dead curve = drift. Say so instead of printing as fact."""
    runs = _runs_dir(tmp_path)
    _write_run(runs, "finished_run", "data/pile_train_full.txt", 1.4033,
               13400, mtime=time.time() - 86400)
    _wire(tmp_path, monkeypatch, runs, arms=[
        {"id": "some-arm", "family": "f", "verdict": "running", "goal": "g",
         "run_dirs": ["finished_run"]}])

    block = build_state()
    assert "Verdict drift" in block
    assert "some-arm" in block


def test_live_run_is_not_flagged_as_drift(tmp_path, monkeypatch):
    """A genuinely moving curve must not raise a false alarm."""
    runs = _runs_dir(tmp_path)
    _write_run(runs, "live_run", "data/pile_train_full.txt", 1.4033, 13400)
    _wire(tmp_path, monkeypatch, runs, arms=[
        {"id": "live-arm", "family": "f", "verdict": "running", "goal": "g",
         "run_dirs": ["live_run"]}])

    assert "Verdict drift" not in build_state()


def test_splice_preserves_everything_outside_the_fences():
    """Prose above and below the block is the author's; never touch it."""
    doc = (f"# Title\n\nHand-written intro.\n\n{FENCE_BEGIN}\n"
           f"old generated junk\n{FENCE_END}\n\nHand-written footer.\n")
    out = splice(doc, "new generated content")
    assert "Hand-written intro." in out
    assert "Hand-written footer." in out
    assert "old generated junk" not in out
    assert "new generated content" in out
    assert out.count(FENCE_BEGIN) == 1
    assert out.count(FENCE_END) == 1


def test_splice_refuses_a_readme_with_no_fences():
    """No fence = someone removed the contract. Fail loudly."""
    with pytest.raises(SystemExit):
        splice("# A README with no fences\n", "content")