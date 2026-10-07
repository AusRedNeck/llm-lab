"""The matched-token anchor must survive pruning.

step 16,000 of a 131,072-tok/step run is pythia-160m@step1000 to the token
(131,072 x 16,000 = 2,097,152,000), and it is NOT a multiple of the 5000
milestone grid -- so the policy would delete the checkpoint the parity run's
headline comparison is measured from. These tests pin the arithmetic, the
per-run effective-batch lookup, and the refusal to guess.
"""
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import prune_checkpoints as pc  # noqa: E402


def _write_header(runs_dir, stamp, batch=8, accum=32, ctx=512):
    run_dir = os.path.join(runs_dir, f"{stamp[:8]}_{stamp[8:]}")
    os.makedirs(run_dir, exist_ok=True)
    head = {"args": {"batch": batch, "accum": accum},
            "cfg": {"context_length": ctx}}
    with open(os.path.join(run_dir, "loss.jsonl"), "w", encoding="utf-8") as f:
        f.write(json.dumps(head) + "\n")
    return run_dir


def test_parity_run_anchors_on_pythia_step1000(tmp_path):
    # 8 x 32 x 512 = 131,072 tok/step; 2,097,152,000 / 131,072 = 16,000 exactly
    _write_header(str(tmp_path), "202610070000")
    anchors = pc.anchor_steps("202610070000", runs_dir=str(tmp_path))
    assert anchors == {16_000, 32_000}
    assert 2_097_152_000 % 131_072 == 0


def test_anchor_math_matches_pythia_tokens():
    # eff batches whose anchors we can state without a filesystem
    assert pc.PYTHIA_ANCHOR_TOKENS[0] == 2_097_152_000          # pythia step1000
    assert pc.PYTHIA_ANCHOR_TOKENS[1] == 4_194_304_000          # pythia step2000
    assert 2_097_152_000 % (2048 * 1024) == 0                   # pythia's own step


def test_missing_header_returns_empty_not_a_guess(tmp_path):
    # No loss.jsonl -> no anchors. Guessing an effective batch here would
    # "protect" a step that is not the anchor, which is worse than protecting
    # nothing: it looks like coverage and isn't.
    assert pc.anchor_steps("202610071111", runs_dir=str(tmp_path)) == set()
    assert pc.eff_tokens_per_step("202610071111", runs_dir=str(tmp_path)) is None


def test_malformed_stamp_or_header_is_inert(tmp_path):
    assert pc.anchor_steps("", runs_dir=str(tmp_path)) == set()
    assert pc.anchor_steps("short", runs_dir=str(tmp_path)) == set()
    run_dir = _write_header(str(tmp_path), "202610072222")
    with open(os.path.join(run_dir, "loss.jsonl"), "w", encoding="utf-8") as f:
        f.write("not json\n")
    assert pc.anchor_steps("202610072222", runs_dir=str(tmp_path)) == set()


def test_real_probe_stamp_anchors_but_has_no_file_to_keep():
    """Live check: the two probe stamps resolve anchors (they trained at
    131,072 tok/step) yet hold no step 16000/32000, so the policy output for
    today's prune is unchanged -- proof the new rule is inert until it matters."""
    if not os.path.isdir(pc.RUNS):
        pytest.skip("runs dir not present")
    for stamp in ("202610062126", "202610062350"):
        anchors = pc.anchor_steps(stamp)
        assert anchors == {16_000, 32_000}
        have = [f for f in os.listdir(pc.CKPT)
                if stamp in f and any(f.endswith(f"step{s}.pt") for s in anchors)]
        assert have == []
