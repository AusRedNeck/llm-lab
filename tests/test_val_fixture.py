"""val_fixture — the shared val fixture must resolve, cut, and refuse to drift.

These pin three historical failure modes:
  * scorers hardcoding a cache (the 70M cache's tail is inside the 160M train
    set) — resolve() must follow the ACTIVE run, never a completed one;
  * a stale sidecar silently moving the cut;
  * a cut that disagrees with train.split_corpus (a "same ids" claim that is
    only a comment).
"""
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import val_fixture  # noqa: E402

TOKENS = 10_000


def _write_cache(tmp_path, name="toy.bin", tokens=TOKENS, meta_tokens=None):
    bin_path = tmp_path / name
    ids = (np.arange(tokens, dtype="<i4") % 50_257) + 1
    ids.tofile(str(bin_path))
    meta = {
        "tokens": tokens if meta_tokens is None else meta_tokens,
        "complete": True,
        "vocab": os.path.join(ROOT, "data", "incoming", "pythia70m_hf",
                              "tokenizer.json"),
        "src": "toy",
    }
    meta_path = tmp_path / (name + ".meta.json")
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    return str(bin_path), str(meta_path), meta


def test_cut_matches_split_corpus_formula(tmp_path):
    bin_path, _mp, meta = _write_cache(tmp_path)
    total = val_fixture.total_tokens(bin_path, meta)
    assert total == TOKENS
    # train.split_corpus: cut = int(len(corpus) * (1 - val_frac))
    assert val_fixture.cut_index(total, 0.01) == int(TOKENS * (1 - 0.01))
    ids = val_fixture.tail_ids(bin_path, 0.01, meta)
    cut = val_fixture.cut_index(total, 0.01)
    assert ids.shape[0] == TOKENS - cut
    np.testing.assert_array_equal(ids, np.arange(cut, TOKENS) % 50_257 + 1)


def test_tail_is_exactly_what_split_corpus_holds_out(tmp_path):
    import torch
    from train.train import split_corpus
    bin_path, _mp, meta = _write_cache(tmp_path)
    arr = np.memmap(bin_path, dtype=val_fixture.DTYPE, mode="r")
    _train, val = split_corpus(torch.from_numpy(arr), 0.01, ctx=8)
    fixture = val_fixture.tail_ids(bin_path, 0.01, meta)
    assert val is not None
    np.testing.assert_array_equal(val.numpy().astype(np.int64), fixture)


def test_stale_sidecar_is_refused_not_ignored(tmp_path):
    # meta claiming a different token count than the file holds: the cut would
    # silently move, so this must raise instead of guessing.
    bin_path, _mp, meta = _write_cache(tmp_path, meta_tokens=TOKENS + 5)
    with pytest.raises(SystemExit, match="sidecar is stale"):
        val_fixture.total_tokens(bin_path, meta)


def test_explicit_cache_argument_wins(tmp_path, monkeypatch):
    bin_path, _mp, meta = _write_cache(tmp_path)
    monkeypatch.setattr(val_fixture, "ACTIVE_JOB", str(tmp_path / "missing.json"))
    resolved, meta_path, _meta = val_fixture.resolve(bin_path)
    assert resolved == bin_path
    assert meta_path == bin_path + ".meta.json"


def test_completed_job_does_not_pin_the_fixture(tmp_path, monkeypatch):
    """A finished run must not choose the fixture the next run scores on."""
    other, _mp, _om = _write_cache(tmp_path, name="old_cache.bin")
    job = tmp_path / "train_job.json"
    job.write_text(json.dumps({
        "completed": True,
        "train_args": ["--tok_cache", other],
    }), encoding="utf-8")
    monkeypatch.setattr(val_fixture, "ACTIVE_JOB", str(job))
    resolved, _mp2, _meta = val_fixture.resolve()
    assert resolved == val_fixture.FALLBACK_CACHE
    assert resolved != other


def test_running_job_cache_is_used(tmp_path, monkeypatch):
    active, _mp, _om = _write_cache(tmp_path, name="active.bin")
    job = tmp_path / "train_job.json"
    job.write_text(json.dumps({
        "completed": False,
        "train_args": ["--preset", "pythia160", "--tok_cache", active],
    }), encoding="utf-8")
    monkeypatch.setattr(val_fixture, "ACTIVE_JOB", str(job))
    resolved, _mp2, _meta = val_fixture.resolve()
    assert resolved == active


def test_encoder_comes_from_the_sidecar(tmp_path):
    _bp, _mp, meta = _write_cache(tmp_path)
    enc = val_fixture.encoder_path(meta)
    assert enc.endswith(os.path.join("pythia70m_hf", "tokenizer.json"))
    # a missing encoder is a hard stop: scoring with the wrong one redefines ids
    with pytest.raises(SystemExit, match="encoder .* not found"):
        val_fixture.encoder_path({"vocab": str(tmp_path / "nope.json")})


def test_trainer_bpt_matches_the_trainer_recipe():
    """bytes/token must be measured, not carried over between tails.

    The 70M trainer recorded 3.9104 for the OLD cache tail; recomputing it with
    this helper must reproduce that number, which is what proves the recipe is
    the trainer's and not an approximation of it.
    """
    old_cache = os.path.join(ROOT, "data", "pile_train_full_bpe_pythia70m.bin")
    if not os.path.exists(old_cache):
        pytest.skip("old 70M cache not present")
    meta = json.load(open(old_cache + ".meta.json", encoding="utf-8"))
    ids = val_fixture.tail_ids(old_cache, 0.01, meta)
    bpt = val_fixture.trainer_bpt(ids, val_fixture.encoder_path(meta))
    assert bpt == pytest.approx(3.9104, abs=5e-4)
