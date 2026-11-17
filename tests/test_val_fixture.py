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


def _write_encoder(tmp_path):
    """A real, tiny HF tokenizer standing in for the production encoder.

    The sidecar's `vocab` must point at an encoder that exists in this test's
    tmp dir. Hardcoding the production path made the suite depend on an asset
    this checkout does not have (and on a machine that is not this one).
    """
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    tok = Tokenizer(models.BPE({"<unk>": 0, "a": 1}, [], unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    path = tmp_path / "pythia70m_hf" / "tokenizer.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(path))
    return str(path)


def _write_cache(tmp_path, name="toy.bin", tokens=TOKENS, meta_tokens=None,
                 ids=None):
    bin_path = tmp_path / name
    if ids is None:
        ids = (np.arange(tokens, dtype="<i4") % 50_257) + 1
    ids = np.asarray(ids, dtype="<i4")
    ids.tofile(str(bin_path))
    meta = {
        "tokens": tokens if meta_tokens is None else meta_tokens,
        "complete": True,
        "vocab": _write_encoder(tmp_path),
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
    _train, val = split_corpus(torch.from_numpy(np.array(arr, copy=True)), 0.01, ctx=8)
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
    fallback, _fp, _fm = _write_cache(tmp_path, name="fallback.bin")
    job = tmp_path / "train_job.json"
    job.write_text(json.dumps({
        "completed": True,
        "train_args": ["--tok_cache", other],
    }), encoding="utf-8")
    monkeypatch.setattr(val_fixture, "ACTIVE_JOB", str(job))
    monkeypatch.setattr(val_fixture, "FALLBACK_CACHE", fallback)
    resolved, _mp2, _meta = val_fixture.resolve()
    assert resolved == fallback
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
    assert enc == meta["vocab"]
    assert enc.endswith(os.path.join("pythia70m_hf", "tokenizer.json"))
    # a missing encoder is a hard stop: scoring with the wrong one redefines ids
    with pytest.raises(SystemExit, match="encoder .* not found"):
        val_fixture.encoder_path({"vocab": str(tmp_path / "nope.json")})


def test_reference_tokens_states_the_denominator():
    """An unversioned snapshot dir IS the final 143000-step checkpoint.

    The report must say so, because a 299.9B-token denominator measured against
    a 2B-token run is how "+18.7% behind" happened the first time.
    """
    from score_ours_vs_reference import reference_tokens
    tok, label = reference_tokens(
        os.path.join(ROOT, "data", "incoming", "pythia70m_weights"))
    assert tok == 143000 * 2048 * 1024
    assert "final" in label
    tok2, label2 = reference_tokens(
        os.path.join(ROOT, "data", "incoming", "pythia160m_step1000"))
    assert tok2 == 1000 * 2048 * 1024
    assert label2 == "step1000"


def test_trainer_bpt_matches_the_trainer_recipe(tmp_path):
    """bytes/token must be measured, not carried over between tails.

    Two pins: the helper must reproduce train.val_bytes_per_token exactly on a
    temporary cache (always), and it must reproduce the 70M trainer's recorded
    3.9104 for the OLD cache tail when that corpus is present.
    """
    from train.train import load_tokenizer, val_bytes_per_token
    # ids drawn from the tiny encoder's vocab: "a" (1 byte) and <unk> (4 bytes)
    raw = np.array([1, 0] * 2048, dtype="<i4")
    bin_path, _mp, meta = _write_cache(tmp_path, name="bpt.bin",
                                       tokens=len(raw), ids=raw)
    ids = val_fixture.tail_ids(bin_path, 0.01, meta)
    tok = load_tokenizer(meta["vocab"])
    expected = val_bytes_per_token(ids, tok)
    assert expected > 0
    assert val_fixture.trainer_bpt(ids, meta["vocab"]) == pytest.approx(expected)

    old_cache = os.path.join(ROOT, "data", "pile_train_full_bpe_pythia70m.bin")
    if not os.path.exists(old_cache):
        return
    meta = json.load(open(old_cache + ".meta.json", encoding="utf-8"))
    ids = val_fixture.tail_ids(old_cache, 0.01, meta)
    bpt = val_fixture.trainer_bpt(ids, val_fixture.encoder_path(meta))
    assert bpt == pytest.approx(3.9104, abs=5e-4)
