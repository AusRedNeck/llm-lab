"""Tests for eval/benchmark.py — the fair eval harness."""

import math
import os
import tempfile

import pytest
import torch

from eval.benchmark import (
    nats_per_byte,
    eval_checkpoint,
    eval_report,
    perplexity,
    token_accuracy,
)


# ── nats_per_byte ──────────────────────────────────────────────────

def test_nats_per_byte_basic():
    """val_nll=2.0, bytes_per_token=4.0 -> 0.5 nats/byte."""
    assert nats_per_byte(2.0, 4.0) == pytest.approx(0.5)


def test_nats_per_byte_single_byte_tokens():
    """Byte-level: bytes_per_token=1.0 -> nats/byte == val_nll."""
    assert nats_per_byte(0.73, 1.0) == pytest.approx(0.73)


def test_nats_per_byte_high_compression():
    """BPE with 8x compression: 2.0 / 8.0 = 0.25."""
    assert nats_per_byte(2.0, 8.0) == pytest.approx(0.25)


def test_nats_per_byte_zero_bytes_raises():
    """Can't divide by zero — degenerate tokenizer."""
    with pytest.raises(ValueError):
        nats_per_byte(1.0, 0.0)


def test_nats_per_byte_negative_bytes_raises():
    """Negative bytes-per-token is nonsensical."""
    with pytest.raises(ValueError):
        nats_per_byte(1.0, -1.0)


def test_nats_per_byte_zero_nll():
    """Perfect model (0 nll) -> 0 nats/byte regardless of compression."""
    assert nats_per_byte(0.0, 4.0) == pytest.approx(0.0)


# ── eval_checkpoint ────────────────────────────────────────────────

def _make_toy_checkpoint(path: str, steps: int = 10):
    """Create a minimal checkpoint file for testing."""
    from model.config import T1M_4L128
    from model.transformer import Transformer

    cfg = T1M_4L128
    model = Transformer(
        vocab_size=cfg.vocab_size,
        context_length=cfg.context_length,
        embedding_dim=cfg.embedding_dim,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
    )
    torch.save({
        "cfg": {
            "vocab_size": cfg.vocab_size,
            "context_length": cfg.context_length,
            "embedding_dim": cfg.embedding_dim,
            "num_heads": cfg.num_heads,
            "num_layers": cfg.num_layers,
            "use_rope": False,
            "tokenizer": None,
        },
        "model": model.state_dict(),
        "step": steps,
    }, path)


def test_eval_checkpoint_returns_dict():
    """eval_checkpoint returns a dict with expected keys."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "test_ckpt.pt")
        _make_toy_checkpoint(ckpt_path)
        result = eval_checkpoint(ckpt_path, val_batches=2)
        assert isinstance(result, dict)
        assert "val_loss" in result
        assert "perplexity" in result
        assert "nats_per_byte" in result
        assert "token_accuracy" in result
        assert "step" in result


def test_eval_checkpoint_step_matches():
    """The step in the result matches the checkpoint."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "test_ckpt.pt")
        _make_toy_checkpoint(ckpt_path, steps=42)
        result = eval_checkpoint(ckpt_path, val_batches=1)
        assert result["step"] == 42


def test_eval_checkpoint_perplexity_is_exp_of_loss():
    """perplexity should equal exp(val_loss)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "test_ckpt.pt")
        _make_toy_checkpoint(ckpt_path)
        result = eval_checkpoint(ckpt_path, val_batches=2)
        assert result["perplexity"] == pytest.approx(
            math.exp(result["val_loss"]), rel=1e-5
        )


def test_eval_checkpoint_nats_per_byte_consistent():
    """nats_per_byte should equal val_loss / bytes_per_token."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "test_ckpt.pt")
        _make_toy_checkpoint(ckpt_path)
        result = eval_checkpoint(ckpt_path, val_batches=2)
        # For byte-level (vocab 256), bytes_per_token = 1.0
        assert result["nats_per_byte"] == pytest.approx(
            result["val_loss"], rel=1e-5
        )


def test_eval_checkpoint_accepts_device():
    """eval_checkpoint should accept a device argument."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "test_ckpt.pt")
        _make_toy_checkpoint(ckpt_path)
        device = torch.device("cpu")
        result = eval_checkpoint(ckpt_path, val_batches=1, device=device)
        assert "val_loss" in result


# ── eval_report ────────────────────────────────────────────────────

def test_eval_report_returns_list():
    """eval_report returns a list of dicts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt1 = os.path.join(tmpdir, "ckpt1.pt")
        ckpt2 = os.path.join(tmpdir, "ckpt2.pt")
        _make_toy_checkpoint(ckpt1, steps=100)
        _make_toy_checkpoint(ckpt2, steps=200)
        results = eval_report([ckpt1, ckpt2], val_batches=1)
        assert isinstance(results, list)
        assert len(results) == 2


def test_eval_report_sorted_by_nats_per_byte():
    """Results should be sorted by nats_per_byte (lower = better)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt1 = os.path.join(tmpdir, "ckpt1.pt")
        ckpt2 = os.path.join(tmpdir, "ckpt2.pt")
        _make_toy_checkpoint(ckpt1, steps=100)
        _make_toy_checkpoint(ckpt2, steps=200)
        results = eval_report([ckpt1, ckpt2], val_batches=2)
        nats = [r["nats_per_byte"] for r in results]
        assert nats == sorted(nats)


def test_eval_report_labels():
    """Each result should have a 'label' derived from the filename."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt = os.path.join(tmpdir, "exp004_best.pt")
        _make_toy_checkpoint(ckpt)
        results = eval_report([ckpt], val_batches=1)
        assert results[0]["label"] == "exp004_best"
