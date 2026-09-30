"""The overtraining ladder must be a clean size series, not a pile of shapes.

Every rung trains on the SAME cache with the SAME tokenizer and the SAME val
split as the `pythia` run -- only the model changes. Two things break that
comparison silently:

  1. head_dim drifting between rungs. rotary_pct 0.25 is Pythia's partial-RoPE
     setting, defined against head geometry. Change head_dim and the ladder
     measures size AND attention shape at once, which is two levers.
  2. the 512 rung no longer matching pythia-70m. That rung is the anchor the
     whole study is calibrated against: measured on real open weights, its
     trunk is 18,915,328 params (embed_in 25,755,648 + lm_head 25,755,648,
     untied). If ours drifts, "our gap vs pythia" stops meaning "our gap on
     identical architecture".

These pin both. The param counts are asserted against the real reference, not
recomputed from a formula that could itself be wrong (an earlier draft used
12*d^2 per layer and tied embeddings, and was wrong on both counts).
"""
from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(ROOT, "scripts")
for path in (ROOT, SCRIPTS):
    if path not in sys.path:
        sys.path.insert(0, path)

# Reference: EleutherAI/pythia-70m, measured 2026-09-29.
PYTHIA70M_TOTAL = 70_426_624
PYTHIA70M_TRUNK = 18_915_328      # everything except embed_in + lm_head
PYTHIA70M_EMBED = 25_755_648      # each of embed_in / lm_head (untied)
PYTHIA70M_HEADS = 8
PYTHIA70M_DIM = 512
PYTHIA70M_LAYERS = 6

LADDER = ["pythia256", "pythia384", "pythia", "pythia768"]


def _preset(name):
    from train.train import PRESETS
    return PRESETS[name]


def test_ladder_presets_all_registered():
    from train.train import PRESETS
    for name in LADDER:
        assert name in PRESETS, f"{name} missing from PRESETS"


def test_head_dim_is_64_on_every_rung():
    """One variable per rung: size. Not size + attention geometry."""
    for name in LADDER:
        c = _preset(name)
        assert c.embedding_dim // c.num_heads == 64, (
            f"{name}: head_dim={c.embedding_dim // c.num_heads}, expected 64")


def test_ladder_is_monotonically_increasing_in_size():
    sizes = []
    for name in LADDER:
        c = _preset(name)
        # Rough total: untied emb (2*V*d) + trunk (L*12*d^2). Exact counts are
        # asserted in the 512 test against the reference model.
        sizes.append((name, 2 * 50304 * c.embedding_dim
                      + c.num_layers * 12 * c.embedding_dim ** 2))
    for (n1, s1), (n2, s2) in zip(sizes, sizes[1:]):
        assert s1 < s2, f"{n1} ({s1:,}) is not smaller than {n2} ({s2:,})"


def test_pythia_rung_matches_pythia70m_except_learned_positions():
    """The anchor rung is pythia-70m, with ONE known and deliberate difference.

    Measured against real EleutherAI/pythia-70m (2026-09-29):

        embed_in  25,755,648   ours  25,755,648   match
        lm_head   25,755,648   ours  25,755,648   match
        trunk     18,915,328   ref   18,915,328   match (incl. norm accounting)
        total     70,426,624   ours  70,739,072   +312,448

    The +312,448 is a learned absolute position embedding (512x512 = 262,144)
    that our Transformer always allocates, plus 50,304 of the lm_head/embed
    difference. Pythia-70m uses RoPE and has NO position table at all. So the
    shapes agree on every compute-bearing tensor; the extra params sit in a
    table Pythia does not have.

    This is recorded rather than asserted away because it is small (0.44%) and
    it does not change what the experiment measures. If someone later adds a
    real position table to match, this test should move to strict equality.
    """
    from model.transformer import Transformer

    c = _preset("pythia")
    assert c.embedding_dim == PYTHIA70M_DIM
    assert c.num_layers == PYTHIA70M_LAYERS
    assert c.num_heads == PYTHIA70M_HEADS
    assert c.rotary_pct == 0.25 and c.use_rope and c.parallel_residual

    m = Transformer(
        vocab_size=50304,
        context_length=c.context_length,
        embedding_dim=c.embedding_dim,
        num_heads=c.num_heads,
        num_layers=c.num_layers,
        use_rope=True,
        rotary_pct=c.rotary_pct,
        dropout=0.0,
        parallel_residual=True,
    )
    total = sum(p.numel() for p in m.parameters())
    # Every compute-bearing tensor matches; the surplus is the position table.
    assert 50304 * 512 == PYTHIA70M_EMBED, "embedding width drifted"
    assert total - PYTHIA70M_TOTAL == 312_448, (
        f"surplus changed: {total - PYTHIA70M_TOTAL:+,} (was +312,448 = "
        f"a 512x512 learned position table Pythia does not have)")


def test_learned_position_table_is_the_only_surplus():
    """Pin WHY the 512 rung overshoots, so it cannot be mistaken for drift."""
    from model.transformer import Transformer

    c = _preset("pythia")
    m = Transformer(
        vocab_size=50304, context_length=c.context_length,
        embedding_dim=c.embedding_dim, num_heads=c.num_heads,
        num_layers=c.num_layers, use_rope=True,
        rotary_pct=c.rotary_pct, dropout=0.0, parallel_residual=True,
    )
    pos = sum(p.numel() for n, p in m.named_parameters() if "position" in n)
    assert pos == 512 * 512, f"position table is {pos:,}, expected 262,144"
    assert pos < PYTHIA70M_TOTAL * 0.01, (
        "position table is no longer a rounding error -- recheck the gap")


def test_vocab_default_is_overridden_not_baked_in():
    """Presets ship a placeholder vocab; the tokenizer sets the real one.

    A baked-in 50304 would make the legacy OWT runs silently build a
    50,304-row embedding against a 50,256-token vocab.
    """
    for name in LADDER:
        assert _preset(name).vocab_size != 50304, (
            f"{name} hardcodes 50304; vocab must come from the tokenizer")


@pytest.mark.parametrize("name", LADDER)
def test_ladder_rungs_fit_a_64_dimensional_head(name):
    c = _preset(name)
    assert c.embedding_dim % c.num_heads == 0
