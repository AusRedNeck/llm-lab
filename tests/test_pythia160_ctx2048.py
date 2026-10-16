"""The 2K-context Pythia-160M experiment must be an explicit separate preset."""

from train.train import PRESETS


def test_pythia160_2k_preset_uses_2048_context_and_keeps_parity_knobs():
    cfg = PRESETS["pythia160_2k"]
    assert cfg.context_length == 2048
    assert (cfg.embedding_dim, cfg.num_layers, cfg.num_heads) == (768, 12, 12)
    assert cfg.vocab_size == 50304
    assert cfg.rotary_pct == 0.25
    assert cfg.use_rope is True
    assert cfg.parallel_residual is True
    assert cfg.dropout == 0.0
    assert cfg.use_learned_position_embeddings is False
    assert cfg.lm_head_bias is False


def test_rope_only_variant_has_no_learned_position_table():
    import torch
    from model.transformer import Transformer

    cfg = PRESETS["pythia160_2k"]
    model = Transformer.from_config(vars(cfg))
    assert not any("position_embedding" in name for name, _ in model.named_parameters())
    assert model.lm_head.bias is None
    out = model(torch.zeros((1, 7), dtype=torch.long))
    assert out.shape == (1, 7, 50304)


def test_legacy_config_defaults_to_learned_positions():
    from model.transformer import Transformer

    model = Transformer.from_config({
        "vocab_size": 32, "context_length": 8, "embedding_dim": 16,
        "num_heads": 4, "num_layers": 1,
    })
    assert any("position_embedding" in name for name, _ in model.named_parameters())
    assert model.lm_head.bias is not None


def test_checkpoint_config_records_effective_rope_setting():
    from train.train import PRESETS, checkpoint_config

    cfg = PRESETS["pythia160_2k"]
    saved = checkpoint_config(cfg, use_rope=True)
    assert saved["use_rope"] is True
    assert saved["use_learned_position_embeddings"] is False

