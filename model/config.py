# Model sizes: pick your fight. T1M proves the wiring, S17M learns to talk.
# Names are honest: TIER + params(M) + layers + width. T<2M, S<25M, M<100M, L above.
# Param counts are at the declared vocab; --tokenizer overrides vocab at
# runtime, so train.py always prints the true count for the run.
from dataclasses import dataclass


@dataclass
class ModelConfig:
    # Knobs that set the size of the engine.
    """Configuration for our small experimental language model."""
    vocab_size: int = 256
    context_length: int = 128
    embedding_dim: int = 128
    num_layers: int = 4
    num_heads: int = 4
    dropout: float = 0.0

    def num_params(self) -> int:
        """Rough parameter count so we know what class we're training."""
        # embeddings + pos emb + per-layer (attn 4xD^2 + ffn 8xD^2) + final norm + head
        d = self.embedding_dim
        per_layer = 4 * d * d + 8 * d * d
        total = (
            self.vocab_size * d
            + self.context_length * d
            + self.num_layers * per_layer
            + 2 * d  # final LayerNorm
            + d * self.vocab_size + self.vocab_size  # lm_head
        )
        return total


# The toy (~0.9M params, byte-level). Good for proving
# loss goes down, not for real English.
T1M_4L128 = ModelConfig()

# GPT-2-vocab legacy shape (~49.5M with vocab 50304).
# vocab 50304 = GPT-2 BPE size (padded to multiple of 64 for CUDA).
M49M_6L384 = ModelConfig(
    vocab_size=50304,
    context_length=512,
    embedding_dim=384,
    num_layers=6,
    num_heads=6,
    dropout=0.0,
)

# Byte-level stepping stone (~10.9M): same 6L384 shape but keeps the
# ByteTokenizer (vocab 256). Train this FIRST so you don't
# have to build BPE + data pipeline in the same step.
S11M_6L384 = ModelConfig(
    vocab_size=256,
    context_length=256,
    embedding_dim=384,
    num_layers=6,
    num_heads=6,
    dropout=0.0,
)

# BPE era (~12.5M): 6L384 shape, vocab from checkpoints/bpe2k.json (2256).
# 256 ctx now holds ~4.5x more story than bytes. train.py overrides
# vocab_size from the file at runtime; this stays as the sane default.
S12M_6L384 = ModelConfig(
    vocab_size=2256,
    context_length=256,
    embedding_dim=384,
    num_layers=6,
    num_heads=6,
    dropout=0.0,
)

# Book era (~17.2M): 512 ctx for chapter-length dependencies (LibriSpeech books).
# Vocab overridden from the tokenizer file at runtime (bpe8k -> ~8256).
# Positional table doubles vs 256 ctx — negligible param delta (~0.1M).
S17M_6L384 = ModelConfig(
    vocab_size=8256,
    context_length=512,
    embedding_dim=384,
    num_layers=6,
    num_heads=6,
    dropout=0.1,
)

# M class: same shape, vocab/ctx follow the corpus (like the S era).
# d=640/10H = 64/head — clean attention math.
M50M_10L640 = ModelConfig(
    vocab_size=4512,      # overridden from bpe_owt4k file at runtime
    context_length=512,   # books need the longer view
    embedding_dim=640,
    num_layers=10,
    num_heads=10,
    dropout=0.1,
)

# 1024-context variant: same m50m trunk, doubled context window.
# Params barely change (+0.6M for positional embeddings).
# Tests whether longer context unlocks better generalization.
M50M_10L640_1K = ModelConfig(
    vocab_size=4512,      # overridden from bpe_owt16k at runtime
    context_length=1024,  # doubled from 512
    embedding_dim=640,
    num_layers=10,
    num_heads=10,
    dropout=0.1,
)

M52M_10L640 = ModelConfig(
    vocab_size=2256,      # overridden from bpe2k file at runtime
    context_length=256,   # stories are short — 256 holds plenty
    embedding_dim=640,
    num_layers=10,
    num_heads=10,
    dropout=0.1,          # keep — it earned its place in 006
)

M60M_10L640 = ModelConfig(
    vocab_size=8256,      # overridden from bpe8k file at runtime
    context_length=512,   # books need the longer view
    embedding_dim=640,
    num_layers=10,
    num_heads=10,
    dropout=0.1,
)

# Dense 194M: 3x the M class. 32 attention heads, d=1024.
# The "does bigger actually help on FineWeb?" test.
# ~4-5GB VRAM on 4070 Ti, ~3x slower than 50M.
L194M_14L1024 = ModelConfig(
    vocab_size=8256,      # overridden from bpe8k file at runtime
    context_length=512,
    embedding_dim=1024,
    num_layers=14,
    num_heads=32,         # 32 dims per head, clean
    dropout=0.1,
)

# 24-head variant (~112.2M): 768d / 24H (32 dims/head, same as 194M's head_dim).
# Fits batch 16 on 16GB where 194M doesn't.
L112M_14L768 = ModelConfig(
    vocab_size=8256,      # overridden from bpe8k file at runtime
    context_length=512,
    embedding_dim=768,
    num_layers=14,
    num_heads=24,         # 32 dims per head, clean
    dropout=0.1,
)

# Pythia-160M SHAPE: 12L x 768H x 12 heads x FFN 3072 (64 dims/head).
# Reference: EleutherAI/pythia-160m (spec + tokenizer in references/pythia-160m/).
# NOTE: matches the SHAPE for the capacity series (70.65M -> 113.86M -> 162.6M @ 50k),
# not the NeoX stack: we keep sequential residual, full RoPE, dropout 0.1 (see the
# reference README for the fidelity deltas). ~162.7M @ 50k vocab untied (real count,
# ctx 512), 110.3M @ 16k, 85.3M trunk (the axis that actually carries capacity at
# 50k vocab).
PYTHIA160_12L768 = ModelConfig(
    vocab_size=50256,     # overridden from bpe_owt50k_v2 file at runtime
    context_length=512,
    embedding_dim=768,
    num_layers=12,
    num_heads=12,
    dropout=0.1,
)

# Pythia-scaled: 6L × 512H × 8 heads × FFN 2048. ~52M @ 16k vocab.
# Reference: EleutherAI/pythia-70m uses the same arch (6/512/8/2048).
PYTHIA_6L512 = ModelConfig(
    vocab_size=16256,     # overridden from bpe_owt16k at runtime
    context_length=512,
    embedding_dim=512,
    num_layers=6,
    num_heads=8,
    dropout=0.1,
)

# M70 class, built for the Pythia-70M fight (~66M at 4k vocab).
# d=704/11H = 64/head, same clean math as the M50M. Depth over width:
# 10 layers carries further per token than 6 wide ones at this scale.
# Honest asymmetry vs EleutherAI/pythia-70m-deduped: theirs is ~50M
# embeddings (50k vocab) + ~19M transformer; ours is ~6M embeddings
# + ~60M transformer. Same weight class, more engine. STATE section 21+.
M66M_10L704 = ModelConfig(
    vocab_size=4512,      # overridden from bpe_owt4k file at runtime
    context_length=512,
    embedding_dim=704,
    num_layers=10,
    num_heads=11,
    dropout=0.1,          # keep — it earned its place in 006
)
