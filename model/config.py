# Model sizes: pick your fight. Toy proves the wiring, bytes10m learns to talk.
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


# The toy you have now (~0.8M params, byte-level). Good for proving
# loss goes down, not for real English.
TOY_1M = ModelConfig()

# First real training target (~10M params).
# Trains in hours on the 4070 Ti SUPER, actually talks.
# vocab 50304 = GPT-2 BPE size (padded to multiple of 64 for CUDA).
# If you stay byte-level (256) for now, swap vocab_size back to 256
# and the count drops to ~7M — fine for experiment 002a.
TINY_10M = ModelConfig(
    vocab_size=50304,
    context_length=512,
    embedding_dim=384,
    num_layers=6,
    num_heads=6,
    dropout=0.0,
)

# Byte-level stepping stone: same shape as TINY_10M but keeps your
# current ByteTokenizer (vocab 256). Train this FIRST so you don't
# have to build BPE + data pipeline in the same step.
TINY_10M_BYTES = ModelConfig(
    vocab_size=256,
    context_length=256,
    embedding_dim=384,
    num_layers=6,
    num_heads=6,
    dropout=0.0,
)

# BPE era: same 10M shape, vocab from checkpoints/bpe2k.json (2256).
# 256 ctx now holds ~4.5x more story than bytes. train.py overrides
# vocab_size from the file at runtime; this stays as the sane default.
BPE2K_10M = ModelConfig(
    vocab_size=2256,
    context_length=256,
    embedding_dim=384,
    num_layers=6,
    num_heads=6,
    dropout=0.0,
)
