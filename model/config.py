class ModelConfig:
    """Configuration for our small experimental language model."""
    vocab_size: int = 256
    context_length: int = 128
    embedding_dim: int = 128
    num_layers: int = 4
    num_heads: int = 4
    dropout: float = 0.0