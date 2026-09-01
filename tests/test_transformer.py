import torch

from model.transformer import TokenEmbedding, PositionalEmbedding, InputEmbedding, Transformer

def test_embedding_output_shape():
    embedding = TokenEmbedding(vocab_size=256, embedding_dim=128)

    tokens = torch.tensor([[1, 2, 3, 4]])

    output = embedding(tokens)

    assert output.shape == (1, 4, 128)

def test_positional_embedding_output_shape():
    embedding = PositionalEmbedding(context_length=128, embedding_dim=128)

    positions = torch.arange(4)

    output = embedding(positions)

    assert output.shape == (4, 128)

def test_input_embedding_output_shape():
    embedding = InputEmbedding(
        vocab_size=256,
        context_length=128,
        embedding_dim=128,
    )

    tokens = torch.tensor([[1, 2, 3, 4]])

    output = embedding(tokens)

    assert output.shape == (1, 4, 128)

def test_transformer_returns_logits():
    vocab_size = 32
    context_length = 8
    embedding_dim = 16
    num_heads = 4
    num_layers = 2

    model = Transformer(
        vocab_size=vocab_size,
        context_length=context_length,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        num_layers=num_layers,
    )

    tokens = torch.randint(
        0,
        vocab_size,
        (2, context_length),
    )

    logits = model(tokens)

    assert logits.shape == (
        2,
        context_length,
        vocab_size,
    )

def test_transformer_parameter_count():
    model = Transformer(
        vocab_size=32,
        context_length=8,
        embedding_dim=16,
        num_heads=4,
        num_layers=2,
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(f"\nParameter count: {parameter_count}")

    assert parameter_count > 0


def test_transformer_can_generate_logits():
    model = Transformer(
        vocab_size=32,
        context_length=8,
        embedding_dim=16,
        num_heads=4,
        num_layers=2,
    )

    tokens = torch.tensor([[1, 2, 3, 4]])

    logits = model(tokens)

    # We get one prediction vector for every input position.
    assert logits.shape == (1, 4, 32)

    # The final position is our prediction for what comes next.
    next_token_logits = logits[:, -1, :]

    assert next_token_logits.shape == (1, 32)