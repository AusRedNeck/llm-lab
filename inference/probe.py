import torch

from model.transformer import Transformer


# Keep this tiny so we can easily understand what the model is doing.
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

model.eval()

tokens = torch.tensor([[1, 2, 3, 4]])

with torch.no_grad():
    logits = model(tokens)

# We only care about the final position:
# it represents the model's prediction for the next token.
next_token_logits = logits[:, -1, :]

# Softmax converts arbitrary logits into probabilities.
# The probabilities across the vocabulary should sum to 1.
probabilities = torch.softmax(
    next_token_logits,
    dim=-1,
)

# Find the five most likely tokens.
top_probabilities, top_tokens = torch.topk(
    probabilities,
    k=5,
    dim=-1,
)

print("Top 5 token IDs:     ", top_tokens.tolist())
print("Top 5 probabilities: ", top_probabilities.tolist())
print("Probability sum:     ", probabilities.sum(dim=-1).tolist())

predicted_token = torch.argmax(
    next_token_logits,
    dim=-1,
)

print("Input tokens:       ", tokens.tolist())
print("Logits shape:        ", logits.shape)
print("Next-token logits:   ", next_token_logits.tolist())
print("Predicted token ID:  ", predicted_token.tolist())