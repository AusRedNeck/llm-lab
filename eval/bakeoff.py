# Rope vs no-rope head-to-head on held-out val.
# Usage: uv run python -m eval.bakeoff --val_batches 20

import argparse
import torch

from inference.generate import load_model
from train.train import load_tinystories, get_batch
from eval.benchmark import perplexity, token_accuracy

CKPTS = {
    "rope-1801": "checkpoints/exp002_bytes10m_rope_202609032333_step5000.pt",
    "norope-2313": "checkpoints/exp002_bytes10m_202609032313_step5000.pt",
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_batches", type=int, default=20)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available()
                          else ("mps" if torch.backends.mps.is_available() else "cpu"))
    corpus = load_tinystories(128, device)
    cut = int(len(corpus) * 0.99)
    val = corpus[cut:]

    for name, ckpt in CKPTS.items():
        model, c = load_model(ckpt, device)
        ctx = c["context_length"]
        ppls, accs = [], []
        with torch.no_grad():
            for _ in range(args.val_batches):
                x, y = get_batch(val, 64, ctx, c["vocab_size"],device, [0])
                logits = model(x)
                ppls.append(perplexity(logits, y))
                accs.append(token_accuracy(logits, y))
        print(f"{name}: ppl {sum(ppls)/len(ppls):.2f} | acc {sum(accs)/len(accs):.3f}")

if __name__ == "__main__":
    main()