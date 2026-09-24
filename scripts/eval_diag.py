"""Train-eval diagnostic: run eval_loss on BOTH train and val splits
for both the 512-control and 1K-experiment checkpoints.

Usage: python -m eval_diag --ckpt1k <path> --ckpt512 <path> --corpus <tok.bin> --tokenizer <vocab.json>
"""
import argparse, json, os, sys
import numpy as _np
import torch
import torch.nn.functional as F
from model.config import M50M_10L640, M50M_10L640_1K

def load_bin(path):
    n = os.path.getsize(path) // 4
    return torch.from_numpy(_np.memmap(path, dtype="<i4", mode="r", shape=(n,)))

def split_corpus(corpus, val_frac=0.05, ctx=512):
    cut = int(len(corpus) * (1.0 - val_frac))
    train, val = corpus[:cut], corpus[cut:]
    print(f"  corpus {len(corpus):,} tokens -> train {len(train):,} / val {len(val):,}")
    return train, val

def get_batch(source, batch, ctx, vocab, device, pos=None):
    idx = torch.randint(0, len(source) - ctx - 1, (batch,)).tolist()
    x = torch.stack([source[i:i + ctx] for i in idx]).long().to(device)
    y = torch.stack([source[i + 1:i + ctx + 1] for i in idx]).long().to(device)
    return x, y

def eval_loss(model, data, batch, ctx, vocab, device, batches=10):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for _ in range(batches):
            x, y = get_batch(data, batch, ctx, vocab, device)
            logits = model(x)
            total += F.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1)).item()
    model.train()
    return total / batches

def load_model(ckpt_path, cfg, device):
    from model.transformer import Transformer
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    # State dict may have 'model' key or be flat
    if "model" in state:
        state = state["model"]
    model = Transformer(
        vocab_size=cfg.vocab_size,
        context_length=cfg.context_length,
        embedding_dim=cfg.embedding_dim,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        use_rope=True,
        dropout=cfg.dropout,
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  WARNING missing keys: {missing[:5]}")
    model = model.to(device)
    model.eval()
    return model

def bpb_from_nats(nats, bpb_factor):
    return nats * bpb_factor

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt1k", required=True, help="Path to 1K checkpoint (best or step)")
    ap.add_argument("--ckpt512", required=True, help="Path to 512 checkpoint (best)")
    ap.add_argument("--corpus", required=True, help="Path to tokenized .bin")
    ap.add_argument("--tokenizer", required=True, help="Path to vocab.json")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--batches", type=int, default=20, help="Eval batches per split (20*batch = sequences)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    # Load corpus
    print("Loading corpus...")
    corpus = load_bin(args.corpus)

    # Load tokenizer for vocab size
    with open(args.tokenizer) as f:
        tok = json.load(f)
    vocab = len(tok.get("vocab", tok))
    print(f"  vocab: {vocab}")

    # Load models
    print(f"\nLoading 1K model from {args.ckpt1k}...")
    cfg1k = M50M_10L640_1K
    cfg1k.vocab_size = vocab
    model_1k = load_model(args.ckpt1k, cfg1k, device)

    print(f"Loading 512 model from {args.ckpt512}...")
    cfg512 = M50M_10L640
    cfg512.vocab_size = vocab
    model_512 = load_model(args.ckpt512, cfg512, device)

    # Val normalisation factor (from train.py)
    bpb_factor = 4.0419 * 0.3569  # val_norm * log2(e) factor... wait, let me recalc
    # Actually from the log: "val normalisation: 4.0419 bytes/token -> bpb = val * 0.3569"
    # This means bpb = val_nats * 0.3569? No... let me check.
    # Actually bpb = val_ce / bytes_per_token * log2(e)
    # bytes_per_token = total_bytes / total_tokens
    # The train.py uses: bpb_factor = bytes_per_token * log2(e)
    # From the log: "bpb = val * 0.3569" so bpb_factor = 0.3569
    bpb_factor = 0.3569

    # For 512 ctx: split at ctx=512
    # For 1K ctx: split at ctx=1024
    # BUT the split is done on the flat corpus, so the cut point differs
    for model, label, ctx in [
        (model_1k, "1K", cfg1k.context_length),
        (model_512, "512", cfg512.context_length),
    ]:
        print(f"\n{'='*60}")
        print(f"  {label} model (ctx={ctx}) on both train and val splits")
        print(f"{'='*60}")

        train_data, val_data = split_corpus(corpus, val_frac=0.05, ctx=ctx)

        print(f"\n  Running eval on TRAIN split ({args.batches} batches x {args.batch} seqs)...")
        train_loss = eval_loss(model, train_data, args.batch, ctx, vocab, device, args.batches)
        train_bpb = bpb_from_nats(train_loss, bpb_factor)
        print(f"  train loss: {train_loss:.4f} nats  |  bpb: {train_bpb:.5f}")

        print(f"\n  Running eval on VAL split ({args.batches} batches x {args.batch} seqs)...")
        val_loss = eval_loss(model, val_data, args.batch, ctx, vocab, device, args.batches)
        val_bpb = bpb_from_nats(val_loss, bpb_factor)
        print(f"  val loss:   {val_loss:.4f} nats  |  bpb: {val_bpb:.5f}")

        gap = train_loss - val_loss
        print(f"\n  gap (train - val): {gap:.4f} nats")
        print(f"  gap in bpb:        {train_bpb - val_bpb:.5f}")

        # Also check: how does eval on train HEAD (first 5%) compare to train TAIL?
        head_end = len(train_data) // 20
        print(f"\n  Checking train HEAD vs TAIL (first {head_end:,} vs last {head_end:,} tokens)...")
        head_loss = eval_loss(model, train_data[:head_end], args.batch, ctx, vocab, device, 5)
        tail_loss = eval_loss(model, train_data[-head_end*2:], args.batch, ctx, vocab, device, 5)
        print(f"  train HEAD loss: {head_loss:.4f}  |  train TAIL loss: {tail_loss:.4f}")
