# Train a fresh BPE tokenizer on OpenWebText.
# Usage: python train_tokenizer_openwebtext.py [--sample_mb 50] [--merges 8000]
# The old bpe8k was trained on Gutenberg (19th century). This one learns
# modern internet vocabulary — better encoding for Reddit/web text.
import argparse
import time
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.dirname(os.path.abspath(__file__))
for path in (ROOT, SCRIPTS):
    if path not in sys.path:
        sys.path.insert(0, path)

from model.bpe import train_bpe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/openwebtext_combined.txt",
                    help="Source text file")
    ap.add_argument("--sample_mb", type=int, default=50,
                    help="MB of text to sample for training (50MB captures vocab well)")
    ap.add_argument("--merges", type=int, default=8000,
                    help="BPE merge operations (vocab = 256 + merges)")
    ap.add_argument("--output", default=None,
                    help="Output path (default: checkpoints/bpe_owt_<merges>.json)")
    args = ap.parse_args()

    output = args.output or f"checkpoints/bpe_owt_{args.merges}.json"

    # Read a sample from the start of the file.
    # 50MB is plenty to learn vocabulary distribution — BPE doesn't need the whole corpus.
    target_bytes = args.sample_mb * 1024 * 1024
    print(f"Reading {args.sample_mb}MB from {args.input}...")
    t0 = time.time()
    with open(args.input, "r", encoding="utf-8", errors="replace") as f:
        text = f.read(target_bytes)
    read_time = time.time() - t0
    print(f"  Read {len(text)/1e6:.1f}MB in {read_time:.1f}s")

    # Split into lines for train_bpe (it expects a list of strings).
    lines = text.split("\n")
    print(f"  {len(lines):,} lines")

    # Train BPE — this is the slow part.
    print(f"Training BPE with {args.merges} merges on {len(lines):,} lines...")
    t0 = time.time()
    tok = train_bpe(lines, args.merges)
    train_time = time.time() - t0
    print(f"  Trained in {train_time:.1f}s")
    print(f"  Vocab size: {len(tok.vocab)} (256 base + {args.merges} merges)")

    # Save
    tok.save(output)
    print(f"  Saved to {output}")

    # Benchmark: encode the same text and compare efficiency.
    t0 = time.time()
    ids = tok.encode(text)
    encode_time = time.time() - t0
    chars_per_token = len(text) / len(ids)

    # Also check vocab utilization.
    unique_ids = len(set(ids))

    print(f"\n=== Benchmark on {len(text)/1e6:.1f}MB ===")
    print(f"  Tokens: {len(ids):,}")
    print(f"  Chars/token: {chars_per_token:.2f}")
    print(f"  Vocab used: {unique_ids}/{len(tok.vocab)} ({unique_ids/len(tok.vocab)*100:.0f}%)")
    print(f"  Encode time: {encode_time:.1f}s")

    # Compare against old bpe8k
    old_path = "checkpoints/bpe8k.json"
    if os.path.exists(old_path):
        from model.bpe import BPETokenizer
        old_tok = BPETokenizer.load(old_path)
        old_ids = old_tok.encode(text)
        old_cpt = len(text) / len(old_ids)
        old_unique = len(set(old_ids))
        print(f"\n=== vs old bpe8k (Gutenberg-trained) ===")
        print(f"  Old tokens: {len(old_ids):,}")
        print(f"  Old chars/token: {old_cpt:.2f}")
        print(f"  Old vocab used: {old_unique}/{len(old_tok.vocab)} ({old_unique/len(old_tok.vocab)*100:.0f}%)")
        print(f"  Improvement: {(1 - len(ids)/len(old_ids))*100:.1f}% fewer tokens")
        print(f"  Training speedup: ~{(len(old_ids)/len(ids) - 1)*100:.1f}% more tokens/step")


if __name__ == "__main__":
    main()
