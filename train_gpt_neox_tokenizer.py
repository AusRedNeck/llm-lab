#!/usr/bin/env python3
"""Train a GPT-NeoX-style BPE tokenizer on OpenWebText with configurable vocab size.

Uses the HuggingFace `tokenizers` library directly (not the homegrown BPE).
Outputs in the same format gpt-neox / transformers expect:
    - tokenizer.json      (full tokenizer spec)
    - tokenizer_config.json (config metadata)

Usage:
    python train_gpt_neox_tokenizer.py [--sample_mb 100] [--vocab_size 50256]
    python train_gpt_neox_tokenizer.py --vocab_size 50256 --sample_mb 200

Vocab size = base bytes (256) + merges. So 50256 = 50k merges (GPT-NeoX default),
or 49408 for exact GPT-2 size (49152 merges).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
DATA_FILE = SCRIPT_DIR / "data" / "openwebtext_combined.txt"  # ~40 GB
OUTPUT_DIR = SCRIPT_DIR / "references" / "pythia-50k-owt"       # separate dir
BASE_VOCAB = 256  # byte-level fallback


def read_sample_lines(path: Path, sample_mb: int):
    """Read *sample_mb* MB of text and yield non-empty lines."""
    target_bytes = sample_mb * 1024 * 1024
    count = 0
    chars_read = 0
    with open(path, "rb") as bf:
        while chars_read < target_bytes:
            raw_line = bf.readline()
            if not raw_line:
                break
            decoded = raw_line.decode("utf-8", errors="replace").strip()
            if decoded:
                yield decoded
                count += 1
            chars_read += len(raw_line)


def main():
    ap = argparse.ArgumentParser(
        description="Train a GPT-NeoX BPE tokenizer on OpenWebText."
    )
    ap.add_argument("--input", default=str(DATA_FILE), help="Source text file")
    ap.add_argument("--sample_mb", type=int, default=100,
                    help="MB to sample from source (default 100)")
    ap.add_argument("--vocab_size", type=int, default=50256,
                    help="Total vocab size (base=256 + merges; 50256 = 50k merges)")
    ap.add_argument("--output-dir", default=str(OUTPUT_DIR),
                    help="Output directory")
    args = ap.parse_args()

    n_merges = args.vocab_size - BASE_VOCAB
    print(f"=== GPT-NeoX BPE Tokenizer Training ===")
    print(f"  Input     : {os.path.basename(args.input)} ({args.sample_mb} MB sample)")
    print(f"  Vocab     : {args.vocab_size} (base={BASE_VOCAB} + {n_merges:,} merges)")
    print(f"  Output    : {args.output_dir}")
    print()

    # --- Read sample -------------------------------------------------------
    t0 = time.time()
    lines = list(read_sample_lines(Path(args.input), args.sample_mb))
    elapsed = time.time() - t0
    total_chars = sum(len(l) for l in lines)
    print(f"Read {len(lines):,} lines ({total_chars / 1e6:.1f}M chars) in {elapsed:.1f}s\n")

    # --- Train -------------------------------------------------------------
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
    from tokenizers.models import BPE

    # Build the tokenizer
    tokenizer = Tokenizer(BPE(unknown_token="[UNK]"))

    # Byte-level pre-tokenizer splits into characters (byte fallback)
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # Decoder puts bytes back together
    tokenizer.decoder = decoders.ByteLevel()

    # Post-processor — none needed for raw id sequences
    tokenizer.post_processor = None

    # Trainer — special tokens must include the 256 byte-level specials
    special_tokens = ["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[SEP]", "[MASK]"]

    trainer = trainers.BpeTrainer(
        vocab_size=n_merges + BASE_VOCAB,   # total vocabulary size
        min_frequency=2,                     # skip rare pairs
        show_progress=True,
        special_tokens=special_tokens,
    )

    print(f"Training BPE ({n_merges:,} merges, min_freq=2)...\n")
    t0 = time.time()
    tokenizer.train_from_iterator(lines, trainer=trainer)
    train_time = time.time() - t0
    actual_vocab = len(tokenizer.get_vocab())
    print(f"\nTrained in {train_time:.1f}s")
    print(f"  Actual vocab size: {actual_vocab:,} (requested {args.vocab_size})")

    # Check how many merge rules were learned
    model = tokenizer.model  # type: models.BPE
    n_actual_merges = len(model.get_vocab()) - BASE_VOCAB if hasattr(model, 'get_vocab') else 0
    # Alternative: check via internals
    if hasattr(model, '_merge_rules'):
        n_learned = len(model._merge_rules)
        print(f"  Merge rules learned: {n_learned:,}")

    # --- Save --------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)

    tok_json = os.path.join(args.output_dir, "tokenizer.json")
    config_json = os.path.join(args.output_dir, "tokenizer_config.json")

    tokenizer.save(tok_json)

    config = {
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": "[UNK]",
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": False,
        },
        "pre_tokenizer": {"type": "ByteLevel"},
        "decoder": {"type": "ByteLevel"},
        "post_processor": None,
        "added_tokens": {
            str(i): {
                "id": i,
                "content": chr(i) if i < 256 else f"[special_{i}]",
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            }
            for i in range(min(actual_vocab, 512))  # just a few samples
        },
    }

    with open(config_json, "w") as f:
        json.dump({
            "name_or_path": "gpt-neox-owt-bpe-vocab",
            "model_max_length": 2048,
            "model_type": "gpt_neox",
            "tokenizer_class": "GPTNeoXTokenizerFast",
        }, f, indent=2)

    print(f"\nSaved:")
    print(f"  {tok_json} ({os.path.getsize(tok_json) / 1e6:.1f} MB)")
    print(f"  {config_json}")

    # --- Quick benchmark ---------------------------------------------------
    print(f"\n=== Benchmark on {total_chars/1e6:.1f}M chars ===")
    test_text = "\n".join(lines[:5000])

    t_enc = time.time()
    enc_ids = tokenizer.encode(test_text).ids
    enc_time = time.time() - t_enc

    # Also check old bpe8k for comparison
    old_path = SCRIPT_DIR / "checkpoints" / "bpe8k.json"
    if os.path.exists(old_path):
        from model.bpe import BPETokenizer as OldBPE
        old_tok = OldBPE.load(old_path)
        old_ids = old_tok.encode(test_text)
        old_cpt = len(test_text) / len(old_ids)

    new_cpt = len(test_text) / len(enc_ids)
    print(f"  {args.vocab_size}-tokenizer: {len(enc_ids):,} tokens | {new_cpt:.2f} chars/token | {enc_time*1000/len(test_text):.1f} ms/Mchar")
    if os.path.exists(old_path):
        old_cpt_val = len(test_text) / len(old_ids)
        print(f"  bpe8k (old):              {len(old_ids):,} tokens | {old_cpt_val:.2f} chars/token")
        print(f"  Improvement: {((1/new_cpt)-(1/old_cpt_val))/(1/old_cpt_val)*100:.1f}% fewer tokens/sample")
        print(f"  Savings vs bpe8k: {(len(old_ids)-len(enc_ids))/len(old_ids)*100:.1f}% fewer IDs")

    # Vocab utilization check
    unique_used = len(set(enc_ids))
    print(f"  Unique IDs used: {unique_used}/{actual_vocab:,} ({unique_used/actual_vocab*100:.1f}%)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
