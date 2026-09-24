#!/usr/bin/env python3
"""Re-encode the 1GB OpenWebText sample with the Gen1+Gen2 tokenizer.

Why: the existing .pt is a pre-Gen1 Ronin cache (flattened whitespace).
This rebuilds it with the lossless splitter + bpe_owt8k vocab (eos=True).
Run: uv run python reencode_owt1gb.py
"""
import time
import numpy as np
import torch

from model.bpe import BPETokenizer

# Where things live.
SRC = "data/incoming/openwebtext_sample_1gb.txt"
VOCAB = "data/incoming/bpe_owt8k.json"
DST = "data/incoming/openwebtext_sample_1gb_bpe8k.pt"
CHUNK = 500_000  # chars per encode call — keeps RAM flat


def main():
    # Load the fresh vocab (8257 ids, eos on).
    tok = BPETokenizer.load(VOCAB)

    # Stream the 1GB txt, encode chunk by chunk, stash int32 arrays.
    parts: list[np.ndarray] = []
    total = 0
    t0 = time.time()
    with open(SRC, encoding="utf-8", errors="replace") as f:
        while True:
            text = f.read(CHUNK)
            if not text:
                break
            ids = tok.encode(text)
            parts.append(np.array(ids, dtype=np.int32))
            total += len(ids)
            dt = time.time() - t0
            print(f"  {total:,} toks ({total / dt:,.0f} tok/s)", flush=True)

    # One concat, one torch save — same format as before (int32 tensor).
    all_ids = np.concatenate(parts)
    torch.save(torch.from_numpy(all_ids), DST)
    dt = time.time() - t0
    print(f"Done: {total:,} toks in {dt:.0f}s -> {DST}")

    # Spot check: whitespace survives the round trip now.
    probe = "a  b\n\n  indent"
    assert tok.decode(tok.encode(probe)) == probe, "whitespace round-trip broke!"
    print("Round-trip probe OK:", repr(probe))


if __name__ == "__main__":
    main()
