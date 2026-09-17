#!/usr/bin/env python3
"""Fast BPE trainer — single flat array, numpy everywhere.

Key insight: flatten all chunks into ONE array with CHUNK_SEP markers.
Count pairs on the flat array (vectorized). Fuse by masking + compaction.

Usage:
    uv run python -m model.train_bpe_fast --data data/incoming/openwebtext_sample_1gb.txt \
        --lines 100000 --merges 4000 --out data/incoming/bpe_owt4k.json
"""
from __future__ import annotations
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.bpe import BPETokenizer, _SPLIT

SEP = -1  # chunk separator in flat array


def _load_sample(path: str, max_lines: int, stride: int) -> list[str]:
    import glob
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "**", "*.txt"), recursive=True))
        if not files:
            raise FileNotFoundError(f"no .txt files under {path}")
        texts: list[str] = []
        per_file_stride = max(1, stride // max(1, len(files) // 10))
        for fp in files:
            if len(texts) >= max_lines:
                break
            with open(fp, encoding="utf-8", errors="replace") as f:
                for i, line in enumerate(f):
                    if len(texts) >= max_lines:
                        break
                    if i % per_file_stride == 0:
                        line = line.strip()
                        if line:
                            texts.append(line)
        return texts
    texts = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if i % stride == 0:
                line = line.strip()
                if line:
                    texts.append(line)
                    if len(texts) >= max_lines:
                        break
    return texts


def _build_flat(texts: list[str]) -> np.ndarray:
    """Pretokenize and build flat array with SEP markers between chunks."""
    parts: list[np.ndarray] = []
    for text in texts:
        for match in _SPLIT.findall(text):
            b = match.encode("utf-8")
            arr = np.frombuffer(b, dtype=np.uint8).astype(np.int32)
            parts.append(arr)
            parts.append(np.array([SEP], dtype=np.int32))
    return np.concatenate(parts)


def train_bpe_fast(texts: list[str], num_merges: int,
                   log_every: int = 200) -> BPETokenizer:
    t0 = time.time()
    flat = _build_flat(texts)
    total_bytes = int((flat != SEP).sum())
    print(f"  flat array: {len(flat):,} entries, {total_bytes:,} bytes "
          f"({time.time() - t0:.1f}s)")

    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    merges: dict[tuple[bytes, bytes], int] = {}

    for merge_idx in range(num_merges):
        if merge_idx % log_every == 0:
            elapsed = time.time() - t0
            rate = (merge_idx / elapsed) if elapsed > 0 else 0
            print(f"  merge {merge_idx}/{num_merges} "
                  f"({rate:.1f} merges/s, {elapsed:.0f}s, "
                  f"vocab {len(vocab)}, flat {len(flat):,})", flush=True)

        # ── count pairs on flat array ──
        left = flat[:-1]
        right = flat[1:]
        valid = (left != SEP) & (right != SEP)
        pl = left[valid]
        pr = right[valid]
        if len(pl) == 0:
            break

        # Encode pair as int64 for unique/bincount
        # Max vocab ID at this point = 256 + merge_idx
        maxvid = 256 + merge_idx + 1
        pair_ids = pl.astype(np.int64) * maxvid + pr.astype(np.int64)
        uniq, counts = np.unique(pair_ids, return_counts=True)
        best_idx = int(np.argmax(counts))
        best_count = int(counts[best_idx])
        if best_count == 0:
            break
        best_pid = int(uniq[best_idx])
        a_id = best_pid // maxvid
        b_id = best_pid % maxvid

        # New vocab entry
        merged_bytes = vocab[a_id] + vocab[b_id]
        new_id = len(vocab)
        vocab[new_id] = merged_bytes
        merges[(vocab[a_id], vocab[b_id])] = merge_idx

        # ── fuse: mask + compact ──
        # Find all (a_id, b_id) pair positions in flat array
        fleft = flat[:-1]
        fright = flat[1:]
        pair_mask = (fleft == a_id) & (fright == b_id) & (fleft != SEP) & (fright != SEP)

        if not pair_mask.any():
            continue

        # For each pair position i: replace flat[i] = new_id, mark flat[i+1] = SEP
        # Then compact: remove all SEP positions
        pair_positions = np.where(pair_mask)[0]
        flat[pair_positions] = new_id
        # Mark right-side positions for deletion
        # (Need to be careful: a position might be right-side of one pair AND left-side of another)
        # Simple approach: mark all i+1 positions, then compact
        delete_positions = pair_positions + 1
        # Create keep mask (everything NOT in delete_positions AND NOT SEP)
        keep = np.ones(len(flat), dtype=bool)
        keep[delete_positions] = False
        # Also keep existing SEP markers (chunk boundaries)
        # Actually, we want to remove the old pair's right element, not SEP markers
        # The SEP at chunk boundaries should stay
        flat = flat[keep]

    elapsed = time.time() - t0
    final_bytes = int((flat != SEP).sum())
    compression = total_bytes / max(1, final_bytes)
    print(f"  done: vocab {len(vocab)}, compression {compression:.2f}x ({elapsed:.0f}s)")

    # ── convert flat array back to chunks for BPETokenizer ──
    # Split on SEP, store as vocab-indexed chunks (for encoding)
    # But BPETokenizer needs byte-based vocab, so just return it directly
    return BPETokenizer(vocab, merges, eos=False)


def main():
    ap = argparse.ArgumentParser(description="Fast BPE trainer (numpy)")
    ap.add_argument("--data", default="data/incoming/openwebtext_sample_1gb.txt")
    ap.add_argument("--lines", type=int, default=100000)
    ap.add_argument("--stride", type=int, default=88)
    ap.add_argument("--merges", type=int, default=4000)
    ap.add_argument("--out", default="data/incoming/bpe_owt4k.json")
    args = ap.parse_args()

    t0 = time.time()
    texts = _load_sample(args.data, args.lines, args.stride)
    mb = sum(len(t) for t in texts) / 1e6
    print(f"sample: {len(texts)} lines, {mb:.1f}MB ({time.time() - t0:.0f}s)")

    t0 = time.time()
    tok = train_bpe_fast(texts, args.merges)
    print(f"trained: vocab {len(tok.vocab)} in {time.time() - t0:.0f}s")

    probe = "Once upon a time there was a little princess."
    n = len(tok.encode(probe))
    print(f"probe: {n} tokens vs {len(probe.encode())} bytes: {probe!r}")

    tok.save(args.out)
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
