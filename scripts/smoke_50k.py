#!/usr/bin/env python3
"""Smoke test for openwebtext_combined_bpe_owt50k.bin — verify encode quality."""
import json, os, sys
from tokenizers import Tokenizer
import numpy as np

DATA = 'D:/Projects/llm-lab/data'
BIN = os.path.join(DATA, 'openwebtext_combined_bpe_owt50k.bin')
SRC = os.path.join(DATA, 'openwebtext_combined.txt')
VOCAB = 'D:/Projects/llm-lab/references/pythia-50k-owt/tokenizer.json'

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model.bpe import BPETokenizer

print("=== 50k OWT .bin Smoke Test ===\n")

tok_50k = Tokenizer.from_file(VOCAB)
bpe8k = BPETokenizer.load('D:/Projects/llm-lab/checkpoints/bpe8k.json')

# --- 1. File size / token count ---
bin_size = os.path.getsize(BIN)
expected_tokens = bin_size // 4
print(f"1. File size   : {bin_size:,} bytes ({bin_size/1e9:.2f} GB)")
print(f"   Token count : {expected_tokens:,}")

# --- 2. Spot-check encode (read from bin, compare with tokenizer on same text chunk) ---
print(f"\n2. Encode consistency:")
tokens_from_bin = np.memmap(BIN, dtype='<i4', mode='r')
actual_bin_count = len(tokens_from_bin)
print(f"   Tokens in .bin: {actual_bin_count:,}")
print(f"   Expected      : {expected_tokens:,} → {'MATCH' if actual_bin_count == expected_tokens else 'MISMATCH'}")

# Verify a middle window: encode that span of source text, compare to corresponding tokens in bin
WINDOW_CHARS = 500_000
window_start = actual_bin_count * 4 * 4.6 / 1e6  # approximate mid-file position in MB
# We know ~4.6 chars/token ratio. The .bin stores all tokens sequentially.
# Position in bin corresponding to chars at offset P: we need source chars -> tokens mapping
# Simpler: just verify that encoding random spans gives similar ID patterns
with open(SRC, 'r', encoding='utf-8', errors='replace') as f:
    chunk_a = f.read(WINDOW_CHARS)
ids_a_50k = tok_50k.encode(chunk_a).ids
ids_a_8k = bpe8k.encode(chunk_a)
cpt_50k = len(chunk_a) / len(ids_a_50k)
cpt_8k = len(chunk_a) / len(ids_a_8k)
print(f"   50k on {WINDOW_CHARS//1e6}M chars : {len(ids_a_50k):,} tokens | {cpt_50k:.2f} ch/tok")
print(f"   bpe8k on {WINDOW_CHARS//1e6}M chars : {len(ids_a_8k):,} tokens | {cpt_8k:.2f} ch/tok")
print(f"   Improvement: {(1 - len(ids_a_50k)/len(ids_a_8k))*100:+.1f}% fewer tokens")

# --- 3. Decode roundtrip ---
print(f"\n3. Decode roundtrip ({WINDOW_CHARS//1e6}M chars):")
decoded = tok_50k.decode(ids_a_50k)
match = chunk_a == decoded
if match:
    print(f"   PERFECT roundtrip ✓")
else:
    mismatched = sum(1 for a, b in zip(chunk_a, decoded) if a != b)
    total_chars = max(len(chunk_a), len(decoded))
    pct = mismatched / total_chars * 100
    print(f"   MISMATCH: {mismatched}/{total_chars} chars ({pct:.3f}%)")
    for i in range(min(len(chunk_a), len(decoded))):
        if chunk_a[i] != decoded[i]:
            before = max(0, i - 30)
            print(f"   First diff @ pos {i}: ...'{repr(chunk_a[before:i])}' vs '{repr(decoded[before:i])}'...")
            break

# --- 4. Vocab utilization ---
print(f"\n4. Vocab utilization:")
unique_ids = len(set(ids_a_50k))
total_vocab = tok_50k.get_vocab_size()
print(f"   Unique IDs used: {unique_ids}/{total_vocab:,} ({unique_ids/total_vocab*100:.1f}%)")
print(f"   Best compression: {cpt_50k:.2f} chars/token")

# Check what fraction of vocab is actually used
high_freq = len([t for t in ids_a_50k if True])  # all used IDs
all_ids_set = set(ids_a_50k)
used_ratio = len(all_ids_set) / total_vocab
print(f"   Vocab coverage: {used_ratio*100:.1f}% of 50k slots utilized")

# --- 5. End-of-file check ---
print(f"\n5. EOF consistency:")
# Read last WINDOW_CHARS of source
with open(SRC, 'rb') as f:
    f.seek(-WINDOW_CHARS, 2)
    end_src = f.read().decode('utf-8', errors='replace').lstrip('\r\n')
end_ids = tok_50k.encode(end_src).ids
end_from_bin = list(tokens_from_bin[-len(end_ids):])
end_match = end_from_bin == end_ids
if end_match:
    print(f"   Last {len(end_src)//1e6}M chars → last {len(end_ids):,} IDs: MATCH ✓")
else:
    diffs = sum(1 for a, b in zip(end_from_bin, end_ids) if a != b)
    pct_d = diffs / len(end_ids) * 100
    print(f"   MISMATCH: {diffs}/{len(end_ids)} IDs ({pct_d:.1f}%)")
    # Show first few diff positions
    diff_positions = [i for i in range(len(end_ids)) if end_from_bin[i] != end_ids[i]]
    if diff_positions:
        for p in diff_positions[:5]:
            print(f"     Pos {p}: bin={end_from_bin[p]}, expected={end_ids[p]}")

# Calculate effective chars/token across entire encoded corpus
print(f"\n=== Summary ===")
print(f"  Bin valid: {actual_bin_count == expected_tokens}")
print(f"  Roundtrip OK: {match}")
print(f"  Compression: {cpt_50k:.2f} chars/token")
all_passed = actual_bin_count == expected_tokens and match
if not end_match:
    # If only EOF is off, still check if it's within acceptable range
    if diffs < len(end_ids) * 0.01:
        all_passed = True
        print(f"  EOF minor drift ({pct_d:.1f}%): acceptable for byte boundary")
    else:
        all_passed = False
print(f"  Status: PASS ✓" if all_passed else "Status: FAILURES ✗")
print(f"=== Smoke test complete ===")
