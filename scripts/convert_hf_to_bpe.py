#!/usr/bin/env python3
"""Convert an HF tokenizer.json to BPETokenizer-compatible .json format.

WHY THIS IS NOT A STRAIGHT COPY OF STRINGS
HF's ByteLevel BPE writes every token as `bytes_to_unicode()` escapes: byte 32
(space) is stored as 'Ġ' (U+0120), and the bytes that are whitespace/control get
mapped into U+0100+. Our BPETokenizer stores pieces as RAW BYTES wrapped in
latin-1 (model/bpe.py: `_latin`/`_unlatin`), so it can only hold code points
0-255. Copying HF strings verbatim produced a file that CRASHED the trainer at
load time:

    UnicodeEncodeError: 'latin-1' codec can't encode character '\u0120'
      -> model/bpe.py:98  BPETokenizer.load

Fix: invert HF's bytes_to_unicode map per character, then latin-1 the raw bytes.

EOS: the old version flagged the `[EOS]` special token (id 3) as the document
separator, which made BPETokenizer append a 50,257th token for '<|endoftext|>'
and shift vocab_size off 50,256. Only a literal '<|endoftext|>' piece in the
vocab counts as EOS now; anything else converts to False.
"""
import json
import os
import sys


def bytes_to_unicode() -> dict[int, str]:
    """HF/GPT-2 byte -> printable-unicode-char map (the one the vocab was saved in)."""
    bs = (list(range(ord("!"), ord("~") + 1))          # '!' .. '~'
          + list(range(ord("\u00a1"), ord("\u00ac") + 1))   # ¡ .. ¬
          + list(range(ord("\u00ae"), ord("\u00ff") + 1)))  # ® .. ÿ
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


UNI_TO_BYTE = {v: k for k, v in bytes_to_unicode().items()}


def to_raw(s: str) -> str:
    """HF-escaped token string -> raw bytes -> latin-1 str our BPETokenizer can load."""
    raw = bytearray()
    for ch in s:
        b = UNI_TO_BYTE.get(ch)
        if b is None:
            # Never guess: a char outside the map means this is not a ByteLevel
            # vocab, and silently keeping it would rebuild the original crash.
            raise ValueError(f"character {ch!r} (U+{ord(ch):04X}) is not in "
                             f"bytes_to_unicode -- wrong tokenizer type?")
        raw.append(b)
    return bytes(raw).decode("latin-1")


def convert(hf_path: str, out_path: str):
    with open(hf_path, encoding="utf-8") as f:
        hf = json.load(f)
    model = hf["model"]

    # HF vocab: escaped-string -> int. Ours: int -> latin-1 raw-byte string.
    bpe_vocab = {int(i): to_raw(tok) for tok, i in model["vocab"].items()}

    # HF merges: "a b" (gpt-neox string form) or ["a", "b"] -> [[raw, raw], ...].
    bpe_merges = []
    for m in model.get("merges", []):
        parts = m.split() if isinstance(m, str) else m
        if len(parts) != 2:
            raise ValueError(f"merge is not a pair: {m!r}")
        bpe_merges.append([to_raw(a) for a in parts])

    # EOS only when the literal end-of-text piece exists in the vocab; bpe.py
    # then finds its real id instead of inventing one past the end of the vocab.
    eos = any(v == "<|endoftext|>".encode("utf-8").decode("latin-1")
              for v in bpe_vocab.values())

    with open(out_path, "w") as f:
        json.dump({"vocab": {str(i): s for i, s in bpe_vocab.items()},
                   "merges": bpe_merges, "eos": eos}, f, indent=2)

    print(f"Converted {os.path.basename(hf_path)} -> {os.path.basename(out_path)}")
    print(f"  Vocab: {len(bpe_vocab):,} tokens (ids {min(bpe_vocab)}..{max(bpe_vocab)})")
    print(f"  Merges: {len(bpe_merges):,}")
    print(f"  EOS: {eos}")
    return out_path


if __name__ == "__main__":
    hf_path = sys.argv[1] if len(sys.argv) > 1 else "references/pythia-50k-owt/tokenizer.json"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "checkpoints/bpe_owt50k.json"
    convert(hf_path, out_path)
