# BPE on bytes: let the data pick the vocab.
# Bytes treat "princess" as 8 strangers. BPE fuses frequent pairs
# ("t"+"h" -> "th") until common words are single tokens.
# Base vocab = all 256 bytes, so ANY text encodes. No seed hacks.
import json
import re
from collections import Counter

# Splitter: GPT-2 style. Every char lands in exactly one chunk, so
# findall can never skip (old \s?\S+ silently ate extra spaces).
# Branches: contractions | letters | digits | other marks | whitespace.
_SPLIT = re.compile(
    r"'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+"
)


# One token that is never a word: documents end, then this, then next doc.
EOS = "<|endoftext|>"


def _latin(b: bytes) -> str:
    # Bytes -> JSON-safe string, one char per byte. Reversible via _unlatin.
    return b.decode("latin-1")


def _unlatin(s: str) -> bytes:
    return s.encode("latin-1")


class BPETokenizer:
    # Trained vocab: id -> byte piece, plus merge ranks that define encoding.
    def __init__(self, vocab: dict[int, bytes], merges: dict[tuple[bytes, bytes], int],
                 eos: bool = False):
        # eos=False keeps old vocabs byte-identical (no silent id drift).
        self.vocab = dict(vocab)
        self.merges = merges
        self.eos_id = None
        if eos:
            piece = EOS.encode("utf-8")
            ids = [i for i, s in self.vocab.items() if s == piece]
            self.eos_id = ids[0] if ids else len(self.vocab)
            self.vocab.setdefault(self.eos_id, piece)
        self.token_to_id = {s: i for i, s in self.vocab.items()}

    def encode(self, text: str) -> list[int]:
        # EOS splits first so merges never fuse across documents.
        if self.eos_id is not None and EOS in text:
            ids: list[int] = []
            for n, seg in enumerate(text.split(EOS)):
                if n:
                    ids.append(self.eos_id)
                ids.extend(self._encode_seg(seg))
            return ids
        return self._encode_seg(text)

    def _encode_seg(self, text: str) -> list[int]:
        # Per chunk: utf-8 bytes first, then fuse the cheapest-ranked pair.
        ids: list[int] = []
        for chunk in _SPLIT.findall(text):
            ids.extend(self._encode_chunk(chunk))
        return ids

    def _encode_chunk(self, chunk: str) -> list[int]:
        # One regex chunk in, token ids out.
        parts = [bytes([b]) for b in chunk.encode("utf-8")]
        while len(parts) > 1:
            # -1, never None: the value that goes into the slice below must
            # always be an int, so it cannot raise "slice indices must be
            # integers" no matter what the pair scan does.
            best, best_rank = -1, None
            for i in range(len(parts) - 1):
                rank = self.merges.get((parts[i], parts[i + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best, best_rank = i, rank
            if best < 0:
                break  # no known pair left -> keep bytes (always encodable)
            parts[best:best + 2] = [parts[best] + parts[best + 1]]
        return [self.token_to_id[p] for p in parts]

    def decode(self, ids: list[int]) -> str:
        # Pieces are bytes. Glue, then utf-8 (replace guards cut sequences).
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump({
                "vocab": {str(i): _latin(s) for i, s in self.vocab.items()},
                "merges": [[_latin(a), _latin(b)] for a, b in self.merges],
                "eos": self.eos_id is not None,
            }, f)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        # Merge rank = order in the saved list. Rank IS the encoding rule.
        # Old files have no "eos" key -> False -> vocab loads untouched.
        with open(path) as f:
            raw = json.load(f)
        vocab = {int(i): _unlatin(s) for i, s in raw["vocab"].items()}
        merges = {(_unlatin(a), _unlatin(b)): r for r, (a, b) in enumerate(raw["merges"])}
        return cls(vocab, merges, eos=raw.get("eos", False))


def train_bpe(texts: list[str], num_merges: int) -> BPETokenizer:
    # Count how words actually look (as bytes), fuse the top pair, repeat.
    freqs: Counter[tuple[bytes, ...]] = Counter()
    for text in texts:
        for chunk in _SPLIT.findall(text):
            freqs[tuple(bytes([b]) for b in chunk.encode("utf-8"))] += 1

    # All 256 bytes earn ids first. Encoding can never fail after this.
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    merges: dict[tuple[bytes, bytes], int] = {}

    for _ in range(num_merges):
        # Most frequent adjacent pair across the whole corpus wins.
        pairs: Counter[tuple[bytes, bytes]] = Counter()
        for word, n in freqs.items():
            for i in range(len(word) - 1):
                pairs[(word[i], word[i + 1])] += n
        if not pairs:
            break
        best = max(pairs, key=lambda p: pairs[p])
        merges[best] = len(merges)
        vocab[len(vocab)] = best[0] + best[1]
        # Rewrite every word containing the pair, so counts stay honest.
        fused: Counter[tuple[bytes, ...]] = Counter()
        for word, n in freqs.items():
            out, i = [], 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == best:
                    out.append(word[i] + word[i + 1])
                    i += 2
                else:
                    out.append(word[i])
                    i += 1
            fused[tuple(out)] += n
        freqs = fused

    return BPETokenizer(vocab, merges)
