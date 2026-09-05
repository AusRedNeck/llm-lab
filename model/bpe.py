# BPE on bytes: let the data pick the vocab.
# Bytes treat "princess" as 8 strangers. BPE fuses frequent pairs
# ("t"+"h" -> "th") until common words are single tokens.
# Base vocab = all 256 bytes, so ANY text encodes. No seed hacks.
import json
import re
from collections import Counter

# Split that keeps spaces attached: "a b" -> ["a", " b"]. Spaces matter.
_SPLIT = re.compile(r"\s?\S+")


def _latin(b: bytes) -> str:
    # Bytes -> JSON-safe string, one char per byte. Reversible via _unlatin.
    return b.decode("latin-1")


def _unlatin(s: str) -> bytes:
    return s.encode("latin-1")


class BPETokenizer:
    # Trained vocab: id -> byte piece, plus merge ranks that define encoding.
    def __init__(self, vocab: dict[int, bytes], merges: dict[tuple[bytes, bytes], int]):
        self.vocab = vocab
        self.merges = merges
        self.token_to_id = {s: i for i, s in vocab.items()}

    def encode(self, text: str) -> list[int]:
        # Per chunk: utf-8 bytes first, then fuse the cheapest-ranked pair.
        ids: list[int] = []
        for chunk in _SPLIT.findall(text):
            parts = [bytes([b]) for b in chunk.encode("utf-8")]
            while len(parts) > 1:
                best, best_rank = None, None
                for i in range(len(parts) - 1):
                    rank = self.merges.get((parts[i], parts[i + 1]))
                    if rank is not None and (best_rank is None or rank < best_rank):
                        best, best_rank = i, rank
                if best is None:
                    break  # no known pair left -> keep bytes (always encodable)
                parts[best:best + 2] = [parts[best] + parts[best + 1]]
            ids.extend(self.token_to_id[p] for p in parts)
        return ids

    def decode(self, ids: list[int]) -> str:
        # Pieces are bytes. Glue, then utf-8 (replace guards cut sequences).
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump({
                "vocab": {str(i): _latin(s) for i, s in self.vocab.items()},
                "merges": [[_latin(a), _latin(b)] for a, b in self.merges],
            }, f)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        # Merge rank = order in the saved list. Rank IS the encoding rule.
        with open(path) as f:
            raw = json.load(f)
        vocab = {int(i): _unlatin(s) for i, s in raw["vocab"].items()}
        merges = {(_unlatin(a), _unlatin(b)): r for r, (a, b) in enumerate(raw["merges"])}
        return cls(vocab, merges)


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
