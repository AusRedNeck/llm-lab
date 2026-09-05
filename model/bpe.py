# BPE: let the data pick the vocab.
# Bytes treat "princess" as 8 strangers. BPE fuses frequent pairs
# ("t"+"h" -> "th") until common words are single tokens.
import json
import re
import string
from collections import Counter

# Split that keeps spaces attached: "a b" -> ["a", " b"]. Spaces matter.
_SPLIT = re.compile(r"\s?\S+")

# Seeded alphabet: every printable ASCII earns an id even if training
# text never shows it. Unseen letters still encode; merges learn the rest.


class BPETokenizer:
    # Trained vocab: id -> piece, plus merge ranks that define encoding.
    def __init__(self, vocab: dict[int, str], merges: dict[tuple[str, str], int]):
        self.vocab = vocab
        self.merges = merges
        self.token_to_id = {s: i for i, s in vocab.items()}

    def encode(self, text: str) -> list[int]:
        # Per chunk: start from chars, fuse the cheapest-ranked pair first.
        ids: list[int] = []
        for chunk in _SPLIT.findall(text):
            parts = list(chunk)
            while len(parts) > 1:
                best, best_rank = None, None
                for i in range(len(parts) - 1):
                    rank = self.merges.get((parts[i], parts[i + 1]))
                    if rank is not None and (best_rank is None or rank < best_rank):
                        best, best_rank = i, rank
                if best is None:
                    break  # no known pair left -> keep chars (always encodable)
                parts[best:best + 2] = [parts[best] + parts[best + 1]]
            ids.extend(self.token_to_id[p] for p in parts)
        return ids

    def decode(self, ids: list[int]) -> str:
        # Pieces already carry their spaces. Just glue.
        return "".join(self.vocab[i] for i in ids)

    def save(self, path: str) -> None:
        # JSON can't hold tuple keys, so joins pairs with \x00.
        with open(path, "w") as f:
            json.dump({
                "vocab": {str(i): s for i, s in self.vocab.items()},
                "merges": ["\x00".join(pair) for pair in self.merges],
            }, f)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        # Rebuild both maps; merge rank = order in the saved list.
        with open(path) as f:
            raw = json.load(f)
        vocab = {int(i): s for i, s in raw["vocab"].items()}
        merges = {tuple(p.split("\x00")): r for r, p in enumerate(raw["merges"])}
        return cls(vocab, merges)


def train_bpe(texts: list[str], num_merges: int) -> BPETokenizer:
    # Count how words actually look, then fuse the most common pair, repeat.
    freqs: Counter[tuple[str, ...]] = Counter()
    for text in texts:
        for chunk in _SPLIT.findall(text):
            freqs[tuple(chunk)] += 1

    # Every char earns an id first, so encoding can always fall back.
    # Seeded with printable ASCII: training text won't cover every letter.
    chars = sorted(set(string.printable) | {ch for word in freqs for ch in word})
    vocab: dict[int, str] = {i: ch for i, ch in enumerate(chars)}
    merges: dict[tuple[str, str], int] = {}

    for _ in range(num_merges):
        # Most frequent adjacent pair across the whole corpus wins.
        pairs: Counter[tuple[str, str]] = Counter()
        for word, n in freqs.items():
            for i in range(len(word) - 1):
                pairs[(word[i], word[i + 1])] += n
        if not pairs:
            break
        best = max(pairs, key=lambda p: pairs[p])
        merges[best] = len(merges)
        vocab[len(vocab)] = best[0] + best[1]
        # Rewrite every word containing the pair, so counts stay honest.
        fused: Counter[tuple[str, ...]] = Counter()
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
