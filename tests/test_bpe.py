from model.bpe import train_bpe, BPETokenizer

TINY = [
    "Once there was a princess",
    "Lily played with the ball",
    "Once Lily saw the princess",
    "the ball was big and brave",
]

def test_train_then_round_trip():
    tok = train_bpe(TINY, num_merges=30)
    for text in TINY + ["A brave bird played!"]:
        assert tok.decode(tok.encode(text)) == text

def test_merges_actually_compress():
    # Frequent pairs should fuse: encoded len < char len on train text.
    tok = train_bpe(TINY, num_merges=30)
    raw = sum(len(t) for t in TINY)
    coded = sum(len(tok.encode(t)) for t in TINY)
    assert coded < raw

def test_vocab_size_math():
    # Vocab = seeded alphabet (+ any new chars) + merges. No more, no less.
    import string
    chars = len(set(string.printable) | set("".join(TINY)))
    tok = train_bpe(TINY, num_merges=20)
    assert len(tok.vocab) == chars + 20

def test_common_word_becomes_one_token():
    # "the" is everywhere -> should fuse into a single id.
    tok = train_bpe(TINY, num_merges=50)
    assert len(tok.encode("the")) == 1

def test_save_load_round_trip(tmp_path):
    tok = train_bpe(TINY, num_merges=30)
    path = str(tmp_path / "bpe.json")
    tok.save(path)
    back = BPETokenizer.load(path)
    assert back.decode(back.encode("Lily and the princess")) == "Lily and the princess"
    assert back.vocab == tok.vocab
    assert back.merges == tok.merges
