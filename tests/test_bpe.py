from model.bpe import train_bpe, BPETokenizer

TINY = [
    "Once there was a princess",
    "Lily played with the ball",
    "Once Lily saw the princess",
    "the ball was big and brave",
]

def test_train_then_round_trip():
    tok = train_bpe(TINY, num_merges=30)
    # Unicode the toy corpus never saw: bytes carry it, merges don't care.
    for text in TINY + ["A brave bird played!", "café ñ “quotes”"]:
        assert tok.decode(tok.encode(text)) == text

def test_merges_actually_compress():
    # Frequent pairs should fuse: encoded len < char len on train text.
    tok = train_bpe(TINY, num_merges=30)
    raw = sum(len(t) for t in TINY)
    coded = sum(len(tok.encode(t)) for t in TINY)
    assert coded < raw

def test_vocab_size_math():
    # Vocab = 256 bytes + merges. No more, no less.
    tok = train_bpe(TINY, num_merges=20)
    assert len(tok.vocab) == 256 + 20

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

def test_whitespace_is_lossless():
    # Double spaces, tabs, newlines, indent: all must survive the trip.
    # The splitter may fuse, but it must never drop.
    tok = train_bpe(TINY, num_merges=30)
    nasty = "a  b   c\n    indented\n\tcode  123  don't"
    assert tok.decode(tok.encode(nasty)) == nasty

def test_eos_off_by_default():
    # Old behavior: no extra id, vocab math untouched.
    tok = train_bpe(TINY, num_merges=20)
    assert tok.eos_id is None
    assert len(tok.vocab) == 256 + 20

def test_eos_is_one_token_and_round_trips():
    tok = train_bpe(TINY, num_merges=20)
    eos_tok = BPETokenizer(tok.vocab, tok.merges, eos=True)
    assert len(eos_tok.vocab) == 256 + 20 + 1
    assert eos_tok.eos_id is not None
    ids = eos_tok.encode("princess<|endoftext|>ball")
    assert ids.count(eos_tok.eos_id) == 1
    assert eos_tok.decode(ids) == "princess<|endoftext|>ball"

def test_eos_save_load(tmp_path):
    tok = train_bpe(TINY, num_merges=20)
    eos_tok = BPETokenizer(tok.vocab, tok.merges, eos=True)
    path = str(tmp_path / "bpe_eos.json")
    eos_tok.save(path)
    back = BPETokenizer.load(path)
    assert back.eos_id == eos_tok.eos_id
    assert back.vocab == eos_tok.vocab

def test_old_files_load_untouched(tmp_path):
    # Pre-EOS files have no "eos" key: vocab must load byte-identical.
    import json
    tok = train_bpe(TINY, num_merges=20)
    path = str(tmp_path / "bpe_old.json")
    with open(path, "w") as f:
        json.dump({
            "vocab": {str(i): s.decode("latin-1") for i, s in tok.vocab.items()},
            "merges": [[a.decode("latin-1"), b.decode("latin-1")]
                       for a, b in tok.merges],
        }, f)
    back = BPETokenizer.load(path)
    assert back.eos_id is None
    assert back.vocab == tok.vocab
