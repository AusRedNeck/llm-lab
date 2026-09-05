from model.bpe import train_bpe
from model.config import BPE2K_10M
from train.train import PRESETS, encode_file_lines

TINY = ["Once there was a princess", "Lily played with the ball"]


def test_encode_file_lines_matches_direct(tmp_path):
    # File helper must agree with tok.encode, line for line.
    tok = train_bpe(TINY, num_merges=20)
    p = tmp_path / "tiny.txt"
    p.write_text("\n".join(TINY) + "\n\n")  # trailing blank line skipped
    assert encode_file_lines(str(p), tok) == tok.encode(TINY[0]) + tok.encode(TINY[1])


def test_encode_file_lines_limit(tmp_path):
    tok = train_bpe(TINY, num_merges=20)
    p = tmp_path / "tiny.txt"
    p.write_text("\n".join(TINY) + "\n")
    assert encode_file_lines(str(p), tok, limit=1) == tok.encode(TINY[0])


def test_bpe2k_preset_registered():
    # Preset exists, points at the real vocab, keeps the 10M shape.
    assert PRESETS["bpe2k"] is BPE2K_10M
    assert BPE2K_10M.vocab_size == 2256
    assert BPE2K_10M.context_length == 256
