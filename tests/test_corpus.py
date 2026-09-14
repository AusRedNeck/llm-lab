import torch

from train.train import resolve_corpus_files, split_corpus


def test_split_corpus_normal():
    # Big corpus: both splits usable (1% tail must exceed one window).
    c = torch.arange(20000)
    train, val = split_corpus(c, 0.01, 128)
    assert val is not None
    assert len(train) == 19800
    assert len(val) == 200


def test_split_corpus_tiny_val_turns_off():
    # Val tail shorter than one window: val off, train keeps going.
    c = torch.arange(6000)
    train, val = split_corpus(c, 0.01, 512)
    assert val is None
    assert len(train) == 5940


def test_resolve_corpus_single_file(tmp_path):
    p = tmp_path / "ch1.txt"
    p.write_text("hello\n")
    assert resolve_corpus_files(str(p)) == [str(p)]


def test_resolve_corpus_dir(tmp_path):
    (tmp_path / "a.txt").write_text("a\n")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.txt").write_text("b\n")
    files = resolve_corpus_files(str(tmp_path))
    assert len(files) == 2
    assert files == sorted(files)
