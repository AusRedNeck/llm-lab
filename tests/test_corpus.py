import torch

from train.cleanse import cleanse_text
from train.train import resolve_corpus_files, split_corpus


def test_cleanse_markers():
    # Boilerplate out, story stays, END marker cuts the tail.
    raw = ("Title page junk\n"
           "*** START OF THIS PROJECT GUTENBERG EBOOK TEST ***\n"
           "Once upon a time.\n"
           "*** END OF THIS PROJECT GUTENBERG EBOOK TEST ***\n"
           "License legalese.\n")
    body, status = cleanse_text(raw)
    assert status == "ok"
    assert body == "Once upon a time.\n"


def test_cleanse_nospace_variant():
    # "***START OF" without the space still matches.
    raw = ("junk\n***START OF THE PROJECT GUTENBERG EBOOK X,\nbody here\n")
    body, status = cleanse_text(raw)
    assert status == "ok"
    assert "junk" not in body and "body here" in body


def test_cleanse_quarantine():
    # No markers, old-style, australia: returned untouched + flagged.
    _, s1 = cleanse_text("Project Gutenberg Etext of Foo\nbody\n")
    _, s2 = cleanse_text("<table><a href=\"http://gutenberg.net.au\">x</a></table>\nbody\n")
    _, s3 = cleanse_text("Just a story, no headers at all.\n")
    assert (s1, s2, s3) == ("oldstyle", "australia", "no_markers")


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
