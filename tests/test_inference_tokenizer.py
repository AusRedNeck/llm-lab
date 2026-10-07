from tokenizers import Tokenizer, decoders, models, pre_tokenizers

from inference.generate import load_tokenizer


def test_generate_tokenizer_loader_handles_huggingface_json(tmp_path):
    tokenizer = Tokenizer(models.BPE({"<unk>": 0, "a": 1}, [], unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    path = tmp_path / "tokenizer.json"
    tokenizer.save(str(path))
    loaded = load_tokenizer(str(path))
    ids = loaded.encode("a")
    assert ids == [1]
    assert loaded.decode(ids) == "a"
