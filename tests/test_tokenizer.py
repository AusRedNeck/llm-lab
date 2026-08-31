from model.tokenizer import ByteTokenizer

def test_encode_ascii():
    tokenizer = ByteTokenizer()

    assert tokenizer.encode("hello") == [104, 101, 108, 108, 111]

def test_decode_ascii():
    tokenizer = ByteTokenizer()

    assert tokenizer.decode([104, 101, 108, 108, 111]) == "hello"

def test_encode_decode_round_trip():
    tokenizer = ByteTokenizer()
    text = "Hello, world"

    assert tokenizer.decode(tokenizer.encode(text)) == text

def test_unicode_round_trip():
    tokenizer = ByteTokenizer()
    text = "Hello, café!"

    assert tokenizer.decode(tokenizer.encode(text)) == text