# ByteTokenizer: the simplest thing that works.
# Text -> numbers (A=65 etc), numbers -> text. No vocab to learn yet.
class ByteTokenizer:
    """Convert text to bytes and bytes back to text."""
    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, tokens: list[int]) -> str:
        return bytes(tokens).decode("utf-8")
    