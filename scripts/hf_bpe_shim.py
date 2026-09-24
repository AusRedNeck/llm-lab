#!/usr/bin/env python3
"""Wrapper to make HF BPE tokenizers match the BPETokenizer.encode() interface.

BPETokenizer.encode(text) -> list[int]
HF Tokenizer.encode(text)  -> Encoding(ids=[...])

This shim provides .encode(text) -> list[int] so token_io.encode_file_stream()
and tokenize_owt_16k.py work with HF-format vocab files without changes.
"""
from __future__ import annotations

import os


class HfBpeShim:
    """Thin wrapper: tok.encode(text) -> list[int], compatible with BPETokenizer API."""

    _hf_module = None

    def __init__(self, hf_tok):
        self._hf = hf_tok

    @classmethod
    def _get_tokenizer(cls):
        if cls._hf_module is None:
            from tokenizers import Tokenizer as T
            cls._hf_module = T
        return cls._hf_module

    @classmethod
    def from_file(cls, path: str) -> 'HfBpeShim':
        """Load from HF tokenizer.json."""
        return cls(cls._get_tokenizer().from_file(path))

    @classmethod
    def load(cls, path: str) -> 'HfBpeShim':
        """Alias for from_file (compat with BPETokenizer.load())."""
        return cls.from_file(path)

    def encode(self, text: str):
        """Encode text → list[int] (matches BPETokenizer signature)."""
        ids = self._hf.encode(text)
        return ids.ids if hasattr(ids, 'ids') else ids

    def decode(self, ids: list[int]) -> str:
        """Decode IDs back to text."""
        return self._hf.decode(ids)


if __name__ == "__main__":
    # Quick test
    path = "D:/Projects/llm-lab/references/pythia-50k-owt/tokenizer.json"
    shim = HfBpeShim.load(path)
    print(f"Loaded {os.path.basename(path)}")
    result = shim.encode("Hello world! This is a test.")
    print(f"encode('Hello world! ...') -> {result}")
    decoded = shim.decode(result)
    print(f"decode -> '{decoded}'")
