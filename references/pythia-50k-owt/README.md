# GPT-NeoX BPE Tokenizer — 50k merges, trained on OpenWebText

## Summary

- **Vocab size**: 50,256 (base 256 + 50,044 merge rules)
- **Training data**: 200 MB sample from `data/openwebtext_combined.txt`
- **Format**: HuggingFace-compatible BPE tokenizer with ByteLevel pre-tokenizer/decoder
- **Merges**: String format (`"a b"`) compatible with gpt-neox / transformers
- **Saved as**: `train_gpt_neox_tokenizer.py --vocab_size 50256 --sample_mb 200`

## Benchmark vs bpe8k (Gutenberg-trained)

On 2M chars of OpenWebText:

| Tokenizer | Tokens | Chars/token | Unique IDs | Vocab used |
|-----------|--------|-------------|------------|------------|
| **50k-OWT** | 424,684 | 4.71 | 28,139 | 56% |
| bpe8k (Gut)| 629,782 | 3.18 | 5,930 | ~1% |

**Result**: 32.6% fewer tokens, 1.48x better compression ratio.

The Gutenberg-trained bpe8k only uses ~1% of its vocabulary on internet text — it learned words from 19th-century literature that don't appear in modern web content. The OWT-trained 50k learns actual subword patterns from the target corpus.

## Files

- `tokenizer.json` — Full tokenizer spec (vocab + merges)
- `tokenizer_config.json` — Metadata (model_type=gpt_neox, vocab_size=50256)

## Usage with transformers

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("references/pythia-50k-owt")
ids = tok.encode("Hello world!")
```

## Re-training

```bash
cd D:/Projects/llm-lab
python train_gpt_neox_tokenizer.py --vocab_size 50256 --sample_mb 200
```

To increase vocab size or adjust training data, modify `--vocab_size` and `--sample_mb`.
