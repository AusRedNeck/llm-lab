# Pythia-160M Reference

Source: EleutherAI/pythia-160m (HuggingFace)
License: Apache 2.0
Paper: arxiv.org/abs/2304.01373

## Architecture
- Params: 160M
- Layers: 12
- Hidden: 768
- Heads: 12 (64 dim/head)
- Context: 2048
- Vocab: 50304 (GPT-NeoX BPE)
- Activation: GELU
- Position: RoPE (25% of hidden dims, base 10000)
- Parallel residual: True (attn + MLP in parallel, not sequential)
- Tie embeddings: False
- Dropout: 0.0 (no dropout in Pythia)

## Training
- Corpus: The Pile (825GB, 22 sources)
- Tokens: 299,892,736,000 (~300B)
- Batch size: 2M tokens (2,097,152)
- Steps: 143,000
- Checkpoints: 154 (every ~2B tokens)
- Optimizer: AdamW (standard)

## Key Differences from Our m50m
1. 3× the params (160M vs 55M)
2. 4× the context (2048 vs 512)
3. 30× the data (300B vs 9.75B tokens)
4. Parallel residual (we use sequential)
5. No dropout (we use 0.1)
6. RoPE on 25% of dims (we apply to all)
7. GPT-NeoX architecture (slightly different attn/MLP layout)

## Why It's a Good Yardstick
- Same research goal: understand training dynamics at small scale
- Full training details published
- 154 checkpoints available for curve comparison
- Apache 2.0 (we can use it)
- RoPE + modern architecture (not legacy GPT-2)
