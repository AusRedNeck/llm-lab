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
- Batch size: 2M tokens (2,097,152) = micro 32 x 32 GPUs x 2048 ctx
- Steps: 143,000
- Checkpoints: 154 (every ~2B tokens)
- Optimizer: Adam, betas (0.9, 0.95), eps 1e-8, weight-decay 0.1, grad clip 1.0
- **LR: 6.0e-4 peak, cosine to 6e-5 (10%), warmup 0.01 = 1,430 steps (1%)**
- Precision: fp16 mixed, dropout 0
- Source: EleutherAI/pythia `models/160M/pythia-160m.yml` (the config they
  actually trained with), plus the suite table in that repo's README.

### Suite learning rates (all at 2,097,152 tokens/step)
14M/31M/70M **1.0e-3** · 160M **6.0e-4** · 410M/1B **3.0e-4** · 1.4B 2.0e-4 ·
2.8B 1.6e-4 · 6.9B/12B 1.2e-4

### What that means for our runs
We train at eff 131,072 tokens/step = **1/16 of Pythia's batch**, so their raw
6e-4 is not our number. Linear batch scaling puts the equivalent at
6e-4/16 = **3.75e-4**, which is why the 160M Pile probe brackets 3e-4 vs 5e-4
(the 70M parity run followed the same pattern: Pythia-70m's 1e-3 scaled is
6.25e-4, we ran 5e-4). The rest of the recipe we already match: betas,
weight-decay 0.1, clip 1.0, cosine to 10%, dropout 0, rotary 0.25, parallel
residual. Remaining differences: ctx 512 vs 2048, bf16 autocast vs fp16, and
warmup — theirs is 1% of the schedule, our default `min(200, steps//10)` is
0.8% of a 24,694-step run but 10% of a 2,000-step probe.

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
