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
The historical `pythia160` preset and its completed/partial runs used ctx 512 plus our
learned absolute-position table; they are useful history, but **not context/position-scheme
parity**. The new `pythia160_2k` preset is separate and aligns the key architecture settings:
ctx 2048, RoPE on 25% of head dimensions, no learned absolute-position table, bias-free
output head, parallel residual, and zero dropout. The old preset remains unchanged.

Our target effective batch is 131,072 tokens/update, 1/16 of Pythia's 2,097,152. At ctx 2048,
micro-batch 2 × accum 32 nearly filled the 16GB GPU; the measured safe choice is micro-batch 1 ×
accum 64. Linear LR scaling gives 3.75e-4 as a reference point, not an approved run value.
The current 1k trial used 3e-4 but its 1k-step cosine is not evidence for a 16k schedule. Other
training-recipe differences include bf16 autocast vs fp16 and warmup; the earlier 512-context
run cannot establish 2K-context parity.

## Chunked 2K comparison plan
The Pythia-160M step-1000 token anchor is 2,097,152,000 tokens. At our effective batch this is
16,000 optimizer updates. Shane selected four 4,000-step windows, each about 10.2 hours at the
measured 9.2 seconds/update. Chunks are one continuous run, not four independent LR cycles:
keep `--steps 16000` for the global cosine and set `--stop-at-step` to 4000, 8000, 12000, then
16000. The final schedule must start fresh; do not resume the 1k-step-cosine trial into it.

Each intermediate segment checkpoints and pauses, leaving `train_job.state.json` at
`paused_at_chunk`; the next segment starts only after Shane explicitly says go. The watchdog pins
one run name/stamp, the run keeps one `runs/<run_name>/loss.jsonl` and one registry row, and
`scripts/continue_train_chunk.py --stop-at-step <next-endpoint>` performs a validated resume. The
chunking support is tested on CPU split-vs-uninterrupted and with a sandboxed watchdog handoff.
Before the 16k run, choose the long-schedule LR and guard settings; the 1k `3e-4` result is a
pipeline/stability check, not a Pythia parity score.

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
