# llm-lab Roadmap

## Current Status (2026-09-19)

### Experiments Completed
- **exp011c** (m50m, 512 ctx, 16k vocab): best bpb 1.5854 @ step 5500, overfit after, degrade guard abort at step 8200
- **exp012** (m50m, 512 ctx, 16k vocab): A=dropout 0.0, B=accum2 eff-64. Both 8k steps.
- **exp013** (m50m_1k, 1024 ctx, 16k vocab): best bpb 1.7655 @ step 2400, early stop at step 4800. 1024 ctx did NOT help — same train/val gap pattern as 512, gap was a training-loop artifact not a data issue.
- **eval_diag**: Both checkpoints show near-zero train/val gap on frozen evaluation. The 0.6-0.7 nat gap during training was an artifact of live model updates, not data recycling or distribution shift.

### Key Findings
1. The train/val gap during training is NOT a data problem — it's how eval_loss interacts with the live training state
2. 1024 ctx doesn't improve generalization at this model size / data regime
3. The model overfits at ~0.5-1% corpus coverage (44-90M tokens seen)
4. Best bpb so far: 1.585 (512 ctx) vs 1.765 (1024 ctx) — 512 wins

## Next: Pythia Comparison Experiment

### Goal
Train Pythia-70M architecture (6L × 512H × 8 heads × FFN 2048) with GPT-NeoX tokenizer (50,304 vocab) on OWT corpus. Compare against published Pythia benchmarks.

### Architecture
- Config: `PYTHIA_6L512` (preset `pythia`)
- 6 layers, 512 hidden, 8 heads, FFN 2048 (4× hidden)
- ~70M params with 50k vocab (vs ~36M with 16k vocab)
- Uses our Transformer implementation (pre-norm, GELU FFN, RoPE)

### Tokenizer
- GPT-NeoX tokenizer from `EleutherAI/pythia-70m-deduped` (50,304 vocab)
- Same tokenizer the real Pythia was trained with
- Enables direct comparison to published results

### Plan

#### Phase 1: Mac Tokenization (Mac Lisa)
1. `pip install transformers tokenizers torch`
2. Tokenize 80 OWT shards with GPT-NeoX tokenizer → `openwebtext_combined_bpe_gptneox.bin`
3. Save vocab as `bpe_gptneox.json`
4. Run smoke 200 to verify it works
5. Report results

#### Phase 2: Ronin LR Sweep
Before the overnight run, do a short LR sweep:
- Test LRs: 6e-4, 1e-3, 1.5e-3
- 10-15 minutes each (~500-1000 steps)
- Compare val bpb at equal tokens
- Bad LR guess costs a whole night

#### Phase 3: Ronin Overnight Run
After LR sweep picks the winner:
- **Micro-batch**: 32 (keep accum=2 → effective batch 64)
- If OOM: drop to micro-batch 16, accum 4
- **Schedule by measured throughput**, not steps:
  1. Run 100 steps, measure tokens/sec
  2. Target ~1B tokens (7 hours at ~40k tok/s)
  3. Set total steps = budget / (64 × 1024)
- **Warmup**: 500-1000 steps
- **Peak LR**: from sweep (likely 6e-4 to 1.5e-3)
- **Cosine decay** to 10% of peak

#### Overnight Safeguards
- Checkpoint every ~30 minutes (atomic writes)
- Eval every 500 steps on fixed val set (~200 sequences)
- Every eval: score ~100 recently served batches + ~100 random train sequences (data recycling check)
- Degrade guard: require several consecutive worsening evals, not one bump
- Generate samples from 3 fixed prompts every ~1,000 steps
- **Target**: ~3.5-3.9 nats/token (~1.25-1.4 bpb) after 1B tokens

#### Performance Optimizations
- bf16 autocast
- F.scaled_dot_product_attention (SDPA)
- torch.compile
- Pre-tokenized memmapped shards (already done)

### Throughput Safeguards (from Claude, 2026-09-19)
- **Confirm throughput holds 10+ min** — step-100 reading doesn't show thermal throttling
- **Loader not cleared until ~45M tokens** (~step 700 at 64 seq/step). Clean evals at 100 are just pipeline validation.
- **Tripwire**: if served loss >0.3 nats below random_train, stop — data recycling detected
- **Check at steps 700, 1000, 2000** — these are the real loader tests
- **LR sweep first** (6e-4, 1e-3, 1.5e-3, warmup 800) — bad LR costs a whole night
- **Plan A recommended**: 1B tokens baseline (15k steps, 3.5h) + LR sweep + first rung up. Plan B (2B overnight) risks a full night on suboptimal LR.
