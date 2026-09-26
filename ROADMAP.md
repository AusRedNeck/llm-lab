# llm-lab Roadmap

## Status update (2026-09-23) — read first; supersedes stale bits below

### exp014 done + FROZEN LADDER verdict (`scripts/eval_frozen_ladder.py`)
- exp014 (m50m @ bpe_owt50k_v2, full OWT): self-aborted step 5900 by the degrade guard; live best 1.7256 @ 2700; job completed.
- Ladder = all 12 ckpts scored frozen on ONE identical batch set (logs/frozen_ladder_exp014.json): **val bpb rises monotonically after 2500 — 1.7383 → 1.9402 @ 5500. REAL overfit. Guard fired correctly; best.pt is essentially the true peak (1.7400 vs 1.7383 @ 2500).**
- Frozen train ≈ frozen val at every rung (|gap| ≤ 0.006 nats) → splits clean; frozen ≈ live val (5500: 5.9349 vs 5.9871) → live val metric and guard decisions were accurate. eval_diag's "artifact" holds ONLY for the live TRAIN line / within-ckpt gap — it does NOT extend to the cross-step val trajectory.
- Standing pattern: m50m collapses at 440–901M tokens regardless of vocab (exp014 440M, exp011c 901M).

### Scoreboard — best bpb at equal tokens (directional; val ≈ frozen within ~0.01)

| Rank | Run/Vocab | Best bpb | Step | Tokens | Notes |
|------|-----------|----------|------|--------|-------|
| 1 | 015 pythia 50k | **1.4699** | 12000 | ~490M | annealing phase after ~7000 |
| 2 | 011c m50m 16k | 1.5854 | 5500 | ~90M | 16k era champion |
| 3 | 016 pythia160 50k | 1.5890 | 2100 | ~69M | architecture fix confirmed |
| 4 | 014 m50m 50k | 1.7256 | 2700 | ~44M | confounded (eff32) |
| 5 | 010 m50m 4k | 1.7596 | 2600 | ~41M | vocab bottleneck |

### LR Ladders — both closed

**16k ladder** (pythia preset, 2500 steps each, eff64):
6e-4 1.648 · 1e-3 1.573 · 1.5e-3 1.543 · 2e-3 1.527 · **3e-3 1.519** (winner) · 4e-3 1.743 (broken)

**50k ladder** (pythia preset, 1000 steps each, eff64):
1e-3 1.741 · **1.5e-3 1.695** (winner) · 2e-3 1.747 · 3e-3 2.503 (degenerate)

### exp016 — pythia160 structural fix verdict

**Pre-fix** (2026-09-23 06:28): collapsed at step 3900. Train loss flatlined ~3.49,
val spiked to 4.96. The structural mismatch audit found:
- Sequential vs parallel residuals (the big one)
- Full RoPE vs partial RoPE (`rotary_pct=0.25` in Pythia)

**Post-fix** (2026-09-23 13:17): parallel residuals + partial RoPE applied.
Clean train descent, best 1.5890 @ 2100, early-stop at 4700 (val > best+15%
while TRAIN still falling = classic overfit-at-hot-LR).

**Verdict:** Architecture fix CONFIRMED (crash cured). Capacity answer NO —
160M loses to 70M champion 1.4699. Pattern: val bottoms at 1300-2400 steps
then climbs while train descends in all three 50k long arms. The 70M's
improvement lived in its annealing phase (lr decay after step ~7000) —
neither 160M arm got there before the guard aborted.

**Hypothesis:** schedule-length, not size. The 20k schedule with degrade-frac
0.15 stops 160M before it reaches the regime where 70M improved.

### Standing pattern — m50m collapses at 440–901M tokens regardless of vocab

### GPU idle, no training in flight, watchdog at rest

---

## Viz Layer (completed 2026-09-23)

The viz layer answers three questions *while a run is live*:
1. Is this curve's shape the data problem, the capacity problem, or the LR problem?
2. What have we already proven about this lever — are we about to re-run a confounded arm?
3. When a run destabilizes — what broke first: logits, gradients, or attention?

### Components

**experiments.json** — one row per ARM: id, family, run_dirs, control_id,
lever, goal, verdict, note. The single source of truth for "what did we try
and what happened."

**viz/registry.py** — anti-circularity layer. Maps every run dir to ≤1 arm,
re-derives the real arg diff from loss.jsonl headers, flags undeclared or
confounded levers. `--check` exits 1 on any problem.

**viz/arm.py** — sanctioned launch gate. Writes both `train_job.json` and
the registry row atomically. Refuses to overwrite an unfinished arm.
All future launches go through here.

**viz/dashboard.py v2** — zero-deps HTML decision board with tokens-seen
x-axis, overfit panel, stability panel (gnorm/logit_max/entropy), and arm
board. `--watch` rebuilds on a timer.

**viz/probe.py** — offline checkpoint X-ray via the glass-box capture path.
Per-layer hidden-state norms, effective rank, attention entropy, locality,
top-k gap. Writes `probes/<ckpt-stem>.json`.

**train/train.py leading metrics** — gnorm, logit_max/std, entropy, tval
logged every 100 eval steps. Backward-compatible (old readers skip missing
keys). Wrapped in try/except so viz bugs never kill training.

### How to use

Launch an arm:
```
python viz/arm.py --id p160-lr2e-4-fix --family capacity-50k \
    --goal "cooler LR on the fixed 160M stack" --control p160-fix-long \
    --lever lr=5e-4->2e-4 \
    -- --steps 20000 --batch 8 --accum 8 --lr 2e-4 --preset pythia160 ...
python train_watchdog.py    # detached launch + supervision
```

Check registry health:
```
python viz/registry.py              # human report
python viz/registry.py --check      # machine: exit 1 on problems
python viz/registry.py --family capacity-50k
```

Probe a checkpoint:
```
python viz/probe.py checkpoints/...step2000.pt [--rotary-pct 0.25]
python viz/probe.py checkpoints/...step*.pt --out probes/
```

Build the dashboard:
```
python viz/dashboard.py              # one-shot
python viz/dashboard.py --watch 30   # rebuild every 30s
open viz/training.html
```

---

## Structural Mismatch Audit (2026-09-23) — Pythia vs Our Implementation

### Three mismatches against EleutherAI/pythia-160m config.json

**1. Sequential vs parallel residuals — the big one.**
Pythia uses `use_parallel_residual=true` (GPTNeoX): attention and FFN
branch independently from pre-norm'd x, both dropout-gated, summed, then
final norm. We had sequential cascade — attn→residual→norm→FFN→residual→norm.
Fix: `block.py` rewritten for parallel residuals (commit `e6a526a`).

**2. Rotary encoding — full coverage vs partial.**
Pythia config: `"rotary_pct": 0.25` — only 25% of features per head get
rotary position bias. We had full RoPE. Fix: `rotary_pct` parameter added
to TransformerBlock; set 0.25 for Pythia parity.

**3. Activation — GELU is correct (matches).** No change needed.

### What happened at step 3900
Not an LR crash (LR was still near-peak). The sequential-residual structure
revealed its weakness after ~100M tokens: deep optimization surface,
validation gradients diverge while training gradients stay bounded.

---

## Experiment Families

### LR ladders (completed)
- **pythia16k-lr-ladder**: 5 arms, winner 3e-3 (bpb 1.5192 @ 2500)
- **pythia50k-70m-lr-probe**: 4 arms, winner 1.5e-3 (bpb 1.6953 @ 1000)
- **pythia160-lr-probe**: 3 arms, winner 5e-4 (bpb 1.7280 @ 1000, pre-fix)

### Capacity series (concluded)
- **exp014** (m50m 50k): fail, worst of trio, confounded by eff_batch
- **exp015** (pythia 70M 50k): pass, **1.4699 champion**
- **exp016** (pythia160 50k): architecture fix pass, capacity NO

### Context length
- **exp013** (m50m 1k ctx): inconclusive, OOM-forced config change mid-flight

### Reference
- **m50m-16k-ref**: directional 1.5854, era ended when 50k stream landed

---

## Open Questions

1. **Schedule-length hypothesis**: the 160M arms early-stop before the
   annealing phase where 70M improved. Does a longer schedule (or cosine
   restart) let 160M reach that regime?
2. **LR on the fixed stack**: all three 160M LR probes ran pre-fix. The
   winning LR may differ on the fixed architecture.
3. **Checkpoint retention**: 87 files / 35 GB currently; retention policy
   is Shane's call (prune_checkpoints.py handles the mechanics).

---

## Original experiments (pre-registry era)

### 001 — Tiny Transformer (DONE)
Baseline byte-level transformer, minimal architecture.

### 002 — BPE Tokenization (DONE)
BPE with vocab 2048, embed 384, 6L/6H, ctx 256, ~10M params.

### 003 — RoPE (DONE)
Rotary position embeddings on top of BPE2K.

### 004 — BPE Efficiency Win (DONE)
BPE vs byte-level: ~30% less surprise per byte.

### 005 — Overfitting Knee (DONE)
20k steps, val peaked at 2.086 @ 4500 then climbed to 2.89.

### 006 — Dropout + Early Stopping (DONE)
Dropout 0.1, patience=5, early stop works.

### 007 — Book Era (LibriSpeech) (DONE)
~10M params on 193M tokens. Best val 4.84. Capacity wall.

### 008 — TinyStories 50M (DONE)
Massive overfit — 50M too much for TinyStories.

### 010 — M50M 4k Vocab (DONE)
Best 4.0986. 4k vocab too small; embedding bottleneck.

### 011c — M50M 16k Vocab (DONE)
Best **1.5854** bpb @ step 5500. 16k beats 4k by ~10%.
Overfit at ~1% corpus coverage; knee ~5,500 steps.
