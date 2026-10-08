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

## Visual Observatory + inference (implemented; verification 2026-10-06)

The local browser Observatory now extends the run/arm decision board with
checkpoint loading, generation, layer/token tracing, captured attention/FFN
signals, and checkpoint comparison. It is served by `python -m viz.server` on
loopback; it is an unauthenticated local tool and must not be exposed externally.
The API/UI architecture, routes, constraints, and usage are documented in
[`VISUAL-OBSERVATORY.md`](VISUAL-OBSERVATORY.md). Verification: project test
suite **178 passed, 2 skipped**.

Remaining plan: the next experiment is the **160M Pythia-parity run on the Pile**, not the older 66M-vs-Pythia plan. The 70M full-Pile run established the baseline. **Launched 2026-10-07 23:00** as arm `p160-pile-parity-full`; what the launch actually did is in the *Launched* subsection below.

### Launch status (armed 2026-10-06, running 2026-10-07)

Done and verified: the full 13.1GB Pile text is tokenized on the **real HF
Pythia encoder** (`data/pile_train_full_bpe_pythia_hf.bin`, 3,236,695,626
tokens — id-level and decode-level verified, `--selftest` green);
`pythia-160m` main + `step1000` reference weights are downloaded to
`data/incoming/`; and the shared val fixture moved off the old 2.004B cache
tail (see below).

**Closed 2026-10-06/07 — the LR bracket and the VRAM smoke.** Two rungs at
2000 steps, identical init and data order, same fixture: **5e-4 final-block
1.4616 vs 3e-4 1.6204 — 0.159 bpb better, 3.1× the eval noise band**, every
paired eval in the last 8 favouring 5e-4. 3e-4 (the linear-batch-scaled Pythia
number, 6e-4/16 = 3.75e-4) is rejected: scaling down Pythia's 2,097,152
tok/step undershoots this stack, matching the 70M precedent. **5e-4 selected
for the parity run.** Full series and caveats:
`reports/lr_probe_160m_pile.json`. The micro-8 × accum-32 VRAM smoke came with
it: measured peak **8,982MB of 16,376MB**, so the plan's "EXTRAPOLATED —
smoke it" question is answered.

### Launched 2026-10-07 23:00 — the 24,694-step parity run

`viz/arm.py` armed **`p160-pile-parity-full`**: pythia160 preset, lr 5e-4, micro 8
× accum 32 (eff 131,072 tok/step), one full pass over
`data/pile_train_full_bpe_pythia_hf.bin` = **24,694 steps**, guards on (degrade
×1.15, saturation stop on a flat 25% of schedule), early stopping off
(`--patience-frac 0`). Supervised by watchdog → schtasks `Hermes_TrainRun` →
detached launcher, so an app restart cannot kill it. Row, lever, and running note:
`experiments.json` → `p160-pile-parity-full`.

What the launch actually did — the log reads like two runs because it is:

- First launch (23:00) died at **step 47** after ~4 min. No checkpoint written and
  no cause captured in `logs/p160-pile-parity-full.log`.
- The watchdog relaunched at **23:12:44** (relaunch 1 of 6). With no step checkpoint
  to resume from it started **fresh** in `runs/20261007_2312_pythia160_tokenizer_pile_train_full`
  — step-1 loss 10.9683 is bit-identical to the LR probe's, so init and data order
  match the rungs and LR is still the only lever. `runs/20261007_2300_...` is the
  dead 47-step prefix; both dirs stay registered to the arm.
- **Pace: 29,966 tok/s sustained** (the trainer's own steps 31–130 throughput
  block) = 4.37 s/step → **~30h**, ETA early **2026-10-09**. The plan's 10–11h and
  the job spec's own ~28h estimate were both short; neither applied the 2.3× FLOPs/token
  to the 70M's measured 81k tok/s.
- Peak GPU 8,982 / 16,376 MB. First three evals on the shared fixture: 2.5580 @100
  → 2.2638 @200 → 2.1254 @300. Bar to beat: `pythia-160m@step1000` = **1.4614
  bpb**, hit at our step 16,000 (131,072 × 16,000 = 2,097,152,000 tokens, the
  anchor, keeper-protected).

Still open: an **optional 6e-4 rung** (~1h45m), since 5e-4 is the best rung *tested*
rather than a proven optimum (Pythia's raw 160M peak is 6e-4) — run it after the
full run lands, not instead of it.

**Checkpoint retention decided 2026-10-07 (Shane: "keep the best" + the
reasons to keep more):** per run keep `_best` + the newest step (crash-resume
point) + every 5000th (diagnostic trajectory) + **the pythia-matched anchor
step** + finals, delete the rest. The anchor is the new bit: our step 16,000 is
pythia-160m@step1000 *to the token* (131,072 × 16,000 = 2,097,152,000) and it
is not on the 5000 grid, so the old policy would have deleted the checkpoint
the matched-token headline is measured from. Applied: **98 files / 89.2 GB
reclaimed** (276 → 178 files, 18/18 keepers verified present), leaving ~130 GB
and 1.2 TB free on D:. The long run will write ~50 step ckpts ≈ 98 GB and prune
back to ~14 GB after it finishes.

### The val fixture moved (2026-10-06) — read before quoting any bpb

Every scorer used to hardcode `pile_train_full_bpe_pythia70m.bin`. Its val
tail (source chars ~8.04–8.11GB) lies **inside** the 160M training set, so it
cannot be the shared fixture. The fixture is now the last 1% of
`pile_train_full_bpe_pythia_hf.bin` (`[3204328669:]`, 32,366,957 ids,
3.8743 bytes/token), resolved by `scripts/val_fixture.py` — which reads
`train_job.json`'s `--tok_cache` and ignores completed jobs — and gated by
`scripts/verify_val_slice_identity.py`. Re-scored on that one id set:

| | ctx 512 | native ctx 2048 |
|---|---|---|
| ours (70M, 2.004B tok) | **1.4735** | — |
| pythia-70m @ step1000 (2.097B tok) | 1.5680 | 1.4657 |
| gap | **−0.0945 (6.0% ahead)** | +0.0079 (0.5%, wash) |

Reference points on the same tail: pythia-70m *final* 1.1836,
pythia-160m@step1000 1.4614, pythia-160m final 1.0456 (the 160m numbers are
the baseline our 160M run is measured against). Superseded: the
1.3947/1.4715 (5.2%) pair — same method, previous fixture; verdict direction
unchanged. `data/incoming/pile_val_slice.txt` is retired and its meta now says
so; `extract_val_slice.py` must not be re-run.

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
   winning LR may differ on the fixed architecture. PARTIALLY ANSWERED
   2026-10-07: on the fixed (parity) stack at Pile scale, 5e-4 beats 3e-4 by
   0.159 bpb at matched 2000 steps — see `pile-160m-lr-probe`. The high side
   (6e-4, Pythia's raw peak) is still untested.
3. **Checkpoint retention**: DECIDED 2026-10-07 — keep best + last + every
   5000th + the pythia-matched anchor step (step 16,000 of a 131,072-tok/step
   run) + finals; `prune_checkpoints.py --apply` reclaimed 89.2 GB and keeps
   keepers manifest-protected. Earlier note: 87 files / 35 GB, policy was
   Shane's call; mechanics unchanged.

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
