LLM Lab

Purpose
-------
Learn by building a small language model from scratch,
then progressively introduce modern techniques and a tool harness.

Principles
----------
- Understand before abstracting.
- Keep model and harness separate.
- Prefer measurable experiments.
- Reproduce experiments across machines.
- Record what changed and what happened.

Setup
-----
After git pull, run:

    bash scripts/setup.sh

This creates the venv, syncs deps, and installs CUDA torch on Windows.
Mac just needs `uv sync` (scripts/setup.sh handles both).

To verify CUDA: `uv run python -c "import torch; print(torch.cuda.is_available())"`

Machines
--------
- PC (Ronin, 4070 Ti Super)  — primary dev/training (CUDA)
- Mac (MacBook Pro, MPS)      — training/inference
- Mini (Mac Mini, Win10)      — inference/testing

---

## Project Structure

```
llm-lab/
├── train/
│   └── train.py          # trainer — resume, early stopping, leading metrics
├── model/
│   ├── transformer.py    # Transformer (pre-norm, GELU, RoPE, parallel residuals)
│   ├── block.py          # TransformerBlock (parallel residual, rotary_pct)
│   ├── attention.py      # MHA + rotary position embeddings
│   └── config.py         # ModelConfig presets (m50m, pythia, pythia160, etc.)
├── model/bpe.py          # BPE tokenizer
├── viz/
│   ├── dashboard.py      # zero-deps HTML dashboard (--watch, tokens-seen x-axis)
│   ├── registry.py       # run registry checker (anti-circularity layer)
│   ├── arm.py            # arm new experiments (writes spec + registry row)
│   ├── probe.py          # offline checkpoint X-ray (layer norms, attn entropy)
│   ├── hooks.py          # forward-capture connector (used by probe.py)
│   └── training.html     # generated dashboard (open in browser)
├── experiments.json      # experiment registry (one row per ARM)
├── train_job.json        # active job spec (arm.py writes this)
├── train_watchdog.py     # cron supervisor (every 10 min)
├── train_launch.py       # detached launcher (called by watchdog)
├── run_train.bat         # Task Scheduler entry point
├── run_train_hidden.vbs  # silent wrapper for Task Scheduler
├── prune_checkpoints.py  # keeper-aware checkpoint retention
├── audit_run_data.py     # data integrity checker
└── data/                 # corpora + tokenized binaries
```

---

## Viz Layer

Three layers that answer "is this run healthy, and did this lever help?"

### Layer A — Leading Metrics (in `train/train.py`)

Every 100-step eval now logs additional scalars (backward-compatible: old
readers skip missing keys):

| Key | What | Why |
|-----|------|-----|
| `gnorm` | Pre-clip gradient L2 norm | Rising norm precedes collapse |
| `logit_max` | Max logit from last batch | Rising scale = instability |
| `logit_std` | Std of logit distribution | Overconfidence meter |
| `ent` | Mean predictive entropy | "Guessing" vs "overconfident-wrong" |
| `tval` | Val loss on same fixed val data | `train - tval` = memorization gap |

All wrapped in `try/except` so a viz bug can never kill a training run.

### Layer B — Run Registry (`experiments.json` + `viz/registry.py`)

Every experiment ARM has a row: id, family, run_dirs, control_id, lever,
goal, verdict, note. `viz/registry.py --check` verifies:

- Every run dir maps to exactly one arm
- The real arg diff (from loss.jsonl headers) matches the claimed lever
- Multi-lever arms are flagged `CONFOUNDED` (unless `accepted_confounded`)
- Unregistered run dirs are flagged

`viz/arm.py` is the sanctioned way to launch arms — it writes both
`train_job.json` and the registry row atomically.

### Layer C — Dashboard (`viz/dashboard.py`)

Zero-deps, one self-contained HTML file, `file://` works. Run:

    python viz/dashboard.py              # one-shot
    python viz/dashboard.py --watch 30   # rebuild every 30s

Panels:
1. **Tokens-seen x-axis** — comparable across batch/vocab/ctx differences
2. **Overfit panel** — train vs val vs tval gap
3. **Stability panel** — gnorm, logit_max, entropy per run (collapse reads as shape)
4. **Best-so-far marker** with early-stop rule annotation
5. **Arm board** — table of runs with lever, control, diff, verdict

### Layer D — Checkpoint Probe (`viz/probe.py`)

Offline X-ray of a checkpoint (never in the training loop). One 256-token
forward through `viz/hooks.py` capture path:

    python viz/probe.py checkpoints/...step2000.pt [--rotary-pct 0.25]
    python viz/probe.py checkpoints/...step*.pt --out probes/

Metrics: per-layer hidden-state norm, effective rank, attention entropy,
locality (copy/induction signal), top1/top10 gap, probe bpb.

---

## Experiments

### 001 — Tiny Transformer
Status: DONE
Baseline byte-level transformer. Minimal architecture to validate the harness.

### 002 — BPE Tokenization (BPE2K)
Status: DONE
Switched from byte-level to BPE with vocab size 2048 (actual vocab: 2256 with special tokens).
Config: embed 384, 6 layers, 6 heads, context 256, ~10M params.
Data: TinyStories.

### 003 — RoPE (Rotary Position Embeddings)
Status: DONE
Replaced learned positional embeddings with rotary position embeddings on top of the BPE2K config.
Config: same as 002 — vocab 2256, ctx 256, embed 384, 6L, 6H, ~10M params.
Batch 64, lr 3e-4, TinyStories.

### 004 — BPE Efficiency Win
Status: DONE
BPE vs byte-level on same architecture (bpe2k + RoPE, 5k steps):

| Model       | Val NLL/token | Bytes/token | Nats/byte |
|-------------|---------------|-------------|-----------|
| bytes rope  | 0.7314        | 1.00        | 0.7314    |
| bpe2k rope  | 2.1150        | 4.11        | 0.5148    |

~30% less surprise per byte of story. BPE era confirmed.

### 005 — Overfitting Knee (20k steps)
Status: DONE
Same bpe2k+rope recipe, 4x the steps. Val peaked at 2.086 @ step 4500,
then climbed to 2.89 by step 20k. Discovered early stopping empirically.
Cross-machine reproducibility confirmed (PC ≈ Mac to 4 decimal places).

### 006 — Dropout + Early Stopping
Status: DONE
Dropout 0.1 trades a tiny bit of peak val (2.15 vs 2.09) for flat val curve.
Early stopping fires correctly: patience=5 on val checks.

### 007 — Book Era First Crunch (LibriSpeech)
Status: DONE
Config: libri10m preset, ctx 512, ~10M params. Early stop @ step 6800, best val 4.84.
Verdict: pipeline works end-to-end on real literature (193M tokens).

### 008 — TinyStories 50M Control
Status: DONE
Config: bpe50m preset, ~52M params. Step 20000, train 0.003, val 5.89.
Verdict: massive overfit — 50M is too much for TinyStories.

### 010 — M50M Full OpenWebText (4k vocab)
Status: DONE
Config: m50m preset, vocab 4512, ~55M params. Best val 4.0986 @ step 2600.
Verdict: 4k vocab too small; embedding layer is bottleneck. Next: 16k vocab.

### 011c — M50M Full OpenWebText (16k vocab)
Status: DONE
Best **bpb 1.5854** @ step 5500. Guard fired at step 8200 (val > best+30%).
16k beats 4k by ~10% byte-normalised. Overfit at ~1% corpus coverage;
knee at ~5,500 steps; 20k schedule was ~3.7x too long.

Pipeline (rewritten Sep 18 — the first three attempts all died):
  run scripts/run_16k_smoke.bat   rehearsal: 3 shards x 20M chars -> verify -> 200 train steps (~2.5 min)
  run scripts/run_16k_full.bat    the real thing: 80 shards, serialized, resumable (~7h)
  scripts/tokenize_owt_16k.py     orchestrator (manifest, resume, preflight disk check)
  scripts/token_io.py             streaming encoder + memmap reader (no %TEMP%, no RAM blowup)
  scripts/verify_tokens.py        7 acceptance checks — run before trusting any long encode
Train with the .bin:  --tok_cache data/openwebtext_combined_bpe_owt16k.bin

### 014 — M50M 50k Vocab
Status: DONE (verdict: fail)
Config: m50m trunk on bpe_owt50k_v2, eff batch 32. Best 1.7256 @ 2700.
WORST of the three long arms. But confounded: 16k ref ran eff64 vs this eff32.
50k-toxicity signal on the m50m trunk — directional only.

### 015 — Pythia 70M on 50k Vocab
Status: DONE (verdict: pass)
Config: pythia preset, bpe_owt50k_v2, lr 1.5e-3, eff batch 64.
Best **1.4699** @ step 12000 (frozen audit 1.4692). The champion.
Improvement lived in the annealing phase (lr decay after step ~7000).

### 016 — Pythia 160M
Status: DONE (verdict: fail on capacity, pass on architecture fix)
Config: pythia160 preset, bpe_owt50k_v2, lr 5e-4, eff batch 64.
Pre-fix: collapsed at step 3900. Post-fix (parallel residuals + partial RoPE):
clean train descent, best 1.5890 @ 2100, early-stop at 4700.
**Capacity answer: NO** — 160M still loses to 70M champion.
**Architecture fix: CONFIRMED** — crash cured, but 160M never got to the
annealing phase where 70M's improvement lived. Hypothesis: schedule-length,
not size.

---

## Current State (2026-09-23) — read this first, it's the handoff

**GPU idle, no training in flight, watchdog at rest.** The completed
pythia160 fix-long run (attempt 2, stamp 202609231317) is the latest finished
job.

### Scoreboard — best bpb at equal tokens (directional; val ≈ frozen within ~0.01)

| Rank | Run/Vocab | Best bpb | Step | Tokens | Notes |
|------|-----------|----------|------|--------|-------|
| 1 | 015 pythia 50k | **1.4699** | 12000 | ~490M | annealing phase after ~7000 |
| 2 | 011c m50m 16k | 1.5854 | 5500 | ~90M | 16k era champion |
| 3 | 016 pythia160 50k | 1.5890 | 2100 | ~69M | architecture fix confirmed |
| 4 | 014 m50m 50k | 1.7256 | 2700 | ~44M | confounded (eff32) |
| 5 | 010 m50m 4k | 1.7596 | 2600 | ~41M | vocab bottleneck |

### LR Ladders — both closed

011c resumes from 011b's keeper (step 2500, val 4.7666 / bpb 1.7013) with
`--patience-frac 0.15 --min-steps-frac 0.6 --degrade-frac 0.30`, so the earliest a
plateau can stop it is step 15000, where the LR has decayed to ~7e-5 from 3e-4.
Launcher: `scripts/run_exp011c_train.bat` — its header documents the git-bash invocation
trap (`cmd //c foo.bat` silently does nothing; use
`MSYS_NO_PATHCONV=1 cmd.exe /c "foo.bat"`) and the interpreter trap (bare `python`
is the CUDA venv; `llm-lab/.venv` is CPU-only torch and will train 50x slower
without ever looking wrong).

**16k ladder** (pythia preset, 2500 steps each, eff64):
6e-4 1.648 · 1e-3 1.573 · 1.5e-3 1.543 · 2e-3 1.527 · **3e-3 1.519** (winner) · 4e-3 1.743 (broken)

**50k ladder** (pythia preset, 1000 steps each, eff64):
1e-3 1.741 · **1.5e-3 1.695** (winner) · 2e-3 1.747 · 3e-3 2.503 (degenerate)

### Standing pattern — m50m collapses at 440–901M tokens regardless of vocab

### 16k OWT corpus — DONE
9,745,672,850 tokens (38.98GB int32), 80/80 shards, verify 7/7 green.
`data/openwebtext_combined_bpe_owt16k.bin`

### Disk / retention
87 files / 35 GB — `prune_checkpoints.py` keeps `_best.pt`, last step-ckpt,
and every 5000th step. Keepers backed up to Google Drive.

### Data integrity audit
31 run dirs, 31 curves, 87 checkpoints, 18/18 keeper hashes verified — 0 errors.

### Curated corpora on disk (~149GB)
`data/cosmopedia/` (86 GiB), `data/finewiki/` (36 GiB), `data/open_web_math/` (26 GiB).

---

## Operational Notes

### Training launch rule
**Never launch training as a child of the Hermes app process.**
An app update will kill it. Use detached launch:

    python train_watchdog.py              # cron fires this every 10 min
    python viz/arm.py --id ... -- ...     # write spec + registry, then watchdog

Or manually: `schtasks /run /tn Hermes_TrainRun`

**16k OWT corpus (original notes):** `data/openwebtext_combined_bpe_owt16k.bin` (raw int32, ~38GB, ~9.46B
tokens expected). Encoded in parallel (8 worker processes, `scripts/run_16k_full_par.bat`),
~1.6M tok/s aggregate.

### Watchdog supervision chain
watchdog → schtasks Hermes_TrainRun → hidden vbs → run_train.bat →
train_launch.py → trainer

Task Scheduler is the parent; app restarts cannot kill the run.
Watchdog relaunches resume from the newest STEP-ckpt, never `_best.pt`.

### Early stopping parameters
- `--patience-frac 0.15` — patience as fraction of total steps
- `--min-steps-frac 0.6` — no plateau stop before 60% of schedule
- `--degrade-frac 0.15` — abort if val > best + 15% (measured noise ~3.75%)

Check progress:  `tail -2 logs/16k_tokenize_par.log`   (status lines every 60s)
Rerun by hand:   `scripts/run_16k_full_par.bat`   (resumable — finished shards are skipped)
Verify:          `python scripts/verify_tokens.py --bin data/openwebtext_combined_bpe_owt16k.bin --vocab data/bpe_owt16k.json --src data/openwebtext/shards/train-00000-of-00080.txt --manifest`
Tests:           `python -m pytest tests/test_tokenizer_safety.py tests/test_bpe_resilience.py -q`  (16 tests)

### Pythia architecture parity
Post-fix (commit `e6a526a`): parallel residuals + `rotary_pct=0.25`.
Matches GPTNeoX `use_parallel_residual=true` and partial rotary encoding.
Set `--rotary-pct 0.25` when probing post-fix checkpoints.

---

## Next

The 160M arms keep early-stopping before reaching the annealing phase where
the 70M's improvement lived. Two open questions:

1. **Schedule-length**: does a longer schedule (or cosine restart) let 160M
   reach its annealing phase and close the gap to 1.4699?
2. **LR on the fixed stack**: all three 160M LR probes ran pre-fix; the
   winning LR may differ on the fixed architecture.

---

### Keeper backups — finally OFF-machine

The 18 keepers (6.70 GB) now exist in four places: the working dir, `llm-lab-private/checkpoints/keepers/`, a second physical SSD, and — the one that survives the machine being lost — **Google Drive** (`llm-lab keepers`). The first Drive attempt hit `storageQuotaExceeded` on the 16 GB free tier; the account has since moved to 100 GB. Worth knowing what was actually filling it: not the keepers (5.07 GB) but **7.35 GB of two `lmu_session_*.jsonl` telemetry files** plus 1.27 GB of a retokenized corpus cache that already exists locally. Off-machine also means the upload order matters.

### Data integrity audit — 0 errors, and three findings that only LOOK like corruption

`audit_run_data.py` (new) checks curve integrity, run identity, checkpoint/step agreement, keeper hashes and hygiene across the whole lab: **31 run dirs, 31 curves, 87 checkpoints in 39 families, 18/18 keeper hashes verified — 0 errors, 7 warnings.**

The seven warnings are honest ones, and getting to them required fixing the *audit* three times, because each of these is expected behaviour that a naive check reports as damage:

| looks like | actually is |
|---|---|
| duplicate steps + schema drift on 7 runs | the `early_stop` **summary row**, which deliberately reuses the final step number. Exclude it from the data set and validate it separately. |
| 37 "orphan" checkpoints | run dirs are stamped `YYYYMMDD_HHMM`; checkpoint families `YYYYMMDDHHMM`. Compare digits, not strings. |
| 16 errors on 9 runs | **abandoned stubs** — 0-byte or header-only curves from the failing dense111m/194m attempts. Parked in `runs/_abandoned_stubs_2026-09-15/` (6 KB, nothing lost). |

What remains is real and worth knowing: **6 checkpoint families have weights but no run dir anywhere** (the Sep 5/11/13 bpe2k + libri10m runs — their curves live on the Mac or in Drive, not here), and **1 stamp-drift case** where one logical run carries two stamps (`..._202609181300` family vs the `20260918_1307` run dir — 7 minutes apart). That drift is exactly why the resume-continuity fix persists the run stamp in every checkpoint. An audit that cries wolf gets ignored, so each check now separates expected variation from real damage.

### Curated corpora on disk

~149GB, fetched 2026-09-18, don't re-download: `data/cosmopedia/` (336 parquet, 86 GiB), `data/finewiki/` (15 parquet, 36 GiB, en only), `data/open_web_math/` (114 parquet, 26 GiB). Fetch script: `scripts/fetch_corpus.py` (validates patterns against the real file list first; resumable). Next big target if we want volume: FineWeb-Edu sample-100BT (267GB, ~100B tokens) — but note the cost: 4 bytes/token means a 373GB file and ~65h of CPU encode single-core (~8h with 8 workers).

### Viz passes 1–5 (Sep 26, pre-v2 board history — v2 keeps the ideas)

Passes 1–4 (bpb-first, null-gap rendering, LR strip + guard line, three-way
panel, noise band, throughput cards, token axis, coverage epochs) were built
against the old board and TDD'd (`tests/test_dashboard.py`). The v2 decision
board (registry arms, stability/loader panels) supersedes the layout but keeps
the substance: bpb-first, guard-abort marker, loader recycling panel, token
axis — plus epochs (`axis: epochs`, `@tok (x.xxep)` in the arms table) from
the `train_tokens` run-header field. Old-board details below for the record:
`viz/dashboard.py` rebuilds `viz/training.html` (offline, zero deps, inline JSON).
Two passes landed, TDD'd (`tests/test_dashboard.py`, 5 tests):

**Pass 1 — bpb-first + null-gap rendering:**
- Default series `val_bpb` (fair across vocabs; raw loss lied in exp 011).
  Toggle order: val_bpb, val, avg50, train.
- Null spans render red dashed `eval gap A-B` on the axis; trailing null
  renders `eval dead from N` (the 160M @3901 signature). Line end vs
  deliberate stop now look different.
- Early-stop marker: green tick + best bpb (deliberate end, not a crash).

**Pass 2 — LR strip + guard line + three-way panel:**
- LR strip (default on): peak-normalised schedule fill under the top edge.
  Flat val at high LR = schedule, keep going. Flat val at decayed LR = knee.
  (The 011 schedule confusion, made visible.)
- Guard line: red dashed at best bpb x (1 + degrade_frac) from the run header,
  val_bpb mode only. Abort threshold visible against the wobble — calibrate by eye.
- 3-way toggle (default off): yellow dashed served bpb + teal dashed
  random-train bpb. Tight = loader healthy (like the 160M: 1.66 vs 1.63).
  Served diving = recycling, kill the run.

Buttons: `series`, `linear/log`, `lr on/off`, `3-way on/off`. Per-run checkboxes,
sample viewer unchanged.
Rebuild: `uv run python viz/dashboard.py [--watch N] [--runs runs]`.
NOTE: local runs lack bpb/three-way fields (those curves live on Ronin);
a fresh local build stays thin until Ronin curves land. `training.html` not rebuilt.

**Pass 3 — noise band (Sep 26):**
- Blue shaded band around best bpb: median check-to-check change x4 each side.
  Guard line should sit well clear of the wobble. Overlap = miscalibrated guard
  (the 011b kill: threshold inside noise; genuine divergence runs ~30x noise).
- `noise_band(vals)` helper in dashboard.py, skips nulls. 2 more tests, 10 green.

**Pass 4 — training finish: throughput + equal-token axis + alerts (Sep 26):**
- Trainer writes a `{"throughput": true, "tok_per_sec", "peak_mem_mb"}` row
  after the warmup window. Parser attaches it to the nearest step.
  Status cards show rate + peak mem + ETA (from tokens_seen delta).
  MPS reports rate only (unified memory, no separate gauge).
- X-axis toggle: step vs tokens-seen (step x batch x accum x ctx from header).
  Step lies across vocabs/batches. Tokens never lie. All overlays (curves,
  gaps, stops, three-way) follow the axis.
- Alerts per run: GUARD TRIPPED / guard near (val_bpb vs degrade_frac),
  EVAL DEAD (trailing nulls), RECYCLING? (served vs random-train gap).
- Smoke-proven: s17m 135-step MPS run, 9.3k tok/s attached correctly.
  `noise_band` helper + 7 dashboard tests, 12 green total.
