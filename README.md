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

    bash setup.sh

This creates the venv, syncs deps, and installs CUDA torch on Windows.
Mac just needs `uv sync` (setup.sh handles both).

To verify CUDA: `uv run python -c "import torch; print(torch.cuda.is_available())"`

Machines
--------
- PC (Ronin, 4070 Ti Super)  — primary dev/training (CUDA)
- Mac (MacBook Pro, MPS)      — training/inference
- Mini (Mac Mini, Win10)      — inference/testing

Experiments
-----------

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
Same bpe2k+rope recipe, 4x the steps. Result: val peaked at 2.086 @ step 4500,
then climbed monotonically to 2.89 by step 20k. Train kept learning; generalization
peaked early. Discovered early stopping empirically.

Cross-machine reproducibility confirmed (Sep 5, 2026):

| Machine | Hardware    | Time (20k steps) | Train Loss | Val Loss (best) | Val Loss (20k) |
|---------|-------------|-------------------|------------|-----------------|----------------|
| PC      | 4070 Ti     | 31 min            | 0.8717     | 2.0858          | 2.8924         |
| Mac     | MPS         | ~2h 15m           | 0.8934     | ~2.09           | ~2.89          |

Val loss matches to 4 decimal places across CUDA and MPS. Same overfit curve.
4.3x speedup on PC vs Mac.

### 006 — Dropout + Early Stopping
Status: DONE
Wired `ModelConfig.dropout` (was dead config) + `--patience` / `--min_delta` early stopping.
Two runs:

| Run         | Dropout | Steps | Best Val | Step | Train (final) | Notes |
|-------------|---------|-------|----------|------|---------------|-------|
| 20260905_1910 | 0.1   | 5k    | 2.1626   | 4900 | 1.9885        | Short horizon — dropout slightly worse (expected) |
| 20260913_1007 | 0.1   | 4.9k  | 2.154    | 4900 | 1.8809        | Clean early stop, no overfit |

Verdict: dropout 0.1 trades a tiny bit of peak val (2.15 vs 2.09 no-dropout) for
flat val curve — no climb back to 2.89. That's exactly what dropout is supposed to do.
Early stopping fires correctly: patience=5 on val checks without improvement.

### 007 — Book Era First Crunch (LibriSpeech)
Status: DONE (Sep 14, 2026 — Ronin 4070 Ti)
Config: libri10m preset — vocab 8256 (bpe8k clean), ctx 512, embed 384, 6L6H,
dropout 0.1, ~10M params. 20k-step budget, early stopped.
Result: early stop @ step 6800, best val 4.84, train 9.18→3.72.
Verdict: pipeline works end-to-end on real literature (193M tokens). Val 4.84 vs
TinyStories 2.15 is expected — books are a much harder corpus. Train/val gap
(3.72/4.84) says capacity wall, not broken code. Next: go bigger (008).

### 008 — TinyStories 50M Control
Status: DONE (Sep 14, 2026 — Ronin 4070 Ti)
Config: bpe50m preset — vocab 2256 (bpe2k), ctx 256, embed 640, 10L10H,
dropout 0.1, ~52M params. 20k-step budget, full run.
Result: step 20000, train 0.003, val 5.89.
Verdict: massive overfit — model memorized TinyStories completely but didn't
generalize (val 5.89). 50M is too much model for TinyStories. Dropout kept it
from collapsing but can't prevent memorization on a small, simple corpus.
This run confirms 50M architecture is functional on CUDA. The real test is
Gutenberg (009) where data volume matches the capacity.

### 009 — Gutenberg 50M (LibriSpeech)
Status: QUEUED
Config: libri50m preset — vocab 8256 (bpe8k), ctx 512, embed 640, 10L10H,
dropout 0.1, ~52M params. 193M tokens from Gutenberg books.
Hypothesis: 50M should generalize better on 193M tokens than TinyStories —
more data matches more capacity. Val should land below 4.84 (007's 10M result).

### 010 — M50M Full OpenWebText (4k vocab)
Status: DONE (Sep 17, 2026 — Ronin 4070 Ti)
Config: m50m preset — vocab 4512 (bpe_owt4k), ctx 512, embed 640, 10L10H,
dropout 0.1, ~55M params. Full OpenWebText (~4.8B tokens).
Budget: 20k steps, batch 32, LR 3e-4. No early stopping (patience=0).
Result: best val 4.0986 @ step 2600. Process died at step ~17,050 (likely OOM or
crash). Final state: train 0.10, val 9.98 — completely overfit.
Verdict: 4k vocab is too small for 55M params. Model runs out of embedding
resolution early and memorizes training data. Best val (4.09) is the same as
every other vocab size — the embedding layer is the bottleneck, not data or
architecture. Next: try 16k vocab (experiment 011).
KEEPER: checkpoints/exp002_m50m_rope_bpe_owt4k_202609172151_step2500.pt
LESSON: **Early stopping must be schedule-aware, not a check count.** 010 ran with
patience=0 and memorised for hours; the first fix (patience=10 checks) was still
wrong in the other direction — 10 checks x val_every 100 = 1000 steps = 5% of a
20k cosine schedule, so a mid-schedule plateau reads as convergence. Use
`--patience-frac` (a fraction of --steps) with `--min-steps-frac` so no plateau
stop can fire while the LR is still near peak. See the 011 results section.

### 011 — M50M Full OpenWebText (16k vocab)
Status: TOKENIZING (Sep 18, 2026 — Ronin)
Config: same as 010 but with bpe_owt16k tokenizer (16,256 vocab). ~70M params
(10M of that is the bigger embedding table).

Pipeline (rewritten Sep 18 — the first three attempts all died):
  run_16k_smoke.bat   rehearsal: 3 shards x 20M chars -> verify -> 200 train steps (~2.5 min)
  run_16k_full.bat    the real thing: 80 shards, serialized, resumable (~7h)
  tokenize_owt_16k.py orchestrator (manifest, resume, preflight disk check)
  token_io.py         streaming encoder + memmap reader (no %TEMP%, no RAM blowup)
  verify_tokens.py    7 acceptance checks — run before trusting any long encode
Train with the .bin:  --tok_cache data/openwebtext_combined_bpe_owt16k.bin

What broke before, and why it can't now:
  * tempfile.mkdtemp() put per-chunk .npy files on C: -> writes only under data/ now
  * torch.cat() of all 80 shard tensors needed ~80GB RAM on a 34GB box -> the
    output is appended shard by shard; peak RAM is one 500K-char chunk (~4MB)
  * torch.load() of a 47GB .pt is how run 010 died at step ~17k -> .bin token
    files are memmapped, so the corpus streams from disk at 0 RAM
  * loss.jsonl was unbuffered-free, so a watcher saw an empty log -> flushed per step

Smoke evidence (Sep 18): encode 40s (370k tok/s), verify 7/7, resume skip all 3
shards in 0.7s, 200 train steps in 81s (0.4s/step -> 20k steps ~2.2h on the 4070 Ti).
NOTE: that 0.4s/step baseline did not hold for the full run — 011b measured
1.20s/step (3.2x slower, ~5.8h for 17.5k steps), most likely GPU contention with
the desktop (15.8/16 GB in use). Always estimate from a measured rate, not from a
smoke run on an idle GPU.

### 011 results — and a measurement error that inverted the verdict

**The per-token comparison was invalid.** 011 stopped at step 3300 with best val
4.8604 against 010's 4.0986, which reads as "16k is 18.6% worse". It is not: a
16k BPE emits FEWER, more informative tokens (0.2463 vs 0.2976 tok/byte, measured
on the same held-out tail text), so its per-token loss is mechanically higher at
identical compression quality. Convert with
`bits/byte = val_nats * log2(e) * tokens_per_byte`:

| run | vocab | tok/byte | best val (per-token) | bits/byte |
|-----|-------|----------|----------------------|-----------|
| 010 | 4k    | 0.2976   | 4.0986               | 1.7596    |
| 011 | 16k   | 0.2463   | 4.8604               | 1.7271    |

**16k was ahead all along**, by 1.8% — and the in-run metric agrees: 011b's
byte-normalised val hit 1.7013. `train/train.py` now reports and DECIDES on
`val_bpb` (loss.jsonl rows carry both `val` and `val_bpb`), and `--min_delta`
applies to bpb.

**The early stop also fired for the wrong reason.** Patience was `10 checks x
val_every 100` = 1000 steps = 5% of a 20k cosine schedule, so 011 stopped at step
3300 while the LR was still 2.8e-4. A flat val there is the schedule, not
convergence — which also leaves 010's "the embedding layer is the bottleneck"
conclusion unproven. Patience is now a FRACTION of --steps (`--patience-frac`),
and no plateau stop can fire before `--min-steps-frac` of the schedule.

**Calibrate the mid-schedule abort from the noise band.** 011b reached best
bpb 1.7013 and was then killed at step 3700 by `bpb 1.7870 > best +5%`. That
threshold sat inside the noise: the measured val series swings a median 5.2%
between consecutive checks (max 10.4%), with excursions up to 5.0% above the
running best from nothing but noise, while genuine memorisation (010) was +143%.
Noise and divergence are ~30x apart, so `--degrade-frac` now defaults to 0.30.
LESSON: never set an abort threshold by feel — measure the noise band first, or
you will abort healthy runs and blame the model.

011c resumes from 011b's keeper (step 2500, val 4.7666 / bpb 1.7013) with
`--patience-frac 0.15 --min-steps-frac 0.6 --degrade-frac 0.30`, so the earliest a
plateau can stop it is step 15000, where the LR has decayed to ~7e-5 from 3e-4.
Launcher: `run_exp011c_train.bat` — its header documents the git-bash invocation
trap (`cmd //c foo.bat` silently does nothing; use
`MSYS_NO_PATHCONV=1 cmd.exe /c "foo.bat"`) and the interpreter trap (bare `python`
is the CUDA venv; `llm-lab/.venv` is CPU-only torch and will train 50x slower
without ever looking wrong).

### 011c — finished. best bpb 1.5854; and the run proves the guard works

**Result:** best **bpb 1.5854** (val 4.4418) at **step 5500** — 90M tokens, **0.93% of the
9.65B-token corpus**. That is the best byte-normalised number on record:

| run | vocab | best bpb |
|-----|-------|----------|
| 010 | 4k    | 1.7596   |
| 011 | 16k   | 1.7271   |
| 011b| 16k   | 1.7013   |
| **011c** | **16k** | **1.5854** |

So 16k/OWT is the right line, and the gap to 4k is now ~10%, not the 1.8% the first
comparison suggested.

**It ended by its own rule, correctly.** At step 8200 the degrade guard fired:
`abort: bpb 2.0975 > best 1.5854 +30% (mid-schedule)`. That is not a crash — the
trainer exits 0 and writes `{"early_stop": true}` into the curve. The val series shows a
*relentless* climb, not a wobble:

| step | bpb | % over best |
|------|-----|-------------|
| 6900 | 1.8599 | +17.3% |
| 7700 | 2.0181 | +27.3% |
| 8200 | 2.0975 | **+32.3%** |

Consecutive check-to-check noise across the whole run: median **1.03%**, p90 3.05%, max
3.75%. A 14-check monotonic climb of 15 points is nowhere near that band, so the abort was
right — arguably late.

**Two conclusions, one of them about schedule length.**
1. **The knee is ~5,500 steps, i.e. 27% of the planned 20k.** This regime overfits at
   ~1% corpus coverage; the remaining 11,800 steps would have produced nothing. The 20k
   schedule was ~3.7x too long for m50m/OWT-16k at this LR.
2. **A patience rule cannot catch overfitting.** It fires on *flat* val; when val is
   climbing you need the degrade guard. Next run: `--degrade-frac 0.15` — still ~4x the
   measured noise ceiling (~3.75%), and it would have stopped this run ~2,700 steps earlier.

**Operational notes worth keeping.** The run had been killed mid-flight at step 5170 by an
app restart (it was a child of the Hermes app process tree) and was resumed from its own
step-5000 checkpoint with weights + optimizer + step intact — the resume also inherited the
early-stop ledger and the run identity, so the curve stayed one continuous file
(`train/train.py`, commit `3689d12`). The supervisor's watchdog initially would have
RELAUNCHED this finished run (it only understood "complete" as "reached target"); it now
treats the `early_stop` record or the `done. final` line as a deliberate end (commit
`552f55c`).

## Current state (2026-09-18) — read this first, it's the handoff

**16k OWT corpus — DONE (2026-09-18 15:11, unattended):** 9,745,672,850 tokens
(38.98GB int32), 80/80 shards, verify **7/7 green** (shape, manifest delta +0, id range,
round-trip decode, shard-boundary offsets at 0 / 40 / 79, memmap loadability).
The waiter filled the dead worker's shards (33-42) and merged in corpus order with
nobody watching — see `logs/followup.log` and `logs/16k_tokenize_followup.log`.
NOTE: the first encode exited 1 on purpose (refusing to merge with shards missing).

**16k OWT corpus (original notes):** `data/openwebtext_combined_bpe_owt16k.bin` (raw int32, ~38GB, ~9.46B
tokens expected). Encoded in parallel (8 worker processes, `run_16k_full_par.bat`),
~1.6M tok/s aggregate.

One worker died mid-run to an unreproducible fault in the BPE encoder
(`TypeError: slice indices must be integers` in model/bpe.py). Its block (shards
33-42) is therefore missing and the run **deliberately refuses to merge**. The
encoder is now hardened (`sentinel -1` + `token_io.encode_resilient` bisect with
evidence capture in `<dst>.encode_errors.jsonl`) and the parent reports worker
deaths immediately. Commit `9a41716`.

Following up unattended: `followup_wait.ps1` holds until the encode exits, then runs
a second pass that reads the merge plan off the parts on disk, encodes only the gap,
merges in corpus order and verifies. **Outcome is one line in `logs/followup.log`**
(VERIFIED OK / VERIFY FAILED / ENCODE INCOMPLETE).

Check progress:  `tail -2 logs/16k_tokenize_par.log`   (status lines every 60s)
Rerun by hand:   `run_16k_full_par.bat`   (resumable — finished shards are skipped)
Verify:          `python verify_tokens.py --bin data/openwebtext_combined_bpe_owt16k.bin --vocab data/bpe_owt16k.json --src data/openwebtext/shards/train-00000-of-00080.txt --manifest`
Tests:           `python -m pytest tests/test_tokenizer_safety.py tests/test_bpe_resilience.py -q`  (16 tests)

**exp 011 line — DONE (2026-09-18 20:43): best bpb 1.5854 @ step 5500, guard fired at
step 8200** (see the 011c section above). The line's question is answered: 16k beats 4k by
~10% byte-normalised, and the model overfits at ~1% corpus coverage, so the knee is ~5,500
steps and a 20k schedule is ~3.7x longer than this regime can use.

**Next levers (the overfit finding points away from "more steps"):** the constraint at 0.93%
coverage is generalisation, not budget — so the next experiments are about per-step data
diversity and regularisation (dropout is already wired), and about a `--degrade-frac 0.15`
guard so an overfitting run stops near its knee instead of 2,700 steps past it.

**Disk / retention (was the next wall, now handled):** 338 ckpts / 181 GB became **87 files
/ 35 GB** — `prune_checkpoints.py` keeps every `_best.pt`, the run's last step-ckpt, and
every 5000th step; it freed **143.7 GB** (D: 540 → 667 GB free) and refuses to run if any
keeper lacks a verified copy.

**Keeper backups — finally OFF-machine:** the 18 keepers (6.70 GB) now exist in four places:
the working dir, `llm-lab-private/checkpoints/keepers/`, a second physical SSD, and — the
one that survives the machine being lost — **Google Drive** (`llm-lab keepers`). The first
Drive attempt hit `storageQuotaExceeded` on the 16 GB free tier; the account has since moved
to 100 GB. Worth knowing what was actually filling it: not the keepers (5.07 GB) but **7.35 GB
of two `lmu_session_*.jsonl` telemetry files** plus 1.27 GB of a retokenized corpus cache that
already exists locally. Off-machine also means the upload order matters — see below.

### Data integrity audit — 0 errors, and three findings that only LOOK like corruption

`audit_run_data.py` (new) checks curve integrity, run identity, checkpoint/step agreement,
keeper hashes and hygiene across the whole lab: **31 run dirs, 31 curves, 87 checkpoints in
39 families, 18/18 keeper hashes verified — 0 errors, 7 warnings.**

The seven warnings are honest ones, and getting to them required fixing the *audit* three
times, because each of these is expected behaviour that a naive check reports as damage:

| looks like | actually is |
|---|---|
| duplicate steps + schema drift on 7 runs | the `early_stop` **summary row**, which deliberately reuses the final step number. Exclude it from the data set and validate it separately. |
| 37 "orphan" checkpoints | run dirs are stamped `YYYYMMDD_HHMM`; checkpoint families `YYYYMMDDHHMM`. Compare digits, not strings. |
| 16 errors on 9 runs | **abandoned stubs** — 0-byte or header-only curves from the failing dense111m/194m attempts. Parked in `runs/_abandoned_stubs_2026-09-15/` (6 KB, nothing lost). |

What remains is real and worth knowing: **6 checkpoint families have weights but no run dir
anywhere** (the Sep 5/11/13 bpe2k + libri10m runs — their curves live on the Mac or in Drive,
not here), and **1 stamp-drift case** where one logical run carries two stamps
(`..._202609181300` family vs the `20260918_1307` run dir — 7 minutes apart). That drift is
exactly why the resume-continuity fix persists the run stamp in every checkpoint.

An audit that cries wolf gets ignored, so each check now separates expected variation from
real damage.

**Curated corpora on disk** (~149GB, fetched 2026-09-18, don't re-download):
`data/cosmopedia/` (336 parquet, 86 GiB), `data/finewiki/` (15 parquet, 36 GiB, en
only), `data/open_web_math/` (114 parquet, 26 GiB). Fetch script: `fetch_corpus.py`
(validates patterns against the real file list first; resumable). Next big target if
we want volume: FineWeb-Edu sample-100BT (267GB, ~100B tokens) — but note the cost:
4 bytes/token means a 373GB file and ~65h of CPU encode single-core (~8h with 8 workers).
