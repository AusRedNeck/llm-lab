# llm-lab launch quarantine — 2026-09-24

Status: **NOT RUNNING. DO NOT LAUNCH.**

## Incident

The same `python -u -m train.train ...` command was dispatched about five times in roughly 30 seconds. The previous session checked a Hermes background-wrapper PID rather than the actual Python trainer PID, treated each wrapper as a dead launch, and retried. The resulting trainers all targeted the same 16 GB GPU and the same log/run naming scheme. The user reported that the PC locked during the overlap.

## Evidence

- Current trainer count: 0.
- GPU VRAM: approximately 750 MiB of 16,376 MiB at audit time.
- Collision debris:
  - `runs/20260924_2059_pythia160_bpe_owt50k_v2_openwebtext_combined`: stopped at step 118; invalid.
  - `runs/20260924_2101_pythia160_bpe_owt50k_v2_openwebtext_combined`: 0-byte `loss.jsonl`; invalid.
- Neither directory is registered as an experiment; both are in `experiments.json.ignore_dirs`.
- `train_job.json` is quarantined/completed with `max_restarts=0`.
- The Hermes train watchdog and active-arm watcher crons are paused.
- The Windows `Hermes_TrainRun` scheduled task is disabled.

## Launch-safety changes

1. `train/runtime_lock.py` provides an OS-held byte-range lock. It is acquired immediately after argument parsing and before CUDA initialization.
2. `scripts/train_watchdog.py` performs a global trainer preflight before every direct launch. It refuses to launch if any Python process is running `-m train.train`, regardless of the active job match. A blind probe also fails closed.
3. A quarantined state is terminal and cannot relaunch.
4. Regression tests cover lock contention, refusal before CUDA, release after a training exception, global process detection, blind/blocked launch behavior, and quarantine behavior.

## Experimental validity

The attempted `7.5e-4` spec was not a clean LR-only comparison against the archived control:

- proposed effective batch was 16; the control used effective batch 64;
- it omitted `--use_rope` and `--rotary-pct 0.25`;
- its guard/evaluation settings differed;
- it was overwritten by repeated launch attempts.

A valid next probe requires matched fixed-stack arms at the same schedule, corpus, tokenizer, effective batch, RoPE settings, evaluation cadence, and guards. If the question is still `5e-4` versus `7.5e-4`, run both as a matched pair; do not compare a 1k candidate to a 20k control.

## Verification

- `python -m pytest -q`: 98 passed, 8 pre-existing `torch.load(weights_only=False)` warnings.
- `python viz/registry.py --check`: PASS.
- Watchdog dry run under project venv: no-op.
- Watchdog dry run through Hermes cron entry point: no-op.
- Live process preflight under both interpreters: no trainers.
