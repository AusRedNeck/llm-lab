# Scripts inventory

This is a navigation map, not a promise that every script is current. Commands run from the repository root unless their usage says otherwise. Paths remain unchanged: launchers, tests, cron jobs, and docs refer to them directly.

## Operator / training lifecycle
- `train_watchdog.py`, `train_launch.py`, `check_run_health.py`, `watch_active_arm.py` — supervision, detached launch, run health.
- `audit_run_data.py`, `prune_checkpoints.py`, `keeper_backup.py`, `upload_keepers_to_drive.py` — integrity and checkpoint retention/backup.
- `readme_state.py` — generated README state; use `--check` to validate.
- `kill_downloads.py`, `status_downloads.py` — download-process inspection/cleanup.

## Scoring and evaluation
- `score_matched_tokens.py`, `score_ours_vs_reference.py`, `score_pythia70m_reference.py`, `val_fixture.py`, `verify_val_slice_identity.py` — current comparison and validation-fixture path.
- `eval_diag.py`, `eval_frozen_ladder.py`, `eval_frozen_ladder_exp015.py`, `sample_sufficiency.py`, `batch_ladder_smokes.py`, `confirm_sawtooth_bias.py` — experiment-specific evaluation.
- `retired/diagnose_eval_gap.py`, `retired/eval_variance_study.py`, `extract_val_slice.py` — historical/retired methods; read headers first. `extract_val_slice.py` warns not to rerun.

## Corpus acquisition and preparation
- `fetch_corpus.py`, `fetch_corpora.bat`, `download_clean_corpora.bat`, `download_cosmopedia_owm.bat`, `scan_corpus_sizes.py`, `scan_corpus_sizes2.py` — fetch and inspect corpora.
- `convert_openwebtext.py`, `clean_gutenberg.py`, `stream_pile.py`, `stream_slimpajama.py` — conversion, cleaning, streaming.
- `audit_owt_consist.py`, `audit_owt_dedup.py`, `audit_owt_probe.py`, `audit_owt_quick.py` — OWT-specific audits, not general entry points.

## Tokenizers and token caches
- `token_io.py`, `tokenize_owt_16k.py`, `tokenize_stream.py`, `tokenize_pile_parallel.py`, `verify_tokens.py` — shared/current streaming and cache-validation tools.
- `tokenize_chunk.py`, `tokenize_one.py`, `tokenize_fineweb.py`, `tokenize_gptneox.py`, `encode_owt_50k.py`, `split_chunks.py`, `split_fineweb.py`, `merge_chunks.py`, `sprinkle_tokenize.py`, `reencode_owt1gb.py` — dataset/tokenizer-specific tools; check documented inputs.
- `convert_hf_to_bpe.py`, `hf_bpe_shim.py`, `save_gptneox_vocab.py`, `train_gpt_neox_tokenizer.py`, `train_tokenizer_openwebtext.py` — tokenizer conversion/training.

## Launchers and experiment-specific utilities
- `run_train.bat`, `run_train_hidden.vbs`, `start_training.bat`, `run_pythia70_pile_full.bat`, `run_16k_full.bat`, `run_16k_full_par.bat`, `run_16k_smoke.bat`, `run_tokenizer_16k.bat`, `run_m50m_full.bat`, `run_m50m_sprinkle.bat`, `run_m50m_20k.bat`, `run_m50m_5k.bat`, `run_m50m_4k.bat`, `run_m50m_50k_train.bat`, `run_s17m_4k.bat`, `run_exp011_train.bat`, `run_exp011b_train.bat`, `run_exp011c_train.bat`, `run_exp011c_resume.bat`, `run_exp011c_resume_hidden.vbs`, `run_lr_rungs_20260922.bat` — retained operator launchers; inspect arguments before use.
- `collapse_epoch_arm_20260930.py`, `doc_fifth_stop_20261010.py`, `register_backlog_2026-09-29.py`, `followup_wait.ps1`, `repro_bpe_crash.py`, `generate_qa.py`, `generate_qa_llm.py`, `setup.sh`, `smoke_50k.py`, `smoke_train_50k.py`, `dashboard_server.py` — purpose-specific; some are historical or tied to local data/services. Verify prerequisites.

## Cleanup status / safe next step
Two evaluation scripts have been moved into `scripts/retired/`: their headers mark them historical, and no direct invocation was found in repository files or current Hermes cron jobs. They remain available for reproducing prior analyses. `extract_val_slice.py` remains in the top-level scripts folder because ROADMAP and the fixture verifier cite it as the example of a known-invalid method; do not bury or remove it casually.

A read-only Windows Task Scheduler check found two stale launch records: `Hermes_Exp011cResume` targeted the missing repo-root `run_exp011c_resume_hidden.vbs` (the tracked file was under `scripts/`), and `Hermes_Pythia70v2_10pm` targeted missing `run_pythia70_v2_10pm.bat`. Both tasks were removed after re-querying and matching their actions. The active `Hermes_TrainRun` task was verified to point to the existing `scripts/run_train_hidden.vbs` and left intact.

Before archiving any script, search names across source, tests, docs, `.hermes/`, and scheduler configuration, then confirm external usage where possible. The root `scripts/` directory stays flat for now because its paths are part of the operational interface.