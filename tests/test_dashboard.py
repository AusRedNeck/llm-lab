"""Tests for viz/dashboard.py v2 (decision board) + coverage epochs graft.

v2 shape: load_runs -> run dicts with dense `rows` ({step, tokens, ...}),
run-level eff/ctx/train_tokens, best_bpb/best_step/stop_step. Epochs =
tokens / train_tokens (header field from train.py).
"""
import json

from viz.dashboard import build_arms, load_runs, noise_band


def _write_run(tmp_path, name, header_extra=None, rows=()):
    d = tmp_path / name
    (d / "samples").mkdir(parents=True)
    header = {"args": {"preset": "s17m", "batch": 32, "accum": 1},
              "cfg": {"context_length": 512}, "params_m": 17.0}
    header.update(header_extra or {})
    with open(d / "loss.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps(header) + "\n")
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return d


def test_rows_carry_tokens_per_step(tmp_path):
    _write_run(tmp_path, "run_a", rows=[
        {"step": 100, "train": 5.0, "avg50": 5.1, "val": 4.86, "val_bpb": 1.7271},
        {"step": 200, "train": 4.8, "avg50": 4.9, "val": 4.76, "val_bpb": 1.7013},
    ])
    runs = load_runs(str(tmp_path))
    assert len(runs) == 1
    toks = [e["tokens"] for e in runs[0]["rows"]]
    assert toks == [100 * 32 * 512, 200 * 32 * 512]


def test_null_val_gap_preserved(tmp_path):
    _write_run(tmp_path, "run_gap", rows=[
        {"step": 100, "train": 3.5, "avg50": 3.5, "val": 4.96, "val_bpb": 1.8},
        {"step": 200, "train": 3.49, "avg50": 3.49, "val": None},
    ])
    runs = load_runs(str(tmp_path))
    assert runs[0]["rows"][1].get("val") is None
    assert runs[0]["best_bpb"] == 1.8
    assert runs[0]["best_step"] == 100


def test_early_stop_row_marks_stop_step(tmp_path):
    _write_run(tmp_path, "run_stop", rows=[
        {"step": 100, "train": 5.0, "avg50": 5.1, "val": 4.8, "val_bpb": 1.7},
        {"early_stop": True, "step": 8200, "best_bpb": 1.5854},
    ])
    runs = load_runs(str(tmp_path))
    assert runs[0]["stop_step"] == 8200


def test_train_tokens_passthrough_for_epochs(tmp_path):
    _write_run(tmp_path, "run_cov",
               header_extra={"train_tokens": 536786585,
                             "corpus_tokens": 542208672},
               rows=[
                   {"step": 100, "train": 5.0, "avg50": 5.1,
                    "val": 4.8, "val_bpb": 1.7},
               ])
    runs = load_runs(str(tmp_path))
    r = runs[0]
    assert r["train_tokens"] == 536786585
    # epochs math the JS does: tokens / train_tokens
    ep = r["rows"][-1]["tokens"] / r["train_tokens"]
    assert ep == 100 * 32 * 512 / 536786585


def test_missing_train_tokens_means_no_epochs(tmp_path):
    _write_run(tmp_path, "run_old", rows=[
        {"step": 100, "train": 5.0, "avg50": 5.1, "val": 4.8, "val_bpb": 1.7},
    ])
    runs = load_runs(str(tmp_path))
    assert runs[0]["train_tokens"] is None


def test_arms_row_carries_epochs_denominator(tmp_path):
    _write_run(tmp_path, "run_arm",
               header_extra={"train_tokens": 1000000},
               rows=[
                   {"step": 100, "train": 5.0, "avg50": 5.1,
                    "val": 4.8, "val_bpb": 1.7},
               ])
    runs = load_runs(str(tmp_path))
    by_name = {r["name"]: r for r in runs}
    arms, _problems = build_arms([{"id": "a1", "family": "f",
                                    "run_dirs": ["run_arm"]}], by_name)
    assert arms[0]["train_tokens"] == 1000000
    assert arms[0]["best_tokens"] == 100 * 32 * 512


def test_noise_band_measures_check_to_check():
    # Steady climb: median step change small vs total rise.
    vals = [1.70, 1.71, 1.70, 1.72, 1.71, 1.73, 1.74, 1.75]
    band = noise_band(vals)
    assert band["median"] < 0.02
    assert band["max"] < 0.03
    assert band["n"] == 7


def test_noise_band_ignores_nulls():
    vals = [1.70, None, 1.72, None, 1.71]
    band = noise_band(vals)
    assert band["n"] == 2
