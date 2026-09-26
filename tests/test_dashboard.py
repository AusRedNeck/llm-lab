import json

from viz.dashboard import load_runs, noise_band


def _write_run(tmp_path, name, rows):
    d = tmp_path / name
    (d / "samples").mkdir(parents=True)
    with open(d / "loss.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({"args": {"preset": "s17m"}, "cfg": {}, "params_m": 17.0}) + "\n")
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return d


def test_load_runs_includes_val_bpb(tmp_path):
    _write_run(tmp_path, "run_a", [
        {"step": 100, "train": 5.0, "avg50": 5.1, "val": 4.86, "val_bpb": 1.7271, "lr": 3e-4},
        {"step": 200, "train": 4.8, "avg50": 4.9, "val": 4.76, "val_bpb": 1.7013, "lr": 2.8e-4},
    ])
    runs = load_runs(str(tmp_path))
    assert len(runs) == 1
    assert runs[0]["val_bpb"] == [1.7271, 1.7013]


def test_load_runs_preserves_null_val_gap(tmp_path):
    _write_run(tmp_path, "run_gap", [
        {"step": 100, "train": 3.5, "avg50": 3.5, "val": 4.96, "val_bpb": 1.8, "lr": 4.6e-4},
        {"step": 200, "train": 3.49, "avg50": 3.49, "val": None, "lr": 4.6e-4},
        {"step": 300, "train": 3.49, "avg50": 3.49, "val": None, "lr": 4.6e-4},
    ])
    runs = load_runs(str(tmp_path))
    assert len(runs) == 1
    assert runs[0]["val"] == [4.96, None, None]
    assert runs[0]["val_bpb"][1] is None


def test_load_runs_captures_early_stop(tmp_path):
    _write_run(tmp_path, "run_stop", [
        {"step": 100, "train": 5.0, "avg50": 5.1, "val": 4.8, "val_bpb": 1.7, "lr": 3e-4},
        {"early_stop": True, "step": 8200, "best_bpb": 1.5854},
    ])
    runs = load_runs(str(tmp_path))
    assert len(runs) == 1
    assert runs[0]["early_stop"]["step"] == 8200
    assert runs[0]["early_stop"]["best_bpb"] == 1.5854


def test_load_runs_harvests_three_way(tmp_path):
    _write_run(tmp_path, "run_3way", [
        {"step": 100, "train": 5.0, "avg50": 5.1, "val": 4.8, "val_bpb": 1.7,
         "served": 4.7, "served_bpb": 1.66, "random_train": 4.75,
         "random_train_bpb": 1.68, "lr": 3e-4},
        {"step": 200, "train": 4.8, "avg50": 4.9, "val": 4.76, "val_bpb": 1.69,
         "served": 4.65, "served_bpb": 1.64, "random_train": 4.7,
         "random_train_bpb": 1.66, "lr": 2.8e-4},
    ])
    runs = load_runs(str(tmp_path))
    assert len(runs) == 1
    assert runs[0]["served_bpb"] == [1.66, 1.64]
    assert runs[0]["random_train_bpb"] == [1.68, 1.66]


def test_load_runs_keeps_guard_args(tmp_path):
    d = tmp_path / "run_guard"
    (d / "samples").mkdir(parents=True)
    with open(d / "loss.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({"args": {"preset": "m50m", "degrade_frac": 0.30,
                                     "min_steps_frac": 0.6},
                            "cfg": {}, "params_m": 55.0}) + "\n")
        f.write(json.dumps({"step": 100, "train": 5.0, "avg50": 5.1,
                            "val": 4.8, "val_bpb": 1.7, "lr": 3e-4}) + "\n")
    runs = load_runs(str(tmp_path))
    assert len(runs) == 1
    assert runs[0]["args"]["degrade_frac"] == 0.30


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


def test_load_runs_harvests_throughput(tmp_path):
    _write_run(tmp_path, "run_tps", [
        {"step": 100, "train": 5.0, "avg50": 5.1, "val": None, "lr": 3e-4,
         "tok_per_sec": 42000.0, "peak_mem_mb": 12500.0},
        {"step": 200, "train": 4.8, "avg50": 4.9, "val": 4.76, "val_bpb": 1.69, "lr": 2.8e-4},
    ])
    runs = load_runs(str(tmp_path))
    assert len(runs) == 1
    assert runs[0]["tok_per_sec"] == [42000.0, None]
    assert runs[0]["peak_mem_mb"] == [12500.0, None]


def test_load_runs_computes_tokens_seen(tmp_path):
    d = tmp_path / "run_tok"
    (d / "samples").mkdir(parents=True)
    with open(d / "loss.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({"args": {"preset": "pythia160", "batch": 8, "accum": 2},
                            "cfg": {"context_length": 512}, "params_m": 162.6}) + "\n")
        f.write(json.dumps({"step": 100, "train": 5.0, "avg50": 5.1, "lr": 3e-4}) + "\n")
        f.write(json.dumps({"step": 200, "train": 4.8, "avg50": 4.9, "lr": 2.8e-4}) + "\n")
    runs = load_runs(str(tmp_path))
    assert len(runs) == 1
    # tokens_seen = step * batch * accum * ctx
    assert runs[0]["tokens_seen"] == [100 * 8 * 2 * 512, 200 * 8 * 2 * 512]
