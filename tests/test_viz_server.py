import json
import threading
from urllib.request import urlopen

from viz.server import create_server, dashboard_payload


def _fixtures(root):
    runs = root / "runs"
    run = runs / "20261006_demo"
    run.mkdir(parents=True)
    (run / "loss.jsonl").write_text(
        json.dumps({"args": {"preset": "pythia", "batch": 2, "accum": 1},
                    "cfg": {"context_length": 8}, "params_m": 0.1}) + "\n"
        + json.dumps({"step": 1, "train": 2.0, "avg50": 2.1,
                      "val": 2.2, "val_bpb": 1.5}) + "\n",
        encoding="utf-8")
    exp = root / "experiments.json"
    exp.write_text(json.dumps({"arms": [{"id": "toy", "family": "demo",
        "verdict": "running", "control_id": None, "lever": {}, "goal": "fixture",
        "run_dirs": ["20261006_demo"]}]}), encoding="utf-8")
    return runs, exp


def test_default_shell_has_two_workspaces():
    from pathlib import Path

    html = (Path(__file__).resolve().parents[1] / "viz" / "ui" / "index.html").read_text(encoding="utf-8")
    assert "LLM Observatory" in html
    assert "Training" in html
    assert "Think" in html
    assert "READY" in html
    assert 'id="checkpoint-select"' in html
    assert 'id="prompt-input"' in html
    assert 'id="generate-button"' in html
    assert 'id="token-events"' in html
    assert 'id="inspect-trace"' in html
    assert 'id="attention-bars"' in html
    assert 'id="compare-checkpoints"' in html
    assert 'id="compare-model-a"' in html
    assert "Activity" in html
    assert 'id="activity-view"' in html
    assert 'id="activity-map"' in html
    assert 'id="inference-panel"' in html
    assert 'id="tools-panel"' in html
    assert 'id="memory-panel"' in html
    assert 'id="cron-panel"' in html
    assert 'id="kanban-panel"' in html
    assert 'id="mesh-panel"' in html
    assert 'id="models-panel"' in html
    assert 'id="gap-chart"' in html
    assert 'id="map-pause"' in html
    assert 'id="neuro-view"' in html
    assert 'id="neuro-map"' in html
    assert 'id="neuro-inspect"' in html
    assert 'id="neuro-pause"' in html
    assert 'id="neuro-inspector"' in html


def test_dashboard_payload_reuses_run_and_arm_summaries(tmp_path):
    runs, exp = _fixtures(tmp_path)
    payload = dashboard_payload(runs, exp)
    assert payload["runs"][0]["name"] == "20261006_demo"
    assert payload["arms"][0]["id"] == "toy"
    assert payload["arms"][0]["best_bpb"] == 1.5


def test_dashboard_api_separates_index_from_selected_run_rows(tmp_path):
    from urllib.error import HTTPError

    runs, exp = _fixtures(tmp_path)
    server = create_server(port=0, runs_dir=runs, experiments_file=exp)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        summary = json.loads(urlopen(base + "/api/dashboard").read())
        assert "rows" not in summary["runs"][0]
        assert "samples" not in summary["runs"][0]
        assert "match_dirs" not in summary["arms"][0]
        detail = json.loads(urlopen(base + "/api/runs/20261006_demo").read())
        assert len(detail["rows"]) == 1
        try:
            urlopen(base + "/api/runs/../../outside")
        except HTTPError as exc:
            assert exc.code == 404
        else:
            raise AssertionError("unknown run id must not be served")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_dashboard_summary_reports_tokens_and_stale_log_age(tmp_path):
    import os
    import time
    from viz.server import DashboardStore, dashboard_summary

    runs, exp = _fixtures(tmp_path)
    store = DashboardStore(runs, exp)
    fresh = dashboard_summary(store.get())["runs"][0]
    assert fresh["tokens_seen"] == 16
    assert fresh["live"] is True
    assert fresh["last_update_age_s"] < 5
    log = runs / "20261006_demo" / "loss.jsonl"
    stale = time.time() - 600
    os.utime(log, (stale, stale))
    expired = dashboard_summary(store.get())["runs"][0]
    assert expired["live"] is False
    assert expired["last_update_age_s"] >= 600
    assert expired["log_updated_at"]


def test_snapshot_cache_rebuilds_only_when_run_or_registry_changes(tmp_path, monkeypatch):
    from viz import server

    runs, exp = _fixtures(tmp_path)
    calls = 0
    original = server.dashboard_payload
    def counted(*args):
        nonlocal calls
        calls += 1
        return original(*args)
    monkeypatch.setattr(server, "dashboard_payload", counted)
    store = server.DashboardStore(runs, exp)
    store.get(); store.get()
    assert calls == 1
    with (runs / "20261006_demo" / "loss.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"step": 2, "train": 1.9, "avg50": 2.0}) + "\\n")
    store.get()
    assert calls == 2


def test_dashboard_endpoint_and_health_are_served_locally(tmp_path):
    runs, exp = _fixtures(tmp_path)
    server = create_server(host="127.0.0.1", port=0, runs_dir=runs,
                           experiments_file=exp)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        with urlopen(base + "/api/health") as r:
            assert json.loads(r.read())["ok"] is True
        with urlopen(base + "/api/dashboard") as r:
            assert json.loads(r.read())["runs"][0]["name"] == "20261006_demo"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_non_get_requests_are_rejected(tmp_path):
    from urllib.error import HTTPError
    from urllib.request import Request

    server = create_server(host="127.0.0.1", port=0, runs_dir=tmp_path / "runs",
                           experiments_file=tmp_path / "missing.json")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        request = Request(f"http://127.0.0.1:{server.server_port}/api/dashboard", method="POST")
        try:
            urlopen(request)
        except HTTPError as exc:
            assert exc.code == 405
        else:
            raise AssertionError("POST must not be accepted by the read-only dashboard")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_model_routes_use_service_without_exposing_checkpoint_paths(tmp_path):
    from urllib.request import Request

    class FakeInference:
        def checkpoints(self):
            return [{"id": "demo.pt", "size_bytes": 4}]
        def status(self):
            return {"loaded": False, "checkpoint_id": None, "device": None}
        def devices(self):
            return {"cpu_available": True, "cuda_available": False, "training_busy": False}
        def load(self, checkpoint_id, device):
            assert checkpoint_id == "demo.pt" and device == "cpu"
            return {"loaded": True, "checkpoint_id": checkpoint_id, "device": device}
        def compare_checkpoints(self, body):
            return {"prompt": body["prompt"], "token_ids_match": True,
                    "same_top_prediction": True, "models": []}
        def inspect(self, body):
            assert body["prompt"] == "hello"
            return {"tokens": ["hello"], "layers": [], "capture_kind": "native-full"}
        def start_generation(self, body):
            return {"id": "task1", "status": "running"}
        def generation(self, task_id, after=0):
            return {"id": task_id, "status": "completed", "text": "hello!",
                    "events": [{"step": 1, "token_id": 33}][after:], "cursor": 1}
        def unload(self):
            return {"loaded": False}

    service = FakeInference()
    server = create_server(port=0, ui_dir=tmp_path / "ui",
                           runs_dir=tmp_path / "runs",
                           experiments_file=tmp_path / "missing.json",
                           inference_service=service)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        assert json.loads(urlopen(base + "/api/checkpoints").read())[0]["id"] == "demo.pt"
        load = Request(base + "/api/model/load", data=json.dumps({"checkpoint_id":"demo.pt","device":"cpu"}).encode(), headers={"Content-Type":"application/json"}, method="POST")
        assert json.loads(urlopen(load).read())["loaded"] is True
        inspect = Request(base + "/api/trace", data=json.dumps({"prompt":"hello"}).encode(), headers={"Content-Type":"application/json"}, method="POST")
        assert json.loads(urlopen(inspect).read())["capture_kind"] == "native-full"
        compare = Request(base + "/api/compare", data=json.dumps({"checkpoint_ids":["a","b"],"prompt":"hello"}).encode(), headers={"Content-Type":"application/json"}, method="POST")
        assert json.loads(urlopen(compare).read())["token_ids_match"] is True
        generate = Request(base + "/api/generations", data=b"{}", headers={"Content-Type":"application/json"}, method="POST")
        task = json.loads(urlopen(generate).read())
        assert json.loads(urlopen(base + "/api/generations/" + task["id"]).read())["status"] == "completed"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_activity_api_serves_streams_and_degrades(tmp_path):
    from tests.test_activity import _make_hfm_db, _make_mnemo_db, _make_state_db

    runs, exp = _fixtures(tmp_path)
    server = create_server(port=0, runs_dir=runs, experiments_file=exp,
                           hfm_db=_make_hfm_db(tmp_path),
                           state_db=_make_state_db(tmp_path),
                           mnemo_db=_make_mnemo_db(tmp_path),
                           cron_db=tmp_path / "no-cron.db",
                           jobs_json=tmp_path / "no-jobs.json",
                           kanban_db=tmp_path / "no-kanban.db",
                           peers=[],
                           lab_runs=tmp_path / "no-runs",
                           lab_ckpt=tmp_path / "no-ckpts")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        payload = json.loads(urlopen(base + "/api/activity").read())
        for key in ("inference", "tools", "memory", "map", "updated_at"):
            assert key in payload
        assert payload["inference"]["ok"] is True
        assert payload["tools"]["ok"] is True
        assert payload["memory"]["ok"] is True
        pods = {n["id"] for n in payload["map"]["nodes"] if n["kind"] == "pod"}
        # file-backed streams injected above are healthy; cron/kanban/models
        # were pointed at missing paths (degraded, no pod); mesh has no peers
        assert pods == {"pod:inference", "pod:tools", "pod:memory", "pod:mesh"}
        # a dead source must not 500 — it degrades its own stream
        server2 = create_server(port=0, runs_dir=runs, experiments_file=exp,
                                state_db=tmp_path / "missing.db",
                                hfm_db=tmp_path / "missing2.db",
                                mnemo_db=tmp_path / "missing3.db",
                                cron_db=tmp_path / "missing4.db",
                                jobs_json=tmp_path / "missing5.json",
                                kanban_db=tmp_path / "missing6.db",
                                peers=[],
                                lab_runs=tmp_path / "missing-runs",
                                lab_ckpt=tmp_path / "missing-ckpts")
        thread2 = threading.Thread(target=server2.serve_forever, daemon=True)
        thread2.start()
        try:
            degraded = json.loads(urlopen(
                f"http://127.0.0.1:{server2.server_port}/api/activity").read())
            for stream in ("inference", "tools", "memory", "cron", "kanban", "models"):
                assert degraded[stream]["ok"] is False, stream
            pods = {n["stream"] for n in degraded["map"]["nodes"] if n["kind"] == "pod"}
            assert pods <= {"mesh"}
            assert degraded["map"]["links"] == []
        finally:
            server2.shutdown()
            server2.server_close()
            thread2.join(timeout=2)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_shell_serves_only_files_under_ui_root(tmp_path):
    from urllib.error import HTTPError

    ui = tmp_path / "ui"
    ui.mkdir()
    (ui / "index.html").write_text("<h1>local</h1>", encoding="utf-8")
    (ui / "app.css").write_text("body{color:white}", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("private", encoding="utf-8")
    server = create_server(port=0, ui_dir=ui, runs_dir=tmp_path / "runs",
                           experiments_file=tmp_path / "missing.json")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        assert "local" in urlopen(base + "/").read().decode()
        assert "color:white" in urlopen(base + "/assets/app.css").read().decode()
        try:
            urlopen(base + "/assets/%2e%2e/secret.txt")
        except HTTPError as exc:
            assert exc.code == 404
        else:
            raise AssertionError("static serving must not escape the UI directory")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
