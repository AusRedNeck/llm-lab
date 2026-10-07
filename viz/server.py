"""Local-only HTTP dashboard for the llm-lab observatory."""
from __future__ import annotations

import argparse
import json
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

from viz.activity import STATE_DB, HFM_DB, MNEMO_DB, collect_activity
from viz.dashboard import LIVE_WINDOW_S, build_arms, load_runs

LAB_ROOT = Path(__file__).resolve().parents[1]


def dashboard_payload(runs_dir=None, experiments_file=None):
    """Return the existing dashboard summaries as a JSON-ready payload."""
    runs_dir = Path(runs_dir or LAB_ROOT / "runs")
    experiments_file = Path(experiments_file or LAB_ROOT / "experiments.json")
    runs = load_runs(str(runs_dir)) if runs_dir.is_dir() else []
    meta = json.loads(experiments_file.read_text(encoding="utf-8")) if experiments_file.is_file() else {}
    arms, problems = build_arms(meta.get("arms", []), {r["name"]: r for r in runs})
    return {"updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "runs": runs, "arms": arms, "problems": problems}


class DashboardStore:
    """Cache expensive run parsing until a log or the registry actually changes."""
    def __init__(self, runs_dir, experiments_file):
        self.runs_dir = Path(runs_dir).resolve()
        self.experiments_file = Path(experiments_file).resolve()
        self._lock = threading.RLock()
        self._signature = None
        self._payload = None

    def _current_signature(self):
        files = []
        for path in sorted(self.runs_dir.glob("*/loss.jsonl")) if self.runs_dir.is_dir() else []:
            try:
                stat = path.stat()
                files.append((path.parent.name, stat.st_mtime_ns, stat.st_size))
            except OSError:
                continue
        try:
            stat = self.experiments_file.stat()
            registry = (stat.st_mtime_ns, stat.st_size)
        except OSError:
            registry = None
        return tuple(files), registry

    def get(self):
        with self._lock:
            signature = self._current_signature()
            if signature != self._signature:
                self._payload = dashboard_payload(self.runs_dir, self.experiments_file)
                self._signature = signature
            now = time.time()
            for run in self._payload["runs"]:
                try:
                    log_path = self.runs_dir / run["name"] / "loss.jsonl"
                    mtime = log_path.stat().st_mtime
                    age = max(0.0, now - mtime)
                    run["live"] = age < LIVE_WINDOW_S
                    run["log_updated_at"] = datetime.fromtimestamp(mtime).astimezone().isoformat(timespec="seconds")
                    run["last_update_age_s"] = int(age)
                    if run.get("last_step") is not None and run.get("eff") and run.get("ctx"):
                        run["tokens_seen"] = int(run["last_step"] * run["eff"] * run["ctx"])
                    else:
                        run["tokens_seen"] = None
                except OSError:
                    run["live"] = False
                    run["log_updated_at"] = None
                    run["last_update_age_s"] = None
                    run["tokens_seen"] = None
            return self._payload


class ActivityStore:
    """TTL cache over the three activity streams.

    TTL only, no mtime signature: state.db is written constantly by the
    gateway, so an mtime check would rebuild on every poll. Collect costs
    ~44ms measured; a 10s TTL keeps the UI's 5s poll cheap.
    """

    def __init__(self, state_db=None, hfm_db=None, mnemo_db=None,
                 cron_db=None, jobs_json=None, kanban_db=None, peers=None,
                 runs_dir=None, ckpt_dir=None, ttl_s=10):
        self.state_db = state_db or STATE_DB
        self.hfm_db = hfm_db or HFM_DB
        self.mnemo_db = mnemo_db or MNEMO_DB
        self.cron_db = cron_db
        self.jobs_json = jobs_json
        self.kanban_db = kanban_db
        self.peers = peers
        self.runs_dir = runs_dir
        self.ckpt_dir = ckpt_dir
        self.ttl_s = ttl_s
        self._lock = threading.Lock()
        self._payload = None
        self._built_at = 0.0

    def get(self):
        with self._lock:
            now = time.time()
            if self._payload is None or now - self._built_at > self.ttl_s:
                self._payload = collect_activity(
                    hfm_path=self.hfm_db, state_path=self.state_db,
                    mnemo_path=self.mnemo_db, cron_path=self.cron_db,
                    jobs_path=self.jobs_json, kanban_path=self.kanban_db,
                    peers=self.peers, runs_dir=self.runs_dir,
                    ckpt_dir=self.ckpt_dir)
                self._built_at = now
            return self._payload


def dashboard_summary(payload):
    """Compact index; rows and generated samples are fetched only for one run."""
    run_keys = ("name", "live", "params_m", "preset", "eff", "ctx", "corpus",
                "tokenizer", "val_frac", "train_tokens", "corpus_tokens", "best_bpb", "best_step", "stop_step",
                "last_step", "log_updated_at", "last_update_age_s", "tokens_seen")
    arm_keys = ("id", "family", "verdict", "control", "lever", "goal",
                "confounded_flag", "live", "best_bpb", "best_step", "best_tokens",
                "train_tokens", "stop_step", "params_m", "n_dirs")
    return {
        "updated_at": payload["updated_at"],
        "runs": [{k: run[k] for k in run_keys if k in run} for run in payload["runs"]],
        "arms": [{k: arm[k] for k in arm_keys if k in arm} for arm in payload["arms"]],
        "problems": payload["problems"],
    }


def create_server(host="127.0.0.1", port=8787, *, runs_dir=None,
                  experiments_file=None, ui_dir=None, inference_service=None,
                  state_db=None, hfm_db=None, mnemo_db=None,
                  cron_db=None, jobs_json=None, kanban_db=None, peers=None,
                  lab_runs=None, lab_ckpt=None):
    """Create, but do not start, a local dashboard server (use port=0 in tests)."""
    runs_dir = Path(runs_dir or LAB_ROOT / "runs").resolve()
    experiments_file = Path(experiments_file or LAB_ROOT / "experiments.json").resolve()
    ui_dir = Path(ui_dir or Path(__file__).with_name("ui")).resolve()
    store = DashboardStore(runs_dir, experiments_file)
    activity_store = ActivityStore(state_db=state_db, hfm_db=hfm_db,
                                   mnemo_db=mnemo_db, cron_db=cron_db,
                                   jobs_json=jobs_json, kanban_db=kanban_db,
                                   peers=peers, runs_dir=lab_runs,
                                   ckpt_dir=lab_ckpt)
    if inference_service is None:
        from viz.inference_service import InferenceService
        inference_service = InferenceService(project_root=LAB_ROOT,
                                             checkpoint_root=LAB_ROOT / "checkpoints")

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status, body, content_type):
            data = body if isinstance(body, bytes) else body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _send_json(self, status, payload):
            self._send(status, json.dumps(payload), "application/json; charset=utf-8")

        def _read_json(self):
            length = int(self.headers.get("Content-Length", "0"))
            if length < 1 or length > 16384:
                raise ValueError("JSON request body must be between 1 and 16384 bytes")
            if "application/json" not in self.headers.get("Content-Type", ""):
                raise ValueError("Content-Type must be application/json")
            return json.loads(self.rfile.read(length))

        def _reject_write(self):
            self.send_response(405)
            self.send_header("Allow", "GET")
            self.end_headers()

        do_PUT = do_PATCH = do_DELETE = _reject_write

        def do_POST(self):
            route = urlsplit(self.path).path
            if route not in ("/api/model/load", "/api/model/unload", "/api/generations", "/api/trace", "/api/compare"):
                self._reject_write()
                return
            try:
                if route == "/api/model/load":
                    body = self._read_json()
                    result = inference_service.load(body.get("checkpoint_id"), body.get("device", "cpu"))
                elif route == "/api/model/unload":
                    self._read_json()
                    result = inference_service.unload()
                elif route == "/api/trace":
                    result = inference_service.inspect(self._read_json())
                elif route == "/api/compare":
                    result = inference_service.compare_checkpoints(self._read_json())
                elif route == "/api/generations":
                    result = inference_service.start_generation(self._read_json())
                self._send_json(200, result)
            except ValueError as exc:
                self._send_json(400, {"error": str(exc)})
            except KeyError as exc:
                self._send_json(404, {"error": str(exc)})
            except RuntimeError as exc:
                self._send_json(409, {"error": str(exc)})
            except Exception as exc:
                self._send_json(500, {"error": str(exc)})

        def do_GET(self):
            route = urlsplit(self.path).path
            if route == "/api/health":
                self._send(200, json.dumps({"ok": True, "service": "llm-lab-viz"}),
                           "application/json; charset=utf-8")
                return
            if route == "/api/checkpoints":
                self._send_json(200, inference_service.checkpoints())
                return
            if route == "/api/model/status":
                self._send_json(200, inference_service.status())
                return
            if route == "/api/devices":
                self._send_json(200, inference_service.devices())
                return
            if route.startswith("/api/generations/"):
                task_id = unquote(route.removeprefix("/api/generations/"))
                if not task_id or "/" in task_id or "\\\\" in task_id:
                    self._send_json(404, {"error": "generation not found"})
                    return
                try:
                    query = parse_qs(urlsplit(self.path).query)
                    after = int(query.get("after", ["0"])[0])
                    self._send_json(200, inference_service.generation(task_id, after))
                except KeyError:
                    self._send_json(404, {"error": "generation not found"})
                except ValueError as exc:
                    self._send_json(400, {"error": str(exc)})
                return
            if route.startswith("/api/runs/"):
                name = unquote(route.removeprefix("/api/runs/"))
                if not name or "/" in name or "\\\\" in name:
                    self._send(404, '{"error":"run not found"}', "application/json; charset=utf-8")
                    return
                payload = store.get()
                run = next((item for item in payload["runs"] if item["name"] == name), None)
                if run is None:
                    self._send(404, '{"error":"run not found"}', "application/json; charset=utf-8")
                else:
                    self._send(200, json.dumps(run), "application/json; charset=utf-8")
                return
            if route == "/api/activity":
                try:
                    self._send_json(200, activity_store.get())
                except Exception as exc:
                    self._send(500, json.dumps({"error": str(exc)}),
                               "application/json; charset=utf-8")
                return
            if route == "/api/dashboard":
                try:
                    body = json.dumps(dashboard_summary(store.get()))
                    self._send(200, body, "application/json; charset=utf-8")
                except Exception as exc:
                    self._send(500, json.dumps({"error": str(exc)}),
                               "application/json; charset=utf-8")
                return
            if route == "/":
                target = ui_dir / "index.html"
            elif route.startswith("/assets/"):
                target = (ui_dir / unquote(route.removeprefix("/assets/"))).resolve()
                if ui_dir not in target.parents:
                    self._send(404, "Not found", "text/plain; charset=utf-8")
                    return
            else:
                self._send(404, "Not found", "text/plain; charset=utf-8")
                return
            if not target.is_file():
                self._send(404, "Not found", "text/plain; charset=utf-8")
                return
            kind = "text/html" if target.suffix == ".html" else "text/css" if target.suffix == ".css" else "application/javascript"
            self._send(200, target.read_bytes(), kind + "; charset=utf-8")

    return ThreadingHTTPServer((host, port), Handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Serve the local llm-lab observatory")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    args = parser.parse_args(argv)
    server = create_server(args.host, args.port)
    print(f"LLM Observatory: http://{args.host}:{server.server_port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
