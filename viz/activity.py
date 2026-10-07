"""Activity-stream collectors for the Observatory's Activity workspace.

Three read-only SQLite sources, each degrading independently: a dead or
missing DB returns ``{"ok": False, "error": ...}`` and never raises, so one
offline source takes down its panel, not the payload.

Sources (env-var overridable — paths are machine-local):
  inference  D:/Projects/hfm/hfm_state.db            -> request_log
  tools      <hermes home>/state.db                  -> messages + sessions
  memory     <hermes home>/mnemosyne/data/mnemosyne.db -> working/episodic +
             consolidation_log + graph_edges + triples
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
import urllib.request
from datetime import datetime
from pathlib import Path

HFM_DB = os.environ.get("HFM_DB", "D:/Projects/hfm/hfm_state.db")
STATE_DB = os.environ.get(
    "HERMES_STATE_DB",
    "C:/Users/shane/AppData/Local/hermes/state.db")
MNEMO_DB = os.environ.get(
    "MNEMOSYNE_DB",
    "C:/Users/shane/AppData/Local/hermes/mnemosyne/data/mnemosyne.db")
CRON_DB = os.environ.get(
    "HERMES_CRON_DB",
    "C:/Users/shane/AppData/Local/hermes/cron/executions.db")
JOBS_JSON = os.environ.get(
    "HERMES_JOBS_JSON",
    "C:/Users/shane/AppData/Local/hermes/cron/jobs.json")
KANBAN_DB = os.environ.get("HERMES_KANBAN_DB", "C:/Users/shane/AppData/Local/hermes/kanban.db")
HERMES_CONFIG = os.environ.get("HERMES_CONFIG", "C:/Users/shane/AppData/Local/hermes/config.yaml")
RUNS_DIR = os.environ.get("LLM_LAB_RUNS", "D:/Projects/llm-lab/runs")
CKPT_DIR = os.environ.get("LLM_LAB_CHECKPOINTS", "D:/Projects/llm-lab/checkpoints")

FRESH_WINDOW_S = 60
POD_LABELS = {"inference": "Inference", "tools": "Tool calls", "memory": "Memory",
              "cron": "Cron", "kanban": "Kanban", "mesh": "Mesh", "models": "Models"}


def _ro(path: str | Path):
    con = sqlite3.connect(f"file:{Path(path)}?mode=ro", uri=True)
    con.execute("PRAGMA busy_timeout=500")
    return con


def _iso(epoch_seconds: float) -> str:
    try:
        return datetime.fromtimestamp(epoch_seconds).astimezone().isoformat(timespec="seconds")
    except (OSError, OverflowError, ValueError, TypeError):
        # Windows localtime rejects pre-1970/out-of-range stamps — degrade to raw
        return str(epoch_seconds)


def _degenerate(stream: str, error: Exception) -> dict:
    return {"ok": False, "error": f"{type(error).__name__}: {error}",
            "events": [], "stats": {}}


HFM_HEALTHZ = "http://127.0.0.1:8077/healthz"


def _hfm_proxy_status(url: str) -> str:
    """'up' if HFM answers healthz, else 'down'. url="" skips the probe."""
    if not url:
        return "unknown"
    try:
        with urllib.request.urlopen(url, timeout=0.4) as resp:
            return "up" if getattr(resp, "status", 200) == 200 else "down"
    except Exception:
        return "down"


def collect_inference(path: str | Path | None = None, limit: int = 100,
                      healthz_url: str | None = None) -> dict:
    path = path or HFM_DB
    try:
        with _ro(path) as con:
            rows = con.execute(
                "SELECT ts, provider, model, status, tokens_prompt, tokens_completion,"
                " latency_ms, cost_cents, cached, error FROM request_log"
                " ORDER BY ts DESC LIMIT ?", (limit,)).fetchall()
            last_hour = con.execute(
                "SELECT COUNT(*), COALESCE(SUM(cost_cents),0),"
                " COALESCE(SUM(tokens_prompt + tokens_completion),0),"
                " SUM(CASE WHEN error IS NOT NULL OR status >= 400 THEN 1 ELSE 0 END)"
                " FROM request_log WHERE ts > ?", ((time.time() - 3600) * 1000,)).fetchone()
        events = [{
            "ts": _iso(r[0] / 1000), "ts_epoch_ms": r[0], "provider": r[1],
            "model": r[2], "status": r[3], "tokens_prompt": r[4],
            "tokens_completion": r[5], "latency_ms": r[6], "cost_cents": r[7],
            "cached": bool(r[8]), "error": r[9],
        } for r in rows]
        newest_ms = events[0]["ts_epoch_ms"] if events else 0
        fresh_data = (time.time() * 1000 - newest_ms) < FRESH_WINDOW_S * 1000
        freshness = "LIVE" if fresh_data else "FILE"
        # stale data with a healthy proxy means 'idle', not 'down' — probe only
        # when the data itself can't answer the question
        proxy = "unknown" if fresh_data else _hfm_proxy_status(
            HFM_HEALTHZ if healthz_url is None else healthz_url)
        return {"ok": True, "error": None, "source": str(path),
                "freshness": freshness, "hfm_status": proxy,
                "events": events,
                "stats": {"last_hour_count": last_hour[0], "cost_cents": last_hour[1],
                          "tokens_total": last_hour[2], "errors": last_hour[3] or 0}}
    except Exception as exc:  # degraded panel, never a 500
        return _degenerate("inference", exc)


def collect_tools(path: str | Path | None = None, limit: int = 100) -> dict:
    path = path or STATE_DB
    try:
        with _ro(path) as con:
            rows = con.execute(
                "SELECT m.id, m.session_id, m.timestamp, m.role, m.tool_name,"
                " m.tool_calls, m.tool_call_id, s.title, s.profile_name"
                " FROM messages m LEFT JOIN sessions s ON s.id = m.session_id"
                " WHERE m.tool_calls IS NOT NULL OR m.tool_name IS NOT NULL"
                " ORDER BY m.id DESC LIMIT ?", (limit,)).fetchall()
        events = []
        for r in rows:
            tool_name = r[4]
            if r[3] == "assistant" and r[5]:
                try:
                    calls = json.loads(r[5])
                    tool_name = calls[0].get("function", {}).get("name") or tool_name
                except (ValueError, IndexError, AttributeError):
                    pass
            events.append({"id": r[0], "session_id": r[1], "timestamp": r[2],
                           "iso": _iso(r[2]) if isinstance(r[2], (int, float)) else r[2],
                           "role": r[3],
                           "kind": "result" if r[3] == "tool" else "invocation",
                           "tool_name": tool_name, "tool_call_id": r[6],
                           "session_title": r[7], "profile_name": r[8]})
        by_tool: dict[str, int] = {}
        for e in events:
            if e["tool_name"]:
                by_tool[e["tool_name"]] = by_tool.get(e["tool_name"], 0) + 1
        return {"ok": True, "error": None, "source": str(path), "events": events,
                "stats": {"total": len(events),
                          "invocations": sum(1 for e in events if e["kind"] == "invocation"),
                          "results": sum(1 for e in events if e["kind"] == "result"),
                          "by_tool": by_tool}}
    except Exception as exc:
        return _degenerate("tools", exc)


def _memory_rows(con, table: str, tier: str) -> list[dict]:
    rows = con.execute(
        f"SELECT id, content, source, timestamp, importance, recall_count, last_recalled"
        f" FROM {table}").fetchall()
    return [{"id": r[0], "tier": tier, "content": r[1], "source": r[2],
             "timestamp": r[3], "importance": r[4], "recall_count": r[5] or 0,
             "last_recalled": r[6],
             "sort_key": r[6] or r[3] or "",
             "preview": (r[1] or "")[:120]} for r in rows]


def collect_memory(path: str | Path | None = None, limit: int = 100) -> dict:
    path = path or MNEMO_DB
    try:
        with _ro(path) as con:
            events = (_memory_rows(con, "working_memory", "working")
                      + _memory_rows(con, "episodic_memory", "episodic"))
            events.sort(key=lambda e: e["sort_key"], reverse=True)
            events = events[:limit]
            for e in events:  # keep the payload small: preview, not full content
                e.pop("sort_key", None)
                e.pop("content", None)
            cons = con.execute("SELECT COUNT(*) FROM consolidation_log").fetchone()[0]
            edges = con.execute(
                "SELECT source, target, edge_type, weight FROM graph_edges"
                " ORDER BY weight DESC LIMIT 60").fetchall()
            edge_count = con.execute("SELECT COUNT(*) FROM graph_edges").fetchone()[0]
            triple_count = con.execute("SELECT COUNT(*) FROM triples").fetchone()[0]
        return {"ok": True, "error": None, "source": str(path), "events": events,
                "stats": {"working": sum(1 for e in events if e["tier"] == "working"),
                          "episodic": sum(1 for e in events if e["tier"] == "episodic"),
                          "recalled_total": sum(e["recall_count"] for e in events),
                          "consolidations": cons, "graph_edges": edge_count,
                          "triples": triple_count},
                "graph": [{"source": e[0], "target": e[1], "edge_type": e[2],
                           "weight": e[3]} for e in edges]}
    except Exception as exc:
        return _degenerate("memory", exc)


def collect_cron(path: str | Path | None = None, jobs_path: str | Path | None = None,
                 limit: int = 100) -> dict:
    path = path or CRON_DB
    jobs_path = jobs_path or JOBS_JSON
    try:
        names = {}
        try:  # job names are a nicety — missing/rotten jobs.json degrades labels only
            with open(jobs_path, encoding="utf-8") as fh:
                names = {j.get("id"): j.get("name", "") for j in json.load(fh).get("jobs", [])}
        except Exception:
            pass
        with _ro(path) as con:
            rows = con.execute(
                "SELECT id, job_id, status, started_at, finished_at, error, delivery_outcome"
                " FROM executions ORDER BY rowid DESC LIMIT ?", (limit,)).fetchall()
            open_incidents = con.execute(
                "SELECT COUNT(*) FROM cron_incidents WHERE closed_at IS NULL").fetchone()[0]
        events = [{
            "id": r[0], "job_id": r[1], "job": names.get(r[1], r[1]),
            "status": r[2], "ts": r[3] or "", "started_at": r[3],
            "finished_at": r[4], "error": (r[5] or "")[:160] or None,
            "delivery": r[6],
        } for r in rows]
        by_status: dict[str, int] = {}
        for e in events:
            by_status[e["status"] or "?"] = by_status.get(e["status"] or "?", 0) + 1
        failed = sum(v for k, v in by_status.items() if k in ("failed", "error", "timeout"))
        return {"ok": True, "error": None, "source": str(path), "events": events,
                "stats": {"total": len(events), "failed": failed,
                          "open_incidents": open_incidents, "by_status": by_status}}
    except Exception as exc:
        return _degenerate("cron", exc)


def _ts(value):
    """Epoch seconds or ISO string -> ISO local string; anything else passes through."""
    if isinstance(value, (int, float)):
        return _iso(value)
    return str(value) if value is not None else None


def collect_kanban(path: str | Path | None = None, limit: int = 100) -> dict:
    path = path or KANBAN_DB
    try:
        with _ro(path) as con:
            rows = con.execute(
                "SELECT e.id, e.task_id, e.kind, e.created_at, t.title, t.status, t.assignee"
                " FROM task_events e LEFT JOIN tasks t ON t.id = e.task_id"
                " ORDER BY e.rowid DESC LIMIT ?", (limit,)).fetchall()
            status_rows = con.execute(
                "SELECT status, COUNT(*) FROM tasks GROUP BY status").fetchall()
        by_status = {r[0] or "?": r[1] for r in status_rows}
        events = [{"id": r[0], "task_id": r[1], "kind": r[2], "ts": _ts(r[3]),
                   "title": (r[4] or "")[:120], "task_status": r[5], "assignee": r[6]}
                  for r in rows]
        done_like = sum(v for k, v in by_status.items() if k in ("done", "archived"))
        total_tasks = sum(by_status.values())
        return {"ok": True, "error": None, "source": str(path), "events": events,
                "stats": {"total_tasks": total_tasks, "open": total_tasks - done_like,
                          "by_status": by_status, "events_in_window": len(events)}}
    except Exception as exc:
        return _degenerate("kanban", exc)


def _default_peers() -> list:
    """Parse the bot_peers block of config.yaml -> [(name, url), ...]."""
    peers, in_block, name = [], False, None
    try:
        with open(HERMES_CONFIG, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                indent = len(line) - len(line.lstrip())
                text = line.strip()
                if indent == 0:
                    in_block = text.startswith("bot_peers:")
                    name = None
                    continue
                if not in_block:
                    continue
                if indent == 2 and text.endswith(":"):
                    name = text[:-1]
                elif indent == 4 and text.startswith("url:") and name:
                    peers.append((name, text.split("url:", 1)[1].strip()))
                    name = None
    except Exception:
        return []
    return peers


def collect_mesh(peers=None, timeout: float = 0.4) -> dict:
    """Probe each configured peer's /health. Per-peer failures are events, not errors."""
    try:
        peers = peers if peers is not None else _default_peers()
        events = []
        for name, url in peers:
            t0 = time.time()
            ok, version, error = False, None, None
            try:
                with urllib.request.urlopen(url.rstrip("/") + "/health",
                                            timeout=timeout) as resp:
                    body = json.loads(resp.read() or b"{}")
                    ok, version = True, body.get("version")
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            events.append({"name": name, "url": url, "ok": ok, "version": version,
                           "latency_ms": int((time.time() - t0) * 1000),
                           "error": error, "ts": _iso(time.time())})
        return {"ok": True, "error": None, "source": "config:bot_peers",
                "events": events,
                "stats": {"up": sum(1 for e in events if e["ok"]),
                          "total": len(events)}}
    except Exception as exc:
        return _degenerate("mesh", exc)


def collect_models(runs_dir: str | Path | None = None,
                   ckpt_dir: str | Path | None = None, limit: int = 25) -> dict:
    """The models we are MAKING: checkpoint files + run liveness from llm-lab."""
    runs_dir = Path(runs_dir or RUNS_DIR)
    ckpt_dir = Path(ckpt_dir or CKPT_DIR)
    if not ckpt_dir.is_dir():
        # a missing lab is a missing source, not a healthy empty one
        return _degenerate("models", FileNotFoundError(f"no checkpoints dir: {ckpt_dir}"))
    try:
        all_ckpts = list(ckpt_dir.glob("*.pt"))
        sized = []
        for p in all_ckpts:
            try:
                sized.append((p, p.stat()))
            except OSError:
                continue
        sized.sort(key=lambda item: item[1].st_mtime, reverse=True)
        events = [{"id": p.name, "name": p.name,
                   "kind": "best" if "_best" in p.name else "step",
                   "size_mb": round(st.st_size / 1048576, 1),
                   "ts": _iso(st.st_mtime), "ts_epoch_ms": int(st.st_mtime * 1000)}
                  for p, st in sized[:limit]]
        now, runs, live = time.time(), [], 0
        if runs_dir.is_dir():
            for d in runs_dir.iterdir():
                log = d / "loss.jsonl" if d.is_dir() else None
                if log and log.is_file():
                    age = now - log.stat().st_mtime
                    runs.append((d.name, age))
                    live += age < 120
        runs.sort(key=lambda r: r[1])
        return {"ok": True, "error": None, "source": str(ckpt_dir), "events": events,
                "stats": {"runs": len(runs), "live_runs": live,
                          "checkpoints": len(sized),
                          "size_gb": round(sum(st.st_size for _, st in sized) / 2**30, 2),
                          "newest_run": runs[0][0] if runs else None}}
    except Exception as exc:
        return _degenerate("models", exc)


def _build_map(streams: dict) -> dict:
    nodes: list[dict] = []
    links: list[dict] = []
    for name, payload in streams.items():
        if not payload.get("ok"):  # dead stream gets no pod — honest empty map
            continue
        nodes.append({"id": f"pod:{name}", "kind": "pod", "stream": name,
                      "label": POD_LABELS[name]})
        events = (payload.get("events") or [])[:25]
        hubs: dict[str, str] = {}
        for e in events:
            node_id = f"{name}:{e.get('id', e.get('ts_epoch_ms', e.get('name', '')))}:{e.get('timestamp', e.get('ts', ''))}"
            if name == "inference":
                row_key = f"inference:{e.get('ts_epoch_ms', '')}"
            elif name == "memory":
                row_key = f"memory:{e.get('tier', '')}:{e.get('id', '')}"
            else:
                row_key = f"{name}:{e.get('id') or e.get('name') or ''}"
            nodes.append({"id": node_id, "kind": "event", "stream": name,
                          "label": e.get("tool_name") or e.get("model")
                          or e.get("preview", "")[:40] or name,
                          "row_key": row_key, "detail": e})
            if name == "tools" and e.get("session_id"):
                hub_id = f"session:{e['session_id']}"
                if hub_id not in hubs:
                    hubs[hub_id] = e.get("session_title") or e["session_id"]
                    nodes.append({"id": hub_id, "kind": "session", "stream": "tools",
                                  "label": hubs[hub_id]})
                    links.append({"source": hub_id, "target": f"pod:{name}"})
                links.append({"source": node_id, "target": hub_id})
            else:
                links.append({"source": node_id, "target": f"pod:{name}"})
    return {"nodes": nodes, "links": links}


def collect_activity(hfm_path: str | Path | None = None,
                     state_path: str | Path | None = None,
                     mnemo_path: str | Path | None = None,
                     cron_path: str | Path | None = None,
                     jobs_path: str | Path | None = None,
                     kanban_path: str | Path | None = None,
                     peers: list | None = None,
                     runs_dir: str | Path | None = None,
                     ckpt_dir: str | Path | None = None,
                     limit: int = 100) -> dict:
    """Assemble every stream. None means 'use the machine default' — tests inject all."""
    jobs = {
        "inference": lambda: collect_inference(path=hfm_path, limit=limit),
        "tools": lambda: collect_tools(path=state_path, limit=limit),
        "memory": lambda: collect_memory(path=mnemo_path, limit=limit),
        "cron": lambda: collect_cron(path=cron_path, jobs_path=jobs_path, limit=limit),
        "kanban": lambda: collect_kanban(path=kanban_path, limit=limit),
        "mesh": lambda: collect_mesh(peers=peers),
        "models": lambda: collect_models(runs_dir=runs_dir, ckpt_dir=ckpt_dir,
                                         limit=min(limit, 25)),
    }
    streams = {}
    for name, job in jobs.items():
        try:  # collectors self-guard; this is belt-and-braces
            streams[name] = job()
        except Exception as exc:
            streams[name] = _degenerate(name, exc)
    try:
        graph = _build_map(streams)
    except Exception as exc:
        graph = {"nodes": [], "links": [], "error": f"{type(exc).__name__}: {exc}"}
    return {**streams, "map": graph,
            "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "panel_error": None}
