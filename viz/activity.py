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
from datetime import datetime
from pathlib import Path

HFM_DB = os.environ.get("HFM_DB", "D:/Projects/hfm/hfm_state.db")
STATE_DB = os.environ.get(
    "HERMES_STATE_DB",
    "C:/Users/shane/AppData/Local/hermes/state.db")
MNEMO_DB = os.environ.get(
    "MNEMOSYNE_DB",
    "C:/Users/shane/AppData/Local/hermes/mnemosyne/data/mnemosyne.db")

FRESH_WINDOW_S = 60
POD_LABELS = {"inference": "Inference", "tools": "Tool calls", "memory": "Memory"}


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


def collect_inference(path: str | Path | None = None, limit: int = 100) -> dict:
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
        freshness = "LIVE" if (time.time() * 1000 - newest_ms) < FRESH_WINDOW_S * 1000 else "FILE"
        return {"ok": True, "error": None, "source": str(path), "freshness": freshness,
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


def _build_map(streams: dict) -> dict:
    nodes: list[dict] = []
    links: list[dict] = []
    for name, payload in streams.items():
        if not payload.get("ok"):  # dead stream gets no pod — honest empty map
            continue
        nodes.append({"id": f"pod:{name}", "kind": "pod", "stream": name,
                      "label": POD_LABELS[name]})
        events = (payload.get("events") or [])[:40]
        hubs: dict[str, str] = {}
        for e in events:
            node_id = f"{name}:{e.get('id', e.get('ts_epoch_ms', e.get('id', '')))}:{e.get('timestamp', e.get('ts', ''))}"
            nodes.append({"id": node_id, "kind": "event", "stream": name,
                          "label": e.get("tool_name") or e.get("model")
                          or e.get("preview", "")[:40] or name,
                          "detail": e})
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
                     limit: int = 100) -> dict:
    streams = {}
    for name, fn, path in (("inference", collect_inference, hfm_path),
                           ("tools", collect_tools, state_path),
                           ("memory", collect_memory, mnemo_path)):
        try:
            streams[name] = fn(path=path, limit=limit) if path else fn(limit=limit)
        except Exception as exc:  # collectors self-guard; this is belt-and-braces
            streams[name] = _degenerate(name, exc)
    try:
        graph = _build_map(streams)
    except Exception as exc:
        graph = {"nodes": [], "links": [], "error": f"{type(exc).__name__}: {exc}"}
    return {**streams, "map": graph,
            "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "panel_error": None}
