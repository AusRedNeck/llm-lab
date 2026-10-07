"""Tests for viz.activity — the three activity-stream collectors."""
import json
import sqlite3
import time

import pytest

from viz.activity import (collect_activity, collect_inference,
                          collect_memory, collect_tools)


# ---------------------------------------------------------------- fixtures
def _make_hfm_db(root):
    path = root / "hfm_state.db"
    con = sqlite3.connect(path)
    con.execute(
        "CREATE TABLE request_log (id INTEGER PRIMARY KEY, ts INTEGER, model TEXT,"
        " provider TEXT, status INTEGER, tokens_prompt INTEGER, tokens_completion INTEGER,"
        " latency_ms INTEGER, cost_cents INTEGER, cached INTEGER, error TEXT)")
    now_ms = int(time.time() * 1000)
    rows = [
        (1, now_ms - 5_000, "glm-4.6", "opencode-go", 200, 100, 50, 1200, 3, 0, None),
        (2, now_ms - 120_000, "glm-4.6", "opencode-go", 200, 90, 40, 900, 2, 1, None),
        (3, now_ms - 200_000, "deepseek-v4", "opencode-direct", 429, 10, 5, 0, 0, 0, "quota"),
    ]
    con.executemany("INSERT INTO request_log VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows)
    con.commit()
    con.close()
    return path


def _make_state_db(root):
    path = root / "state.db"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, title TEXT, profile_name TEXT)")
    con.execute("INSERT INTO sessions VALUES ('sess-1', 'demo session', 'engineering')")
    con.execute(
        "CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT, role TEXT,"
        " content TEXT, tool_calls TEXT, tool_name TEXT, tool_call_id TEXT, timestamp REAL)")
    calls = json.dumps([{"function": {"name": "terminal"}}])
    rows = [
        (1, "sess-1", "user", "do it", None, None, None, 1000.0),
        (2, "sess-1", "assistant", None, calls, None, "tc-1", 1001.0),
        (3, "sess-1", "tool", "output", None, "terminal", "tc-1", 1002.0),
        (4, "sess-1", "assistant", None, calls, None, "tc-2", 1003.0),
        (5, "sess-1", "tool", "output", None, "terminal", "tc-2", 1004.0),
        (6, "sess-1", "assistant", "plain text", None, None, None, 1005.0),
    ]
    con.executemany("INSERT INTO messages VALUES (?,?,?,?,?,?,?,?)", rows)
    con.commit()
    con.close()
    return path


def _make_mnemo_db(root):
    path = root / "mnemosyne.db"
    con = sqlite3.connect(path)
    for table in ("working_memory", "episodic_memory"):
        con.execute(
            f"CREATE TABLE {table} (id TEXT, content TEXT, source TEXT, timestamp TEXT,"
            f" importance REAL, recall_count INTEGER, last_recalled TEXT)")
    con.execute("INSERT INTO working_memory VALUES ('w1','memory one','mnemosyne_remember',"
                "'2026-10-07T10:00:00',0.8,3,'2026-10-07T11:00:00')")
    con.execute("INSERT INTO working_memory VALUES ('w2','memory two','user',"
                "'2026-10-06T10:00:00',0.5,0,NULL)")
    con.execute("INSERT INTO episodic_memory VALUES ('e1','episode one','sleep',"
                "'2026-10-05T10:00:00',0.6,1,NULL)")
    con.execute("CREATE TABLE consolidation_log (id INTEGER PRIMARY KEY, items_consolidated INTEGER,"
                " summary_preview TEXT, created_at TEXT)")
    con.execute("INSERT INTO consolidation_log VALUES (1, 5, 'sum', '2026-10-07T04:30:00')")
    con.execute("CREATE TABLE graph_edges (id INTEGER PRIMARY KEY, source TEXT, target TEXT,"
                " edge_type TEXT, weight REAL, timestamp TEXT, created_at TEXT)")
    con.execute("INSERT INTO graph_edges VALUES (1,'w1','w2','rel',0.7,'2026-10-07','2026-10-07')")
    con.execute("CREATE TABLE triples (id INTEGER PRIMARY KEY, subject TEXT, predicate TEXT,"
                " object TEXT, valid_from TEXT, valid_until TEXT, source TEXT,"
                " confidence REAL, created_at TEXT)")
    con.execute("INSERT INTO triples VALUES (1,'user','prefers','vim','2026-10-01',NULL,'user',1.0,'x')")
    con.commit()
    con.close()
    return path


# ---------------------------------------------------------------- inference
def test_inference_collector_parses_rows_and_stats(tmp_path):
    payload = collect_inference(path=_make_hfm_db(tmp_path))
    assert payload["ok"] is True and payload["error"] is None
    assert len(payload["events"]) == 3
    newest = payload["events"][0]
    assert newest["provider"] == "opencode-go" and newest["model"] == "glm-4.6"
    assert newest["status"] == 200 and newest["cached"] is False
    assert payload["events"][1]["cached"] is True
    assert payload["events"][2]["error"] == "quota"
    stats = payload["stats"]
    assert stats["last_hour_count"] == 3
    assert stats["cost_cents"] == 5
    assert stats["tokens_total"] == 100 + 50 + 90 + 40 + 10 + 5
    assert stats["errors"] == 1
    assert payload["freshness"] == "LIVE"


def test_inference_collector_missing_file_degrades(tmp_path):
    payload = collect_inference(path=tmp_path / "nope.db")
    assert payload["ok"] is False
    assert payload["error"]
    assert payload["events"] == []


def test_inference_old_data_reports_file_freshness(tmp_path):
    path = _make_hfm_db(tmp_path)
    con = sqlite3.connect(path)
    con.execute("UPDATE request_log SET ts = ?", (int((time.time() - 7200) * 1000),))
    con.commit()
    con.close()
    payload = collect_inference(path=path)
    assert payload["ok"] is True
    assert payload["freshness"] == "FILE"


# ---------------------------------------------------------------- tool calls
def test_tools_collector_pairs_invocations_and_results(tmp_path):
    payload = collect_tools(path=_make_state_db(tmp_path))
    assert payload["ok"] is True
    kinds = [e["kind"] for e in payload["events"]]
    assert kinds == ["result", "invocation", "result", "invocation"]  # id DESC
    inv = payload["events"][1]
    assert inv["tool_name"] == "terminal"
    assert inv["kind"] == "invocation"
    assert inv["session_title"] == "demo session"
    assert inv["profile_name"] == "engineering"
    stats = payload["stats"]
    assert stats["total"] == 4
    assert stats["invocations"] == 2
    assert stats["results"] == 2
    assert stats["by_tool"] == {"terminal": 4}


def test_tools_collector_missing_file_degrades(tmp_path):
    payload = collect_tools(path=tmp_path / "absent.db")
    assert payload["ok"] is False and payload["events"] == []


# ---------------------------------------------------------------- memory
def test_memory_collector_reads_tiers_stats_and_graph(tmp_path):
    payload = collect_memory(path=_make_mnemo_db(tmp_path))
    assert payload["ok"] is True
    tiers = [e["tier"] for e in payload["events"]]
    assert set(tiers) == {"working", "episodic"}
    assert len(payload["events"]) == 3
    w1 = next(e for e in payload["events"] if e["id"] == "w1")
    assert w1["recall_count"] == 3
    assert w1["preview"].startswith("memory one")
    stats = payload["stats"]
    assert stats["working"] == 2
    assert stats["episodic"] == 1
    assert stats["recalled_total"] == 3 + 1
    assert stats["consolidations"] == 1
    assert stats["graph_edges"] == 1
    assert stats["triples"] == 1
    assert payload["graph"][0]["edge_type"] == "rel"
    assert payload["graph"][0]["weight"] == 0.7


def test_memory_collector_missing_file_degrades(tmp_path):
    payload = collect_memory(path=tmp_path / "gone.db")
    assert payload["ok"] is False and payload["events"] == []


# ---------------------------------------------------------------- aggregate
def test_collect_activity_assembles_map_and_never_raises(tmp_path):
    payload = collect_activity(hfm_path=_make_hfm_db(tmp_path),
                               state_path=_make_state_db(tmp_path),
                               mnemo_path=_make_mnemo_db(tmp_path))
    assert payload["panel_error"] is None
    for key in ("inference", "tools", "memory", "map", "updated_at"):
        assert key in payload
    nodes, links = payload["map"]["nodes"], payload["map"]["links"]
    pod_ids = [n["id"] for n in nodes if n["kind"] == "pod"]
    assert set(pod_ids) == {"pod:inference", "pod:tools", "pod:memory"}
    # every event node links back to a pod (directly or via a session hub)
    node_ids = {n["id"] for n in nodes}
    link_pairs = {(l["source"], l["target"]) for l in links}
    assert all(s in node_ids and t in node_ids for s, t in link_pairs)
    session_hubs = [n for n in nodes if n["kind"] == "session"]
    assert session_hubs and session_hubs[0]["id"].startswith("session:")
    # event nodes carry the row_key the UI uses to highlight their feed row
    event_nodes = [n for n in nodes if n["kind"] == "event"]
    assert event_nodes and all(n.get("row_key") for n in event_nodes)
    assert any(n["row_key"].startswith("tools:") for n in event_nodes)
    assert any(n["row_key"].startswith("memory:") for n in event_nodes)


def test_collect_activity_with_all_sources_missing_still_returns(tmp_path):
    payload = collect_activity(hfm_path=tmp_path / "a.db",
                               state_path=tmp_path / "b.db",
                               mnemo_path=tmp_path / "c.db")
    assert payload["inference"]["ok"] is False
    assert payload["tools"]["ok"] is False
    assert payload["memory"]["ok"] is False
    assert payload["map"]["nodes"] == [] and payload["map"]["links"] == []
