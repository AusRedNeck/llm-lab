"""Tests for viz.activity — the three activity-stream collectors."""
import json
import sqlite3
import time

import pytest

from viz.activity import (collect_activity, collect_cron, collect_inference,
                          collect_kanban, collect_memory, collect_mesh,
                          collect_models, collect_tools)


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


# ---------------------------------------------------------------- cron
def _make_cron_db(root):
    path = root / "executions.db"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE executions (id TEXT, job_id TEXT, status TEXT,"
                " started_at TEXT, finished_at TEXT, error TEXT, delivery_outcome TEXT)")
    con.executemany("INSERT INTO executions VALUES (?,?,?,?,?,?,?)", [
        ("e1", "job1", "completed", "2026-10-07T14:00:28-07:00", "2026-10-07T14:00:38-07:00", None, "suppressed"),
        ("e2", "job1", "failed", "2026-10-07T13:00:28-07:00", None, "boom", None),
        ("e3", "job2", "completed", "2026-10-07T12:00:28-07:00", "2026-10-07T12:00:30-07:00", None, "delivered"),
    ])
    con.execute("CREATE TABLE cron_incidents (id TEXT, closed_at TEXT)")
    con.execute("INSERT INTO cron_incidents VALUES ('i1', NULL)")
    con.execute("INSERT INTO cron_incidents VALUES ('i2', '2026-10-06T00:00:00-07:00')")
    con.commit(); con.close()
    jobs = root / "jobs.json"
    jobs.write_text(json.dumps({"jobs": [
        {"id": "job1", "name": "Daily news"},
        {"id": "job2", "name": "Weekly digest"},
    ]}), encoding="utf-8")
    return path, jobs


def test_cron_collector_maps_names_and_stats(tmp_path):
    db, jobs = _make_cron_db(tmp_path)
    payload = collect_cron(path=db, jobs_path=jobs)
    assert payload["ok"] is True
    assert [e["id"] for e in payload["events"]] == ["e3", "e2", "e1"]  # rowid DESC
    assert payload["events"][1]["job"] == "Daily news"
    assert payload["events"][1]["status"] == "failed"
    assert payload["events"][1]["error"] == "boom"
    stats = payload["stats"]
    assert stats["total"] == 3 and stats["failed"] == 1
    assert stats["open_incidents"] == 1
    assert stats["by_status"] == {"completed": 2, "failed": 1}


def test_cron_collector_missing_db_degrades(tmp_path):
    payload = collect_cron(path=tmp_path / "nope.db", jobs_path=tmp_path / "nope.json")
    assert payload["ok"] is False and payload["events"] == []


# ---------------------------------------------------------------- kanban
def _make_kanban_db(root):
    path = root / "kanban.db"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE tasks (id TEXT, title TEXT, status TEXT, assignee TEXT)")
    con.executemany("INSERT INTO tasks VALUES (?,?,?,?)", [
        ("t1", "Ship the dashboard", "done", "engineering"),
        ("t2", "Write copy", "in_progress", "marketing"),
        ("t3", "Old thing", "archived", None),
    ])
    con.execute("CREATE TABLE task_events (id INTEGER PRIMARY KEY, task_id TEXT,"
                " run_id INTEGER, kind TEXT, payload TEXT, created_at REAL)")
    con.executemany("INSERT INTO task_events VALUES (?,?,?,?,?,?)", [
        (1, "t1", 10, "completed", "{}", 1791400000.0),
        (2, "t2", 11, "started", "{}", 1791400500.0),
    ])
    con.commit(); con.close()
    return path


def test_kanban_collector_reads_events_and_status_counts(tmp_path):
    payload = collect_kanban(path=_make_kanban_db(tmp_path))
    assert payload["ok"] is True
    assert [e["kind"] for e in payload["events"]] == ["started", "completed"]
    assert payload["events"][0]["title"] == "Write copy"
    assert payload["events"][0]["assignee"] == "marketing"
    stats = payload["stats"]
    assert stats["total_tasks"] == 3
    assert stats["open"] == 1  # done + archived are not open
    assert stats["by_status"] == {"done": 1, "in_progress": 1, "archived": 1}


def test_kanban_collector_missing_db_degrades(tmp_path):
    payload = collect_kanban(path=tmp_path / "gone.db")
    assert payload["ok"] is False and payload["events"] == []


# ---------------------------------------------------------------- mesh
def test_mesh_collector_no_peers_is_healthy_but_empty():
    payload = collect_mesh(peers=[])
    assert payload["ok"] is True
    assert payload["stats"] == {"up": 0, "total": 0}


def test_mesh_collector_unreachable_peer_is_an_event_not_an_error():
    payload = collect_mesh(peers=[("dead", "http://127.0.0.1:1")], timeout=0.3)
    assert payload["ok"] is True
    assert payload["stats"]["up"] == 0
    event = payload["events"][0]
    assert event["ok"] is False and event["error"]


# ---------------------------------------------------------------- models
def _make_models_dirs(root):
    runs = root / "runs"
    (runs / "run_live").mkdir(parents=True)
    (runs / "run_live" / "loss.jsonl").write_text("{}\n", encoding="utf-8")
    (runs / "run_old").mkdir()
    (runs / "run_old" / "loss.jsonl").write_text("{}\n", encoding="utf-8")
    old = time.time() - 3600
    import os
    os.utime(runs / "run_old" / "loss.jsonl", (old, old))
    ckpts = root / "checkpoints"
    ckpts.mkdir()
    (ckpts / "exp1_best.pt").write_bytes(b"x" * 2048)
    (ckpts / "exp1_step900.pt").write_bytes(b"x" * 1024)
    return runs, ckpts


def test_models_collector_lists_checkpoints_and_run_liveness(tmp_path):
    runs, ckpts = _make_models_dirs(tmp_path)
    payload = collect_models(runs_dir=runs, ckpt_dir=ckpts)
    assert payload["ok"] is True
    names = [e["name"] for e in payload["events"]]
    assert set(names) == {"exp1_best.pt", "exp1_step900.pt"}
    kinds = {e["name"]: e["kind"] for e in payload["events"]}
    assert kinds["exp1_best.pt"] == "best" and kinds["exp1_step900.pt"] == "step"
    stats = payload["stats"]
    assert stats["runs"] == 2 and stats["live_runs"] == 1
    assert stats["checkpoints"] == 2


def test_models_collector_missing_dirs_degrade(tmp_path):
    payload = collect_models(runs_dir=tmp_path / "r", ckpt_dir=tmp_path / "c")
    assert payload["ok"] is False and payload["events"] == []
    # but an existing, empty checkpoints dir is honest zeros, not an error
    empty = tmp_path / "c3"
    empty.mkdir()
    healthy = collect_models(runs_dir=tmp_path / "r", ckpt_dir=empty)
    assert healthy["ok"] is True and healthy["stats"]["checkpoints"] == 0


# ---------------------------------------------------------------- aggregate
def test_collect_activity_assembles_map_and_never_raises(tmp_path):
    cron_db, jobs_json = _make_cron_db(tmp_path)
    runs, ckpts = _make_models_dirs(tmp_path)
    payload = collect_activity(hfm_path=_make_hfm_db(tmp_path),
                               state_path=_make_state_db(tmp_path),
                               mnemo_path=_make_mnemo_db(tmp_path),
                               cron_path=cron_db, jobs_path=jobs_json,
                               kanban_path=_make_kanban_db(tmp_path),
                               peers=[("self", "http://127.0.0.1:1")],
                               runs_dir=runs, ckpt_dir=ckpts)
    assert payload["panel_error"] is None
    for key in ("inference", "tools", "memory", "cron", "kanban", "mesh",
                "models", "map", "updated_at"):
        assert key in payload
    nodes, links = payload["map"]["nodes"], payload["map"]["links"]
    pod_ids = [n["id"] for n in nodes if n["kind"] == "pod"]
    assert set(pod_ids) == {f"pod:{name}" for name in
                            ("inference", "tools", "memory", "cron",
                             "kanban", "mesh", "models")}
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
                               mnemo_path=tmp_path / "c.db",
                               cron_path=tmp_path / "d.db",
                               jobs_path=tmp_path / "e.json",
                               kanban_path=tmp_path / "f.db",
                               peers=[], runs_dir=tmp_path / "r",
                               ckpt_dir=tmp_path / "c2")
    for stream in ("inference", "tools", "memory", "cron", "kanban", "models"):
        assert payload[stream]["ok"] is False, stream
    # only mesh can be healthy with no files (it probes configured peers; none here)
    pods = {n["stream"] for n in payload["map"]["nodes"] if n["kind"] == "pod"}
    assert pods <= {"mesh"}
    assert payload["map"]["links"] == []
