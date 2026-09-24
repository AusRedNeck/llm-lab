"""Safety tests for the parallel tokenizer — the "no orphans, no wasted work" guarantees.

These run fast: no corpus encoding, just the bookkeeping that decides whether a
long run wastes hours or loses work.

    python -m pytest tests/test_tokenizer_safety.py -v
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(ROOT, "scripts")
for path in (ROOT, SCRIPTS):
    if path not in sys.path:
        sys.path.insert(0, path)

import token_io                                    # noqa: E402
import tokenize_owt_16k as T                       # noqa: E402


class Args:
    """Stand-in for parsed args — only the fields the helpers read."""
    vocab = os.path.join(ROOT, "data", "bpe_owt16k.json")
    shards_dir = os.path.join(ROOT, "data", "openwebtext", "shards")
    chunk_chars = 500_000
    shard_char_limit = 0
    keep_parts = False
    status_every = 5
    no_probe = True


def write_fake_token_file(path: str, tokens: int, shard_names: list[str]) -> None:
    """A real (tiny) int32 file plus the manifest that describes it."""
    with open(path, "wb") as f:
        f.write(b"\x00" * (4 * tokens))
    token_io.write_meta(path + ".manifest.json", {
        **T.settings_key(Args()), "dst": os.path.abspath(path),
        "shards": {n: {"tokens": tokens // max(1, len(shard_names)), "chars": 1,
                       "seconds": 1.0} for n in shard_names},
        "total_tokens": tokens, "complete": False,
    })


def test_split_blocks_covers_everything_exactly_once():
    shards = [f"s{i:02d}" for i in range(80)]
    blocks = T.split_blocks(shards, 8)
    flat = [s for b in blocks for s in b]
    assert flat == shards, "blocks must stay in order — the merge is a plain concat"
    assert len(blocks) == 8
    assert max(len(b) for b in blocks) - min(len(b) for b in blocks) <= 1


def test_split_blocks_clamps_when_more_workers_than_shards():
    blocks = T.split_blocks(["a", "b"], 8)
    assert [s for b in blocks for s in b] == ["a", "b"]
    assert len(blocks) == 2, "never spawn a worker with nothing to do"


def test_reconcile_trims_a_half_written_shard(tmp_path):
    out = str(tmp_path / "t.bin")
    write_fake_token_file(out, 1000, ["a"])
    # Simulate a crash mid-shard: 300 extra tokens of garbage past the record.
    with open(out, "ab") as f:
        f.write(b"\x01" * (4 * 300))
    records = token_io.read_meta(out + ".manifest.json")["shards"]
    kept = T.reconcile(out, records, ["a"])
    assert os.path.getsize(out) == 4 * 1000, "garbage must be cut, not kept"
    assert kept == records


def test_adopt_existing_split_keeps_the_prefix(tmp_path):
    """A serial run's 3 finished shards must become part00, not be re-encoded."""
    out = str(tmp_path / "out.bin")
    shards = [str(tmp_path / f"s{i}.txt") for i in range(5)]
    for s in shards:
        open(s, "w").write("x")
    write_fake_token_file(out, 4 * 900, ["s0.txt", "s1.txt", "s2.txt"])

    done = T.adopt_existing_split(out, shards, Args())
    assert done == 3, "three finished shards are three you never pay for again"
    part = out + ".part00.bin"
    assert os.path.exists(part)
    assert not os.path.exists(out), "the original name must be free for the merge"
    # And the scan must see it as three shards already covered.
    parts, gaps = T.scan_parts(out, shards)
    assert [p["shards"] for p in parts] == [["s0.txt", "s1.txt", "s2.txt"]]
    assert gaps == ["s3.txt", "s4.txt"]


def test_adopt_refuses_a_non_contiguous_set(tmp_path):
    """Shards 0 and 2 done but not 1: those bytes can't be a prefix, so don't pretend."""
    out = str(tmp_path / "out.bin")
    shards = [str(tmp_path / f"s{i}.txt") for i in range(4)]
    for s in shards:
        open(s, "w").write("x")
    write_fake_token_file(out, 4 * 100, ["s0.txt", "s2.txt"])
    assert T.adopt_existing_split(out, shards, Args()) == 0
    assert os.path.exists(out), "left untouched for inspection"


def test_scan_parts_orders_by_corpus_position_and_finds_gaps(tmp_path):
    """Parts must merge in corpus order, and missing shards must be reported."""
    out = str(tmp_path / "out.bin")
    names = [f"s{i}.txt" for i in range(10)]
    shards = [str(tmp_path / n) for n in names]
    for s in shards:
        open(s, "w").write("x")
    # Blocks written back-to-front on purpose: 4-6 first, then 0-2, then 8-9.
    for idx, block in ((3, names[4:7]), (1, names[0:3]), (5, names[8:10])):
        write_fake_token_file(f"{out}.part{idx:02d}.bin", 4 * 100 * len(block), block)

    parts, gaps = T.scan_parts(out, shards)
    assert [p["shards"] for p in parts] == [names[0:3], names[4:7], names[8:10]], \
        "parts must come back sorted by their first shard"
    assert gaps == [names[3], names[7]]


def test_scan_parts_ignores_a_part_holding_non_contiguous_shards(tmp_path):
    """The reorder trap: a part whose shards aren't adjacent can't be concatenated."""
    out = str(tmp_path / "out.bin")
    names = [f"s{i}.txt" for i in range(6)]
    shards = [str(tmp_path / n) for n in names]
    for s in shards:
        open(s, "w").write("x")
    write_fake_token_file(f"{out}.part00.bin", 4 * 200, [names[0], names[3]])

    parts, gaps = T.scan_parts(out, shards)
    assert parts == [], "a scrambled part must be set aside, not merged"
    assert gaps == names, "so every shard is still on the to-do list"


def test_scan_parts_refuses_overlapping_parts(tmp_path):
    out = str(tmp_path / "out.bin")
    names = [f"s{i}.txt" for i in range(4)]
    shards = [str(tmp_path / n) for n in names]
    for s in shards:
        open(s, "w").write("x")
    write_fake_token_file(f"{out}.part00.bin", 4 * 200, names[0:2])
    write_fake_token_file(f"{out}.part01.bin", 4 * 200, names[1:3])

    parts, gaps = T.scan_parts(out, shards)
    assert len(parts) == 1, "only one of the overlapping parts may be trusted"
    assert gaps == names[2:]


def test_merge_verifies_byte_count_and_refuses_on_mismatch(tmp_path):
    out = str(tmp_path / "merged.bin")
    part = str(tmp_path / "merged.bin.part00.bin")
    write_fake_token_file(part, 500, ["s0.txt"])
    # Manifest claims 500 tokens but the file only holds 400 -> must not pass.
    token_io.write_meta(part + ".manifest.json", {
        **T.settings_key(Args()), "dst": os.path.abspath(part),
        "shards": {"s0.txt": {"tokens": 400, "chars": 1, "seconds": 1.0}},
        "total_tokens": 400, "complete": False,
    })
    rc = T.merge_parts([{"part": part}], out, [str(tmp_path / "s0.txt")], Args())
    assert rc == 1, "a byte-count mismatch must fail loudly"
    assert not os.path.exists(out), "never publish a file that failed verification"


def test_reap_orphans_kills_only_the_recorded_process(tmp_path):
    """The exact failure Shane called out: a survivor from a crashed prior run."""
    out = str(tmp_path / "out.bin")
    decoy = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    try:
        import psutil
        started = psutil.Process(decoy.pid).create_time()
        # A stale PID record (wrong start time) must NOT be killed: PIDs get reused.
        token_io.write_meta(T.workers_file(out), {"workers": [
            {"pid": decoy.pid, "block": "0-9", "started": started},
            {"pid": decoy.pid, "block": "10-19", "started": started - 3600},
        ]})
        time.sleep(0.2)
        killed = T.reap_orphans(out)
        assert killed == 1, "the live orphan is reaped once"
        time.sleep(0.5)
        assert decoy.poll() is not None, "the orphan must actually be gone"
        assert not os.path.exists(T.workers_file(out))
    finally:
        if decoy.poll() is None:
            decoy.kill()


def test_worker_mode_never_spawns_children(tmp_path):
    """A worker must be a leaf process, or a crashed parent leaves a tree behind."""
    cmd = [sys.executable, "-u", os.path.join(SCRIPTS, "tokenize_owt_16k.py"),
           "--vocab", os.path.join(ROOT, "data", "bpe_owt16k.json"),
           "--shards-dir", os.path.join(ROOT, "data", "openwebtext", "shards"),
           "--out", str(tmp_path / "w.bin"), "--worker-id", "0",
           "--shard-from", "0", "--shard-to", "1", "--shard-char-limit", "200000",
           "--no-probe"]
    env = {**os.environ, "PYTHONPATH": ROOT}
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "worker 0:" in r.stdout
    assert "parallel" not in r.stdout, "a worker must not enter the parent path"
    assert os.path.getsize(str(tmp_path / "w.bin")) > 0
