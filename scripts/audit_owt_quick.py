#!/usr/bin/env python3
"""Quick OWT shard health probe: parquet row counts + txt cleanliness sample."""
import re, pathlib, random
import pyarrow.parquet as pq

SRC = pathlib.Path("D:/Projects/llm-lab/data/openwebtext/plain_text")
SHARDS = pathlib.Path("D:/Projects/llm-lab/data/openwebtext/shards")

files = sorted(SRC.glob("*.parquet"))
print(f"parquet files: {len(files)}")
total_rows = 0
for f in files:
    pf = pq.ParquetFile(str(f))
    total_rows += pf.metadata.num_rows
print(f"total parquet rows (docs): {total_rows:,}")

# sample 3 parquets for doc-level stats
import pyarrow.compute as pc
html_re = re.compile(r"<(html|div|span|script|style|a\s|img|iframe|br\s?/|p>)", re.I)
for name in ["train-00000-of-00080.parquet", "train-00040-of-00080.parquet", "train-00079-of-00080.parquet"]:
    pf = pq.ParquetFile(str(SRC / name))
    tbl = pf.read(columns=["text"])
    texts = tbl.column("text").to_pylist()
    n = len(texts)
    lens = [len(t) if t else 0 for t in texts]
    lens_sorted = sorted(lens)
    short = sum(1 for L in lens if L < 100)
    empty = sum(1 for L in lens if L == 0)
    html = sum(1 for t in texts if t and html_re.search(t[:2000]))
    print(f"{name}: docs={n:,} empty={empty} short<100={short} html_head_hit={html} "
          f"len p50={lens_sorted[n//2]} p10={lens_sorted[n//10]} p90={lens_sorted[9*n//10]} max={max(lens)}")

# txt shard cleanliness sample: 8 shards spread across corpus
random.seed(0)
shard_files = sorted(SHARDS.glob("*.txt"))
picks = [shard_files[i] for i in [0, 10, 20, 30, 40, 50, 60, 70]]
print(f"\ntxt sample: {[p.name for p in picks]}")
ctrl_re = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
for p in picks:
    data = p.read_bytes()[:5_000_000]
    txt = data.decode("utf-8", errors="replace")
    ctrl = len(ctrl_re.findall(txt))
    repl = txt.count("\ufffd")
    html = len(re.findall(r"</(div|span|p|a|html)>", txt, re.I))
    urls = len(re.findall(r"https?://", txt))
    print(f"{p.name}: ctrl={ctrl} repl={repl} html_close={html} urls={urls} per5MB")
