#!/usr/bin/env python3
"""OWT consistency + contamination scan: shard/concat sizes, bin holes, HTML rate."""
import os, re, pathlib
import pyarrow.parquet as pq

SH = pathlib.Path("D:/Projects/llm-lab/data/openwebtext/shards")
SRC = pathlib.Path("D:/Projects/llm-lab/data/openwebtext/plain_text")
COMB = pathlib.Path("D:/Projects/llm-lab/data/openwebtext_combined.txt")
BIN = pathlib.Path("D:/Projects/llm-lab/data/openwebtext_combined_bpe_owt16k.bin")

shards = sorted(SH.glob("*.txt"))
s_sum = sum(f.stat().st_size for f in shards)
c_size = COMB.stat().st_size
print(f"shards: {len(shards)} files, sum={s_sum/1e9:.3f}GB")
print(f"combined: {c_size/1e9:.3f}GB  delta_vs_shards={(c_size-s_sum)/1e6:.1f}MB "
      f"(expect +80*2B separators = {(len(shards)*2)/1e3:.1f}KB)")

# bin hole check
import json
b_size = BIN.stat().st_size
b_tok = b_size // 4
man = json.load(open(str(BIN) + ".manifest.json"))
claimed = sum(r["tokens"] for r in man["shards"].values())
print(f"bin: {b_size/1e9:.3f}GB = {b_tok:,} toks, manifest claims {claimed:,}, "
      f"delta={b_tok-claimed:+,} {'OK' if b_tok==claimed and b_size%4==0 else 'HOLE!'}")

# HTML + URL + short-doc rate across ALL parquets (head-scan 2k chars, cheap)
html_re = re.compile(r"<(html|div|span|script|style|a\s|img|iframe|br\s?/|p[\s>])", re.I)
files = sorted(SRC.glob("*.parquet"))
t_docs = t_html = t_url = 0
for f in files:
    tbl = pq.read_table(str(f), columns=["text"])
    texts = tbl.column("text").to_pylist()
    del tbl
    for t in texts:
        t_docs += 1
        head = t[:2000] if t else ""
        if html_re.search(head):
            t_html += 1
        if "http://" in head or "https://" in head:
            t_url += 1
    print(f"  {f.name}: cum_docs={t_docs:,} html={t_html:,} url={t_url:,}", flush=True)
print(f"TOTAL docs={t_docs:,} html_head_rate={t_html/t_docs*100:.2f}% url_head_rate={t_url/t_docs*100:.2f}%")
