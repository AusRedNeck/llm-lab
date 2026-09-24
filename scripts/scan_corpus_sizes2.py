"""Round 2: gated / huge / remaining corpus candidates, with a null-size guard."""
import json
import urllib.error
import urllib.parse
import urllib.request

CANDIDATES = [
    "allenai/c4",
    "allenai/peS2o",
    "allenai/dolma",
    "allenai/OLMoE-mix-0924",
    "allenai/OLMo-mix-1124",
    "monology/pile-uncopyrighted",
    "EleutherAI/pile",
    "bigcode/the-stack-v2",
    "bigcode/the-stack",
    "nvidia/Nemotron-CC",
    "nvidia/Nemotron-CC-Math-v1",
    "open-web-math/open-web-math",
    "mlfoundations/dclm",
    "Skylion007/openwebtext",
    "Zyphra/Zyda-2",
    "common-pile/comma_v0.1",
    "HuggingFaceFW/fineweb-edu-score-2",
    "HuggingFaceTB/finemath",
    "cerebras/SlimPajama-627B",
    "togethercomputer/RedPajama-Data-1T",
]
GB = 1024 ** 3


def size_of(repo):
    url = f"https://datasets-server.huggingface.co/size?dataset={urllib.parse.quote(repo)}"
    try:
        with urllib.request.urlopen(url, timeout=40) as r:
            return json.load(r)["size"]
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}"}
    except Exception as e:
        return {"error": type(e).__name__}


out = []
for repo in CANDIDATES:
    s = size_of(repo)
    if "error" in s:
        print(f"{repo:45s} {s['error']}", flush=True)
        out.append({"repo": repo, "error": s["error"]})
        continue
    ds = s.get("dataset", {})
    total = ds.get("num_bytes_original_files") or 0
    rows = ds.get("num_rows") or 0
    cfgs = [c for c in (s.get("configs") or [])
            if c.get("num_bytes_original_files")]
    cfgs.sort(key=lambda c: -c["num_bytes_original_files"])
    print(f"{repo:45s} {total / GB:9.1f} GiB  {rows / 1e6:10.1f}M rows", flush=True)
    rec = {"repo": repo, "gb": total / GB, "rows": rows, "configs": []}
    for c in cfgs[:5]:
        gb = c["num_bytes_original_files"] / GB
        print(f"    - {c['config'][:32]:32s} {gb:8.1f} GiB  "
              f"{(c.get('num_rows') or 0) / 1e6:10.1f}M rows", flush=True)
        rec["configs"].append({"config": c["config"], "gb": gb,
                               "rows": c.get("num_rows") or 0})
    out.append(rec)

with open("D:/Projects/llm-lab/data/corpus_candidates2.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nwrote data/corpus_candidates2.json")
