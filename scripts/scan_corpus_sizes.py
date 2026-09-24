"""Ask the HF datasets-server how big each candidate corpus actually is.

Sizes come back per config, which matters: most of these repos ship the full
dataset plus small "sample/10BT"-style subsets, and those subsets are often the
only practical thing to download.
"""
import json
import urllib.error
import urllib.request

CANDIDATES = [
    "cerebras/SlimPajama-627B",
    "DKYoon/SlimPajama-6B",
    "HuggingFaceFW/fineweb",
    "HuggingFaceFW/fineweb-edu",
    "HuggingFaceFW/fineweb-2",
    "HuggingFaceFW/finemath",
    "HuggingFaceFW/finephrase",
    "HuggingFaceFW/finewiki",
    "HuggingFaceTB/cosmopedia",
    "HuggingFaceTB/cosmopedia-v2",
    "togethercomputer/RedPajama-Data-1T",
    "togethercomputer/RedPajama-Data-1T-Sample",
    "togethercomputer/RedPajama-Data-v2",
    "allenai/dolma",
    "allenai/OLMo-mix-1124",
    "allenai/OLMoE-mix-0924",
    "allenai/c4",
    "allenai/peS2o",
    "monology/pile-uncopyrighted",
    "EleutherAI/pile",
    "bigcode/the-stack-v2",
    "bigcode/the-stack",
    "nvidia/Nemotron-CC",
    "nvidia/Nemotron-CC-Math-v1",
    "open-web-math/open-web-math",
    "mlfoundations/dclm",
    "Skylion007/openwebtext",
]

GB = 1024 ** 3


def size_of(repo):
    url = f"https://datasets-server.huggingface.co/size?dataset={urllib.parse.quote(repo)}"
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            return json.load(r)["size"]
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}"}
    except Exception as e:                       # network/timeout
        return {"error": type(e).__name__}


out = []
for repo in CANDIDATES:
    s = size_of(repo)
    if "error" in s:
        out.append({"repo": repo, "error": s["error"]})
        print(f"{repo:45s} {s['error']}", flush=True)
        continue
    total = s["dataset"]["num_bytes_original_files"]
    rows = s["dataset"]["num_rows"]
    configs = sorted(s.get("configs", []), key=lambda c: -c["num_bytes_original_files"])[:5]
    rec = {"repo": repo, "gb": total / GB, "rows": rows,
           "configs": [{"config": c["config"], "gb": c["num_bytes_original_files"] / GB,
                        "rows": c["num_rows"]} for c in configs]}
    out.append(rec)
    print(f"{repo:45s} {total/GB:9.1f} GiB  {rows/1e6:10.1f}M rows", flush=True)
    for c in rec["configs"]:
        print(f"    - {c['config']:32s} {c['gb']:8.1f} GiB  {c['rows']/1e6:10.1f}M rows",
              flush=True)

with open("D:/Projects/llm-lab/data/corpus_candidates.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nwrote data/corpus_candidates.json")
