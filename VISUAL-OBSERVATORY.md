# LLM Observatory (visualization + inference)

The local Observatory combines the experiment decision board with interactive
checkpoint inspection. It is a local research tool, not a hosted service.

## Start it

From the repository root, with the project environment active:

```bash
python -m viz.server --host 127.0.0.1 --port 8787
```

Open <http://127.0.0.1:8787/>. The server binds to loopback by default; keep it
there. The UI and API are unauthenticated and are not intended to be exposed to
a network. `GET /api/health` is a lightweight health check.

## What it shows

- **Runs and experiment arms:** current run summaries, live/stale status,
  validation metrics and experiment registry verdicts. Run details are loaded
  on demand; the server refreshes its cached summary when loss logs or
  `experiments.json` change.
- **Checkpoint inference:** list available checkpoints and devices, load one
  checkpoint onto CPU or an available accelerator, inspect model status, and
  unload it when finished. Only one model is resident in the inference service.
- **Generation:** submit a prompt and sampling settings, then poll the returned
  generation ID for incremental output. Generation is serialized with model
  operations so concurrent requests do not mutate a model simultaneously.
- **Trace / X-ray:** inspect a prompt's token-level and per-layer signals,
  including hidden states, attention, logits/probabilities and the captured
  attention/FFN contributions. These are diagnostic measurements, not causal
  explanations by themselves.
- **Checkpoint comparison:** compare two checkpoints using the same prompt and
  inspection path; comparisons are not a substitute for a controlled evaluation
  corpus or a frozen metric.

The server reuses the existing dashboard run/arm parsing and the training
pipeline's tokenizer dispatch for generation. Checkpoint loading validates
checkpoint IDs against the checkpoint root; do not treat that as a reason to
place untrusted checkpoint files in the directory.

## Open-weight models on Apple MPS

The active target is Qwen3-1.7B, registered as `hf:qwen3-1.7b` from
`data/incoming/qwen3-1.7b/`. Its pinned revision is
`70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` (Apache-2.0); both safetensors
shards match the Hub SHA-256 metadata. The config identifies a 28-layer
`Qwen3ForCausalLM`: hidden 2048, intermediate 6144, 16 query heads, 8 KV heads,
vocab 151936, native context 40960.

Qwen3-4B was the initial size target. It loaded on this 24 GB Mac in BF16, but
swap grew by about 6.5 GiB against the predeclared <1 GiB gate in the first trial,
which began with 2.5 GiB already in use. After background memory pressure eased,
a guarded retry on 2026-11-16 loaded the same snapshot as BF16 on MPS, generated
8 tokens, and unloaded cleanly: swap started at 1,125.25 MiB and showed no growth;
memory-pressure free fell from 82% to 48% during load/generation and recovered to
81%; MPS driver allocation peaked at 8,054,128,640 bytes and returned to 851,968
bytes after unload. The retry passed the load/generation gate under that observed
system state; it does not prove long-context or Think/Neuro trace behavior for 4B.
It was unloaded before further inference; its verified files remain local with
measurements in `data/incoming/qwen3-4b/model-manifest.json`.

The catalog validates `model.safetensors.index.json`, the Qwen config, and every
referenced shard. Missing shards, traversal paths, and symlinks outside the
model directory keep the snapshot out of the catalog. Loading is local-only
with `trust_remote_code=False`.

On MPS, the loader probes BF16 then FP16 using Qwen3-1.7B-sized projection,
norm, and attention-softmax operations. It loads in the supported low
precision and holds the shared trainer lock until explicit unload. The UI/API
reports device, dtype, runtime/native context, and live MPS driver allocation
versus the recommended maximum. Runtime generation is capped at 4,096 tokens;
the snapshot's native 40,960-token context is not validated by that cap. Qwen
prompts use its chat template with thinking disabled; generated text contains
only new tokens.

Trace is a separate opt-in path capped at 128 tokens. It forces eager attention
and rejects absent or malformed attention, wrong layer/head/sequence shapes,
non-finite or out-of-range values, non-normalized rows, or future-token
attention before any visual is drawn. Think shows measured attention to context
and hidden norms; FFN internals remain explicitly unavailable.

**Verified on this Mac:** Qwen3-1.7B loaded on MPS as BF16 in 3.784 s. MPS
driver allocation peaked at 4,550,230,016 bytes against a 19,069,665,280-byte
recommended maximum, then returned to 458,752 bytes after unload; the shared
MPS lock released. Memory pressure stayed outside warning/critical and swap
usage did not grow (5,551.38 MiB before load; 5,149.12 MiB after generation and
trace). A 32-token generation completed in 2.132 s (15.01 tokens/s). Two real
26-token Think/Neuro traces passed validation: 28 layers, 16 query heads each.
The Qwen3-4B short-generation retry passed; its Think/Neuro trace path remains
unvalidated. Qwen3-1.7B is the registered and visually verified model.

**Visual status:** Think's data-backed attention bars render and explicitly
mark FFN signals unavailable. Neuro draws all 28 layers, but the head-node band
is dense and hard to read at the current viewport. Plumbing is verified; Neuro
large-model legibility remains an open visual-design task.

## HTTP API

All endpoints are on the local server. JSON POST bodies use
`Content-Type: application/json`.

| Method | Route | Purpose |
|---|---|---|
| GET | `/api/health` | Health check |
| GET | `/api/dashboard` | Compact run and arm summary |
| GET | `/api/runs/<name>` | One run's details |
| GET | `/api/checkpoints` | Checkpoint choices |
| GET | `/api/devices` | Available inference devices |
| GET | `/api/model/status` | Loaded-model status |
| POST | `/api/model/load` | Load `{ "checkpoint_id": "…", "device": "cpu" }` |
| POST | `/api/model/unload` | Unload the resident model (`{}` body) |
| POST | `/api/generations` | Start generation with prompt/settings |
| GET | `/api/generations/<id>?after=0` | Fetch output/events after an event index |
| POST | `/api/trace` | Inspect a prompt on the loaded model |
| POST | `/api/compare` | Compare checkpoints |
| GET | `/api/activity` | Inference, tool-call, and memory-access streams + map nodes |

POST routes are limited to the listed inference actions; other write methods and
unknown POST routes are rejected. Request bodies are capped at 16 KiB. Errors
use HTTP 400 for invalid input, 404 for missing resources, 409 for unavailable
or conflicting inference state, and 500 for unexpected failures.

## Activity workspace

The third rail view ("Activity") answers *what is the agent doing right now* —
one neural-map canvas plus seven feed panels:

- **Inference** — model calls from the HFM routing proxy (`request_log`).
- **Tool calls** — invocation/result pairs from Hermes `state.db`
  (`messages` joined to `sessions` for profile/session titles).
- **Memory access** — recent and most-recalled Mnemosyne memories, plus
  consolidation and graph stats.
- **Cron** — recent executions from `cron/executions.db` (job names joined
  from `cron/jobs.json`) plus open-incident count.
- **Kanban** — `task_events` joined to `tasks` from `hermes/kanban.db`, plus
  board status counts.
- **Mesh** — live `/health` probes of `config.yaml`'s `bot_peers` (0.4s
  timeout; a sleeping box reads DOWN, which is correct, not an error).
- **Models** — the checkpoints we are making: `checkpoints/*.pt` newest
  first, run liveness from `runs/*/loss.jsonl` mtime (<120s = live).

All sources are **read-only** (SQLite or file stat) with independent
degradation: a missing or locked source marks its own stream `OFFLINE` and
never fails the payload (`GET /api/activity` always returns 200 with
per-stream `ok` flags). Sources are machine-local and overridable by env:
`HFM_DB`, `HERMES_STATE_DB`, `MNEMOSYNE_DB`, `HERMES_CRON_DB`,
`HERMES_JOBS_JSON`, `HERMES_KANBAN_DB`, `HERMES_CONFIG`, `LLM_LAB_RUNS`,
`LLM_LAB_CHECKPOINTS`.

Honesty caveats baked into the UI:

- The inference feed covers **HFM-routed calls only** — direct-pinned
  provider traffic bypasses `request_log` (the known burn blind-spot).
- When the HFM proxy is down the file is still readable; the badge says
  `FILE` instead of `LIVE` (freshness = newest row within 60s). When data is
  stale the collector probes HFM `/healthz` once (0.4s) so the footer can
  distinguish "proxy down" from "up but idle" — healthz can block on provider
  probes of dead ollama boxes, hence the short timeout.
- HFM itself is **intentionally offline by standing decision** — the panel
  reads its historical `request_log`; it is not a reason to run HFM.

The map is a hand-rolled 2D canvas force simulation (no chart lib, no CDN,
matching the lab's zero-dependency rule): pods per stream, session hubs for
tool events, event nodes colored lime/orange/cyan; clicking a node fills the
inspector and scroll-highlights its feed row. The animation pauses when the
view is hidden; the server caches the payload for 10s (state.db is written
constantly, so a mtime-signature cache would rebuild on every poll).

**The registry:** the Phase-2 refactor landed — panels are config, not
code. Adding a stream takes three steps: (1) write a collector in
`viz/activity.py` returning the degraded-safe shape `{ok, error, events,
stats}`; (2) add its entry to `collect_activity()`'s `jobs` dict and a
`POD_LABELS` key; (3) add one `PANEL_DEFS` entry in `viz/ui/app.js`
(columns, row mapper, stats cards) plus the matching panel markup in
`index.html`. Per-source degradation, the map pod, row-key highlight, and
the feed rendering all come from the registry. The mesh collector is the
only one that does network I/O — keep its timeout under a second because it
runs inside the shared 10s TTL cache.

## Neuro workspace (model topology)

The fourth rail view ("Neuro") draws a loaded checkpoint as a living circuit —
the Oct-5 plan's Part A: **layer pods** (teal rings), **block hubs** (green,
radius scales with hidden norm), **head somata** (orange, radius = peak
attention, edge thickness = concentration), **FFN somata** (red, radius =
activation RMS), and **residual synapses** (dashed gold, layer to layer). All
numbers come from one opt-in `/api/trace` pass (the same payload the Think
workbench renders as bars) — the map is another view of that trace, not a
second capture path.

- Self-serve: prompt + INSPECT in the view; FOCUS TOKEN re-reads the trace
  client-side (arrays are per-token already — no re-capture).
- **Open-weights models work today** via Think's `hf:` checkpoints: those
  backends omit FFN internals, so FFN somata render as dashed "not exposed"
  outlines and the notice says so — honest absence over fake values.
- **Cloud models**: provider APIs expose no activation data; a future adapter
  could feed static architecture metadata only. Interface = `buildNeuro(trace)`
  — anything producing the same `{layers:[attention_by_head, hidden_norms, ...]}`
  shape plugs straight in.
- Pause/drift toggle in the panel head; animation pauses when the view is
  hidden. Attention is a routing measurement, not causal proof.

## Architecture and constraints

- `viz/server.py` serves the static UI and JSON API using the Python standard
  library. `viz/ui/` contains the browser client. `viz/activity.py` holds the
  three activity-stream collectors behind `GET /api/activity`.
- `viz/inference_service.py` owns checkpoint loading, device selection,
  generation, tracing and comparison; `viz/trace.py` formats trace results.
- `model/transformer.py` and `model/block.py` expose optional captured
  intermediate signals. The ordinary forward path remains the training path;
  capture is opt-in and detached from autograd.
- Server defaults point at this repository's `runs/`, `experiments.json`, and
  `checkpoints/`; tests can inject alternate directories and a fake inference
  service.
- This tool is for interactive exploration. For scientific claims, use
  reproducible eval scripts, record the exact checkpoint/config/data, and
  preserve comparability controls.

## Verification

Run the suite from the project environment:

```bash
python -m pytest -q
```

The API and inference paths have focused tests in `tests/test_activity.py`
(all seven activity collectors, map assembly, per-source degradation),
`tests/test_viz_server.py`,
`tests/test_inference_service.py`, `tests/test_inference_tokenizer.py`, and
`tests/test_trace.py`.
