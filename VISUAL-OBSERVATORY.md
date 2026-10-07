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
  `FILE` instead of `LIVE` (freshness = newest row within 60s).

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
