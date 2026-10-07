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

POST routes are limited to the listed inference actions; other write methods and
unknown POST routes are rejected. Request bodies are capped at 16 KiB. Errors
use HTTP 400 for invalid input, 404 for missing resources, 409 for unavailable
or conflicting inference state, and 500 for unexpected failures.

## Architecture and constraints

- `viz/server.py` serves the static UI and JSON API using the Python standard
  library. `viz/ui/` contains the browser client.
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

The API and inference paths have focused tests in `tests/test_viz_server.py`,
`tests/test_inference_service.py`, `tests/test_inference_tokenizer.py`, and
`tests/test_trace.py`.
