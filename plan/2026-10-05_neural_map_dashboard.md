# Neural Map Dashboard — Plan Addition

## Part A: LLM-Lab Observatory Neuro-Map (visual language spec)

### Visual language (must match wysie/mnemosyne-dashboard exactly)
- Layout: 3-column dark dashboard
  - Left: nav sidebar (graph views, filters, pause/drift toggle)
  - Center: canvas = drifting neural cloud (Three.js)
  - Right: NEURAL INSPECTOR panel (read-only source detail on click)
- Canvas nodes:
  - Translucent CATEGORY CLUSTER PODS (teal / orange)
  - Glowing green NEURON HUB nodes
  - Glowing red/orange MEMORY SOMA nodes
  - Faint dashed SYNAPSE edges between
- Animation: idle orbital drift (low-speed rotation/repulsion), Pause control
- Selection: active tab/button uses blue-purple gradient glow
- Inspector click → right panel shows read-only detail (count, links, source text)

### llm-lab adaptation (same skin, model data instead of memory)
- Node mapping:
  - layer → TEAL cluster pod
  - block → neuron hub (green glow)
  - head → memory soma (red/orange glow)
  - FFN → memory soma (red/orange glow)
  - MLP / residual stream → synapse edge
- Data source: `D:/Projects/llm-lab/viz/trace.py` captures activations, attention, FFN states per generation step — feed it as the live data adapter.
- Behavior: idle drift when model is not generating; pulse/glow intensity scales with activation magnitude during active generation; hover on a node shows trace numbers.
- Inspector: click any layer/block/head/FFN → right panel shows trace numbers (attention weights, activation norms, FFN intermediate states) read-only.
- Build target: Three.js variant page (wysie's dashboard already ships a /3d route — this maps directly onto it).

### Wiring
- Runs alongside the observatory, same ports, same read-only posture.
- Each machine reads its own local `mnemosyne.db` (no cross-machine DB sharing).
- Install: `hermes plugins install wysie/mnemosyne-dashboard --enable` on each viewer host.

## Part B — Tool-Call View (memory + tool calls)

### Scope
- **Memory half:** already in wysie's dash on both machines. No action.
- **Tool-call half:** NEW. Extend the SAME wysie dash install (no fork, no separate build).

### Extend wysie's dash with a `toolcall` tab
- Add to existing plugin install on both Ronin (8767) and Mac (8765)
- node = tool call (session → step → tool)
- edge = call sequence / dependency
- inspector = arguments + result snippet (read-only)
- data adapter: read from Hermes session logs (already on disk, no new collector)
- **Cost:** low — same visual language, same data flow, no fork/rebuild

### Port note
wysie's dash on Mac = 127.0.0.1:8765; on Ronin = 127.0.0.1:8767. Do NOT rebind either to 0.0.0.0. If Toast needs to view, screenshot or Tailscale FRS local-forward.

---

## Part C — Neuro map on the Think page

**Status: NOT BUILT. Added 2026-10-07.**

### Why this is the right home (verified in code, not assumed)
- Think (`#think-view`) already owns the whole trace lifecycle: checkpoint load → `INSPECT PROMPT` → attention bars + layer signals → checkpoint replay.
- The Neuro view is a fourth rail entry (`data-view="neuro"`) with its **own** prompt textarea and its **own** `INSPECT PROMPT` button — a second set of controls for one dataset.
- They already share a single data path: one opt-in `/api/trace` pass of the resident model. Neuro's own status message says *"Load a checkpoint in Think first — the neuro map renders one opt-in trace of whatever model is resident."* The coupling is already there; only the UI pretends otherwise.
- `neuroFrame()` bails unless `viewName === "neuro"`, so the map today is invisible at exactly the moment you produce the trace it draws.

One trace, two renderings, two buttons, two prompts, two animation loops. Put the map where the trace is made.

### Change
1. Move `#neuro-map` + `#neuro-inspector` into Think's `.trace-panel` `.trace-views`, as a third article beside **ATTENTION TO CONTEXT** and **LAYER SIGNALS AT SELECTED TOKEN** (label: `STRUCTURE`).
2. Delete `#neuro-prompt` and `#neuro-inspect` — Think's `#prompt-input` and `#inspect-trace` drive both renderings off the one payload.
3. `FOCUS TOKEN` reuses Think's existing `#trace-token` select; `buildNeuro(trace, focus)` already takes focus as its second arg, so no signature change.
4. Animation rule matches the Activity map: `requestAnimationFrame` only while `#think-view` is visible **and** `neuroTrace` is non-null; otherwise cancel.
5. Keep honest absence as-is — `hf:` checkpoints render FFN somata as dashed "not exposed" outlines with the notice saying so.
6. **Decision for Shane:** once this lands, `data-view="neuro"` is redundant (one rail button + one `.toggle("hidden", …)` line to remove). Retire it, or keep it as a full-canvas mode. Retire is the simpler call; keeping it means two canvases rendering the same trace.

### Not in scope
`buildNeuro(trace)` stays the interface boundary. A cloud-model adapter that feeds static architecture metadata still plugs in there; nothing about the embed changes that.

---

## Part D — Activity topology: per-stream filters

**Status: NOT BUILT. Added 2026-10-07.**

### Current state (verified)
- The legend in `.map-panel` head lists all seven colors but is **display-only** — there is no control anywhere in the Activity view.
- All seven streams always render: `PANEL_DEFS.forEach(def => renderStream(def, payload[def.key]))` fills the panels, and `drawMap(payload.map)` draws every pod.
- `POD_LABELS` (`viz/activity.py:42`) and `PANEL_DEFS` (`viz/ui/app.js:376`) carry the identical key set — `inference, tools, memory, cron, kanban, mesh, models`. That pairing is the filter's source of truth.

### Change — client-side only
1. Toggle chips in the `.map-panel` head, one per `PANEL_DEFS` key, default **ON**; the existing color swatch doubles as the chip so legend and filter are one control.
2. A stream turned OFF:
   - hides its `#<key>-panel`;
   - drops its pod **and** its event nodes from `mapLayout()` *before* the `ids` set is built — `links.filter(l => ids.has(l.source) && ids.has(l.target))` then discards the dangling edges for free, no link bookkeeping;
   - greys its legend chip.
3. Persist to `localStorage["activity.streams"]` so kanban/mesh/cron staying off survives a reload instead of resetting every poll cycle.
4. With every stream off, `drawMap` receives `{nodes:[],links:[]}`: `pods.length + 1` cannot divide by zero and the node loop is a no-op, so an empty canvas is the expected paint — verify rather than assume.

### Non-goals (say no to these)
- **No `?streams=` on `GET /api/activity`.** The 10s TTL cache, `ActivityStore`'s mtime signature, and per-source degradation stay byte-for-byte as they are. Adding a query dimension multiplies the cache keys for a UI preference.
- The server keeps collecting all seven. Collectors are read-only SQLite already inside the cache window; mesh is the only network I/O and it is already throttled by that same 10s TTL. Skipping collectors server-side is an optimization we have no evidence we need.
- Filters are a render concern. Panels, map, and inspector all read the same enabled-set — no separate "hide from map but keep the panel" mode.

### Verification
The suite is Python; there is no JS harness. Gate = full `pytest tests/ -q` green (no server regression) plus a manual pass: toggle each stream, confirm panel + pod + edges disappear together, reload and confirm persistence, click a surviving node and confirm the inspector still resolves.

---

## Part E — Node linking (the edge model)

**Status: WAS MISSING AS A FIRST-CLASS SECTION. Added 2026-10-07.**

### The answer to "did linking the nodes make it into the plan?"
Partly. What the plan had before today was edge **decoration**, never an edge **model**:
- Part A: "Faint dashed SYNAPSE edges between" (styling), "MLP / residual stream → synapse edge" (node mapping), inspector shows "count, links, source text".
- Part B: "edge = call sequence / dependency" (one line, tool-call view).

No section said what a link *is*, where it comes from, or how it is derived. The Activity plan (`.hermes/plans/2026-10-07_123104-activity-panels.md`) got closer — its Map model says *"session hub nodes for tool events (link tool invocation → its session hub)"* and Task 6 says *"build nodes/links from `payload.map`"* — but that is a different document, and neither spelled out the shapes. This section closes it.

### What is actually built
**Activity map** — three link shapes, assembled in `_build_map()` (`viz/activity.py:349-381`):
| Shape | Meaning |
|---|---|
| `event → session hub` | a tool call belongs to the session that made it |
| `session hub → pod:<stream>` | the session belongs to its stream |
| `event → pod:<stream>` | everything else attaches directly to its stream |

The two-hop `event → hub → pod` chain is the whole reason a tool call *reads* as attached to a session rather than floating in a cloud.

**Neuro map** (`viz/ui/app.js:744`):
| Shape | Meaning |
|---|---|
| `pod[i-1] → pod[i]`, `kind:"residual"` | the residual stream, layer to layer — dashed gold, `lineWidth 2` |
| weighted edges | `lineWidth = 0.8 + weight * 1.6` — attention concentration as thickness |

### Rule going forward
Every map declares its node kinds **and** its link kinds in one place, next to the code that builds them. A new stream adds its link shape to `_build_map()`, never to the paint loop. The paint loop reads `kind` and `weight` and does not invent relationships.
