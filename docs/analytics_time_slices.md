# Analytics time buckets & entry context

## Time buckets (heatmaps)

Heatmaps are now time-sliced by **round_time_s**. A request can specify:

* `time_bucket` (preset buckets)
* `time_from` / `time_to` (custom bounds, seconds from round start)
* `slice` (legacy label, e.g. `0-15`)

Default behavior remains **all-time** when no time filters are provided.

### Preset buckets

Default presets (configurable via `HEATMAP_TIME_BUCKETS` in settings):

* `early`: 0–15s
* `mid`: 16–35s
* `late`: 36s–end

### Custom bounds

When `time_from` / `time_to` are passed, the API builds a normalized slice label
(`0-15`, `16-35`, `36+`) and caches heatmaps independently for that time range.

## Execute vs Hold (multi-kills)

Multi-kill streaks are classified by **phase** with a professional execute/hold model:

### T-side phases

* **T: entry / map control (`t_entry`)** — kills away from the objective or before a site is identified.
* **T: execute (commit) (`t_execute`)** — early on-site kills, or within
  `ENTRY_HOLD_DELAY_SECONDS` of the first on-site kill in the round.
* **T: site hold (pre-plant) (`t_hold`)** — later on-site kills before the plant.
* **T: post-plant (`t_post_plant`)** — all post-plant kills.

### CT-side phases

* **CT: hold (pre-plant) (`ct_hold`)** — defending within the site hold radius.
* **CT: push (pre-plant) (`ct_push`)** — defensive pushes away from the hold radius.
* **CT: roam / rotation (`ct_roam`)** — no reliable objective center (no plant + no inferred site).
* **CT: retake (post-plant) (`ct_retake`)** — all post-plant kills.

Objective site comes from bomb plant coordinates when available. Otherwise, we infer it
from the player's earliest on-site kill using zones (`place_map` + `site_places`).

For each phase we store counts for `k2`, `k3`, `k4`, `k5` (2k–5k), and the same
breakdown per side when available.

## Assisted vs Solo entry (playstyle)

Entry attempts are detected from **first duels** in a round. Each entry is labeled:

* **Assisted** — an assist/flash-assist is present, or a teammate is nearby within
  `ENTRY_SUPPORT_RADIUS` and `ENTRY_SUPPORT_WINDOW_SECONDS`.
* **Solo** — no supporting signal and a valid proximity check confirms isolation.

The analytics output includes:

* assisted/solo percentages
* early/mid/late distribution for each
* average entry time in seconds

Unknown/insufficient proximity data will mark the breakdown as `approx`.
