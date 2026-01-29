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

Multi-kill streaks are now classified by **context**:

* **Execute** — kills on A/B (or ENTRY) during early phase, or within
  `ENTRY_HOLD_DELAY_SECONDS` after the first on-site kill in the round.
* **Hold** — later on-site kills (or any late streak when no entry window is detected).

This is a heuristic driven by round time + A/B zone detection (from `place_map`/bbox).
If plant events are available in the future, the same interface supports stronger signals.

For each context we store counts for `k2`, `k3`, `k4`, `k5` (2k–5k) and the same
breakdown per side (T/CT) when available.

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
