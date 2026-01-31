from __future__ import annotations

"""
CS2 Heatmaps — analyzer_v8 (Alignment fix)

You reported: points/paths are systematically shifted (down+right) and appear in impossible places.
Root cause in 99% cases:
- Your radar PNG (clean radar) DOES NOT match the awpy map-data.json calibration (pos_x/pos_y/scale),
  because the PNG has different padding/crop/zoom than awpy's default radar.
So even if the demo coords are correct, the projection to pixels is offset.

What this version adds:
1) Explicit pixel offsets you can tune once per map:
      OFFSET_X_PX / OFFSET_Y_PX
2) Optional auto-offset search (dx,dy) that chooses a shift that best fits the map OUTLINE mask.
   This fixes typical padding/crop differences (systematic shift), without you guessing numbers.
3) Heatmap alpha is clipped to the map mask (like v7), so nothing renders outside the map outline.

If after auto-offset you STILL see "inside walls", then your radar is not only shifted, but also scaled.
In that case: use the awpy default radar OR adjust map-data.json scale/pos for your radar.
"""

from pathlib import Path
import hashlib
import json
import logging
import math
import os

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter
from scipy.ndimage import gaussian_filter, label, binary_closing, binary_opening
from matplotlib import cm
from awpy import Demo

ANALYZER_VERSION = "radar-smooth-v8-aligned"
logger = logging.getLogger(__name__)


# ----------------------------
# USER TUNING (most important)
# ----------------------------

# If your overlay is shifted down+right, usually you need negative offsets here.
# Start with (0,0). If you want manual tuning: try (-12, -10) etc.
OFFSET_X_PX = -18
OFFSET_Y_PX = -15

# Auto-offset will estimate a (dx,dy) shift using the map outline mask.
AUTO_OFFSET = True
AUTO_OFFSET_MAX = 60      # search range in pixels: [-60..60]
AUTO_OFFSET_STEP = 2      # coarse step (2 is fast)
AUTO_OFFSET_REFINE = True # refine around best with step=1
AUTO_OFFSET_SAMPLE = 6000 # number of points sampled for scoring (speed)

# Trails vs Zones
TIME_STEP_TICKS = 8          # trails: 16/32; zones: 64/96/128
PRESENCE_SIGMA_PX = 2.2       # trails: 1.4..2.6; zones: 5..12
KD_SIGMA_PX = 2.0

# Hotspot control
PCTL_CLIP = 98.8
GAMMA = 0.5

# Overlay alpha behavior
MAX_ALPHA = 245
ALPHA_POWER = 0.85
ALPHA_CUTOFF = 0.035

# Clip settings (map mask)
MASK_ALPHA_THRESHOLD = 12
MASK_BRIGHT_THRESHOLD = 28
MASK_EDGE_BLUR = 1.6

# Colormaps
CMAP_ALL = "inferno"
CMAP_CT = "Blues"
CMAP_T  = "Reds"
CMAP_KILLS = "magma"
CMAP_DEATHS = "plasma"

# Optional radar enhancement
RADAR_BRIGHTNESS = 1.00
RADAR_CONTRAST   = 1.00
RADAR_COLOR      = 1.00


# ----------------------------
# IO / map data helpers
# ----------------------------
def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _awpy_maps_dir() -> Path:
    return Path(os.path.expanduser("~")) / ".awpy" / "maps"


def _load_map_data(map_name: str) -> dict:
    p = _awpy_maps_dir() / "map-data.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}. Put map-data.json into ~/.awpy/maps/")
    data = json.loads(p.read_text(encoding="utf-8"))
    if map_name not in data:
        raise KeyError(f"Map '{map_name}' not found in map-data.json.")
    return data[map_name]


def _pick_radar_image(map_name: str) -> Path:
    maps_dir = _awpy_maps_dir()
    candidates = [
        maps_dir / f"{map_name}_clean.png",
        maps_dir / f"{map_name}.png",
        maps_dir / f"{map_name}_default.png",
        maps_dir / f"{map_name.replace('de_','')}_clean.png",
        maps_dir / f"{map_name.replace('de_','')}.png",
        maps_dir / f"{map_name.replace('de_','')}_default.png",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"No radar image found in {maps_dir}. Tried: {[c.name for c in candidates]}")


def _prep_radar(map_name: str) -> tuple[Image.Image, dict, str]:
    radar_path = _pick_radar_image(map_name)
    radar = Image.open(radar_path).convert("RGBA")

    if RADAR_BRIGHTNESS != 1.0:
        radar = ImageEnhance.Brightness(radar).enhance(RADAR_BRIGHTNESS)
    if RADAR_CONTRAST != 1.0:
        radar = ImageEnhance.Contrast(radar).enhance(RADAR_CONTRAST)
    if RADAR_COLOR != 1.0:
        radar = ImageEnhance.Color(radar).enhance(RADAR_COLOR)

    meta = _load_map_data(map_name)
    return radar, meta, radar_path.name


def load_radar_and_meta(map_name: str) -> tuple[Image.Image, dict, str]:
    return _prep_radar(map_name)


# ----------------------------
# Mask building (clip to map outline)
# ----------------------------
def _largest_component(mask: np.ndarray) -> np.ndarray:
    lbl, n = label(mask)
    if n <= 1:
        return mask
    sizes = np.bincount(lbl.ravel())
    sizes[0] = 0
    keep = sizes.argmax()
    return lbl == keep


def _build_map_mask(radar: Image.Image) -> Image.Image:
    arr = np.array(radar)
    a = arr[:, :, 3].astype(np.uint8)

    if a.min() < 250:
        m = a > MASK_ALPHA_THRESHOLD
        m = _largest_component(m)
    else:
        rgb = arr[:, :, :3].astype(np.int16)
        bright = rgb.mean(axis=2)
        m = bright > MASK_BRIGHT_THRESHOLD
        m = _largest_component(m)

    m = binary_closing(m, structure=np.ones((5, 5), dtype=bool), iterations=1)
    m = binary_opening(m, structure=np.ones((3, 3), dtype=bool), iterations=1)

    mask_img = Image.fromarray((m.astype(np.uint8) * 255), mode="L")
    if MASK_EDGE_BLUR > 0:
        mask_img = mask_img.filter(ImageFilter.GaussianBlur(MASK_EDGE_BLUR))
    return mask_img


def build_map_mask(radar: Image.Image) -> Image.Image:
    return _build_map_mask(radar)


# ----------------------------
# Data helpers
# ----------------------------
def _pick_existing(df: pd.DataFrame, candidates: list[str]) -> str:
    low = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in low:
            return low[c.lower()]
    raise KeyError(f"None of {candidates} found. Have: {list(df.columns)[:80]}...")


def _filter_by_steamid_numeric(df: pd.DataFrame, col: str, steamid64: str) -> pd.DataFrame:
    target = int(str(steamid64))
    s = pd.to_numeric(df[col], errors="coerce")
    return df[s.eq(target)]


def _to_points_xy(df: pd.DataFrame, xcol: str, ycol: str) -> np.ndarray:
    xs = pd.to_numeric(df[xcol], errors="coerce").to_numpy()
    ys = pd.to_numeric(df[ycol], errors="coerce").to_numpy()
    m = np.isfinite(xs) & np.isfinite(ys)
    return np.stack([xs[m], ys[m]], axis=1).astype(np.float32)


def _world_to_pixel(points_xy: np.ndarray, map_meta: dict, radar_size: tuple[int, int]) -> np.ndarray:
    w, h = radar_size
    pos_x = float(map_meta["pos_x"])
    pos_y = float(map_meta["pos_y"])
    scale = float(map_meta["scale"])

    xs = points_xy[:, 0]
    ys = points_xy[:, 1]
    px = (xs - pos_x) / scale
    py = (pos_y - ys) / scale

    # apply user offsets
    px = px + float(OFFSET_X_PX)
    py = py + float(OFFSET_Y_PX)

    m = (px >= 0) & (px < w) & (py >= 0) & (py < h)
    return np.stack([px[m], py[m]], axis=1).astype(np.float32)


def world_to_pixel(points_xy: np.ndarray, map_meta: dict, radar_size: tuple[int, int]) -> np.ndarray:
    return _world_to_pixel(points_xy, map_meta, radar_size)


def _world_to_pixel_with_time(
    points_xyt: np.ndarray,
    map_meta: dict,
    radar_size: tuple[int, int],
) -> np.ndarray:
    if points_xyt.size == 0:
        return _empty_points_time()
    w, h = radar_size
    pos_x = float(map_meta["pos_x"])
    pos_y = float(map_meta["pos_y"])
    scale = float(map_meta["scale"])

    xs = points_xyt[:, 0]
    ys = points_xyt[:, 1]
    ts = points_xyt[:, 2]
    px = (xs - pos_x) / scale
    py = (pos_y - ys) / scale

    px = px + float(OFFSET_X_PX)
    py = py + float(OFFSET_Y_PX)

    m = (px >= 0) & (px < w) & (py >= 0) & (py < h)
    if not m.any():
        return _empty_points_time()
    return np.stack([px[m], py[m], ts[m]], axis=1).astype(np.float32)


def _downsample_by_tick(df: pd.DataFrame, tick_col: str) -> pd.DataFrame:
    if TIME_STEP_TICKS <= 1:
        return df
    t = pd.to_numeric(df[tick_col], errors="coerce")
    bucket = (t // TIME_STEP_TICKS).astype("Int64")
    return df.loc[~bucket.duplicated(keep="first")]


def _normalize_side_value(v) -> str | None:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    if isinstance(v, (int, np.integer)):
        return "CT" if int(v) == 3 else ("T" if int(v) == 2 else None)
    s = str(v).strip().lower()
    if s in ("ct", "counterterrorist", "counter-terrorist", "counter_terrorist", "3"):
        return "CT"
    if s in ("t", "terrorist", "2"):
        return "T"
    return None


# ----------------------------
# Auto-offset (outline mask fitting)
# ----------------------------
def _auto_offset_shift(pts: np.ndarray, mask_L: Image.Image) -> tuple[int, int, float]:
    """
    Find (dx,dy) that maximizes average mask value at shifted point locations.
    This corrects systematic pixel shift caused by radar padding/crop mismatch.
    """
    if pts.size == 0:
        return 0, 0, 0.0

    mask = np.asarray(mask_L).astype(np.float32) / 255.0
    h, w = mask.shape

    # sample points for speed
    if pts.shape[0] > AUTO_OFFSET_SAMPLE:
        idx = np.random.RandomState(7).choice(pts.shape[0], AUTO_OFFSET_SAMPLE, replace=False)
        p = pts[idx]
    else:
        p = pts

    x = np.rint(p[:, 0]).astype(np.int32)
    y = np.rint(p[:, 1]).astype(np.int32)

    # clamp base
    x = np.clip(x, 0, w - 1)
    y = np.clip(y, 0, h - 1)

    def score(dx: int, dy: int) -> float:
        xx = np.clip(x + dx, 0, w - 1)
        yy = np.clip(y + dy, 0, h - 1)
        return float(mask[yy, xx].mean())

    best_dx = 0
    best_dy = 0
    best_s = score(0, 0)

    rng = range(-AUTO_OFFSET_MAX, AUTO_OFFSET_MAX + 1, AUTO_OFFSET_STEP)
    for dy in rng:
        for dx in rng:
            s = score(dx, dy)
            if s > best_s:
                best_s = s
                best_dx, best_dy = dx, dy

    if AUTO_OFFSET_REFINE:
        # refine around the best with step=1 in a small window
        r2 = range(best_dy - AUTO_OFFSET_STEP, best_dy + AUTO_OFFSET_STEP + 1)
        c2 = range(best_dx - AUTO_OFFSET_STEP, best_dx + AUTO_OFFSET_STEP + 1)
        for dy in r2:
            for dx in c2:
                s = score(dx, dy)
                if s > best_s:
                    best_s = s
                    best_dx, best_dy = dx, dy

    return best_dx, best_dy, best_s


def _apply_shift(pts: np.ndarray, dx: int, dy: int, radar_size: tuple[int, int]) -> np.ndarray:
    if pts.size == 0:
        return pts
    w, h = radar_size
    p = pts.copy()
    p[:, 0] = np.clip(p[:, 0] + dx, 0, w - 1)
    p[:, 1] = np.clip(p[:, 1] + dy, 0, h - 1)
    return p


# ----------------------------
# Heat generation (pixel exact) + CLIP
# ----------------------------
def _density_to_heat_rgba_pixel(
    pts_px: np.ndarray,
    radar_size: tuple[int, int],
    *,
    cmap_name: str,
    sigma_px: float,
    map_mask_L: Image.Image,
) -> Image.Image:
    w, h = radar_size
    if pts_px.size == 0:
        return Image.new("RGBA", (w, h), (0, 0, 0, 0))

    dens = np.zeros((h, w), dtype=np.float32)
    ix = np.rint(pts_px[:, 0]).astype(np.int32)
    iy = np.rint(pts_px[:, 1]).astype(np.int32)
    ix = np.clip(ix, 0, w - 1)
    iy = np.clip(iy, 0, h - 1)
    np.add.at(dens, (iy, ix), 1.0)

    dens = gaussian_filter(dens, sigma=sigma_px)

    dens = np.log1p(dens)
    vmax = np.percentile(dens, PCTL_CLIP) if dens.max() > 0 else 1.0
    vmax = float(vmax) if vmax and vmax > 0 else 1.0
    dens = np.clip(dens, 0, vmax) / vmax

    dens = np.clip(dens, 0, 1) ** GAMMA

    a = dens.copy()
    a[a < ALPHA_CUTOFF] = 0.0
    a = (a ** ALPHA_POWER) * (MAX_ALPHA / 255.0)

    # CLIP alpha to map outline
    mask = (np.asarray(map_mask_L).astype(np.float32) / 255.0)
    a = a * mask

    cmap = cm.get_cmap(cmap_name)
    rgba = cmap(dens)
    rgb = (rgba[:, :, :3] * 255).astype(np.uint8)

    out = np.zeros((h, w, 4), dtype=np.uint8)
    out[:, :, :3] = rgb
    out[:, :, 3] = (a * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(out, mode="RGBA")


def density_to_heat_rgba_pixel(
    pts_px: np.ndarray,
    radar_size: tuple[int, int],
    *,
    cmap_name: str,
    sigma_px: float,
    map_mask_L: Image.Image,
) -> Image.Image:
    return _density_to_heat_rgba_pixel(
        pts_px, radar_size, cmap_name=cmap_name, sigma_px=sigma_px, map_mask_L=map_mask_L
    )


SMALL_PREVIEW_SIZE = (512, 512)


def _small_preview_path(out_path: Path) -> Path:
    return out_path.with_name(out_path.stem + "_512.png")


def _save_composited(radar: Image.Image, heat: Image.Image, out_path: Path) -> None:
    composited = Image.alpha_composite(radar, heat)
    _ensure_dir(out_path.parent)
    composited.save(out_path, optimize=True, compress_level=9)

    small = composited.resize(SMALL_PREVIEW_SIZE, Image.Resampling.LANCZOS)
    small_path = _small_preview_path(out_path)
    small.save(small_path, optimize=True, compress_level=9)


def _make_placeholder_png(out_path: Path, size: tuple[int, int]) -> None:
    _ensure_dir(out_path.parent)
    placeholder = Image.new("RGBA", size, (0, 0, 0, 0))
    placeholder.save(out_path, optimize=True, compress_level=9)

    small = placeholder.resize(SMALL_PREVIEW_SIZE, Image.Resampling.LANCZOS)
    small_path = _small_preview_path(out_path)
    small.save(small_path, optimize=True, compress_level=9)


def _demo_cache_hash(dem_path: Path, radar_name: str, radar_size: tuple[int, int]) -> str:
    h = hashlib.sha1()
    h.update(dem_path.read_bytes())
    h.update(ANALYZER_VERSION.encode("utf-8"))
    h.update(radar_name.encode("utf-8"))
    h.update(f"{radar_size[0]}x{radar_size[1]}".encode("utf-8"))
    return h.hexdigest()


def _empty_points() -> np.ndarray:
    return np.empty((0, 2), dtype=np.float32)


def _empty_points_time() -> np.ndarray:
    return np.empty((0, 3), dtype=np.float32)


def _extract_points_from_demo(
    dem_path: Path,
    steamid64: str,
    map_meta: dict,
    radar_size: tuple[int, int],
    map_mask_L: Image.Image,
    dem: Demo | None = None,
) -> tuple[dict, dict]:
    if dem is None:
        dem = Demo(str(dem_path), verbose=False)
        dem.parse()

    from faceit_analytics.services import demo_events

    ticks_df = demo_events._load_demo_dataframe(
        getattr(dem, "ticks", None),
        ["steamid", "player_steamid", "playerSteamID"],
    )
    ticks_df = ticks_df if ticks_df is not None else pd.DataFrame()
    rounds_df = demo_events._load_demo_dataframe(getattr(dem, "rounds", None), [])
    round_start_ticks, round_start_times, _round_winners, _rounds = demo_events._build_round_meta(
        rounds_df,
        ticks_df=ticks_df,
    )
    tick_rate = demo_events._tick_rate_from_demo(dem)
    sid_col = _pick_existing(ticks_df, ["steamid", "steamID", "player_steamid", "playerSteamID"]) if not ticks_df.empty else None
    xcol = _pick_existing(ticks_df, ["X", "x", "player_X", "player_x"]) if not ticks_df.empty else None
    ycol = _pick_existing(ticks_df, ["Y", "y", "player_Y", "player_y"]) if not ticks_df.empty else None
    tick_col = _pick_existing(ticks_df, demo_events.TICK_COL_CANDIDATES) if not ticks_df.empty else None
    round_col = demo_events._pick_column(ticks_df, demo_events.ROUND_COL_CANDIDATES) if not ticks_df.empty else None

    ticks_my = _filter_by_steamid_numeric(ticks_df, sid_col, steamid64) if sid_col else ticks_df.iloc[0:0]
    low = {c.lower(): c for c in ticks_my.columns}
    if "health" in low:
        hc = low["health"]
        alive = ticks_my[pd.to_numeric(ticks_my[hc], errors="coerce").fillna(0) > 0]
        if len(alive) > 0:
            ticks_my = alive

    if tick_col:
        ticks_my = _downsample_by_tick(ticks_my, tick_col)
    ticks_target_total = int(ticks_my.shape[0])
    t_round_values = []
    valid_rows = []
    missing_t_round = 0
    for _, row in ticks_my.iterrows():
        round_number = demo_events._safe_int(row.get(round_col)) if round_col else None
        t_round = demo_events._round_time_seconds(row, round_number, round_start_ticks, round_start_times, tick_rate)
        if t_round is None:
            missing_t_round += 1
            continue
        t_round_values.append(float(t_round))
        valid_rows.append(row)
    if valid_rows:
        ticks_my = pd.DataFrame(valid_rows)
        ticks_my["__t_round"] = t_round_values
    else:
        ticks_my = ticks_my.iloc[0:0]
        ticks_my["__t_round"] = []
    if ticks_target_total and ticks_target_total > 0:
        missing_ratio = missing_t_round / max(ticks_target_total, 1)
        if missing_ratio >= 0.9:
            logger.warning(
                "Round time missing for %s/%s tick rows (%.1f%%). Check round start meta.",
                missing_t_round,
                ticks_target_total,
                missing_ratio * 100,
            )

    pts_px = _world_to_pixel(_to_points_xy(ticks_my, xcol, ycol), map_meta, radar_size) if xcol and ycol else _empty_points()
    pts_pxt = _empty_points_time()
    if not ticks_my.empty and xcol and ycol and "__t_round" in ticks_my:
        pts_pxt = _world_to_pixel_with_time(
            np.column_stack(
                [
                    pd.to_numeric(ticks_my[xcol], errors="coerce").to_numpy(),
                    pd.to_numeric(ticks_my[ycol], errors="coerce").to_numpy(),
                    pd.to_numeric(ticks_my["__t_round"], errors="coerce").to_numpy(),
                ]
            ).astype(np.float32, copy=False),
            map_meta,
            radar_size,
        )

    auto_dx = 0
    auto_dy = 0
    auto_score = 0.0
    if AUTO_OFFSET and pts_px.shape[0] > 200:
        auto_dx, auto_dy, auto_score = _auto_offset_shift(pts_px, map_mask_L)
        pts_px = _apply_shift(pts_px, auto_dx, auto_dy, radar_size)
        pts_pxt[:, :2] = _apply_shift(pts_pxt[:, :2], auto_dx, auto_dy, radar_size)

    ct_pts = _empty_points()
    t_pts = _empty_points()
    ct_pxt = _empty_points_time()
    t_pxt = _empty_points_time()
    if "side" in low:
        side_col = low["side"]
        side_norm = ticks_my[side_col].map(_normalize_side_value)

        ct_df = ticks_my[side_norm == "CT"]
        t_df = ticks_my[side_norm == "T"]

        ct_pts = _world_to_pixel(_to_points_xy(ct_df, xcol, ycol), map_meta, radar_size)
        t_pts = _world_to_pixel(_to_points_xy(t_df, xcol, ycol), map_meta, radar_size)
        ct_pxt = _world_to_pixel_with_time(
            np.column_stack(
                [
                    pd.to_numeric(ct_df[xcol], errors="coerce").to_numpy(),
                    pd.to_numeric(ct_df[ycol], errors="coerce").to_numpy(),
                    pd.to_numeric(ct_df["__t_round"], errors="coerce").to_numpy(),
                ]
            ).astype(np.float32, copy=False),
            map_meta,
            radar_size,
        )
        t_pxt = _world_to_pixel_with_time(
            np.column_stack(
                [
                    pd.to_numeric(t_df[xcol], errors="coerce").to_numpy(),
                    pd.to_numeric(t_df[ycol], errors="coerce").to_numpy(),
                    pd.to_numeric(t_df["__t_round"], errors="coerce").to_numpy(),
                ]
            ).astype(np.float32, copy=False),
            map_meta,
            radar_size,
        )

        if AUTO_OFFSET and (auto_dx or auto_dy):
            ct_pts = _apply_shift(ct_pts, auto_dx, auto_dy, radar_size)
            t_pts = _apply_shift(t_pts, auto_dx, auto_dy, radar_size)
            ct_pxt[:, :2] = _apply_shift(ct_pxt[:, :2], auto_dx, auto_dy, radar_size)
            t_pxt[:, :2] = _apply_shift(t_pxt[:, :2], auto_dx, auto_dy, radar_size)

    kills_df = demo_events._load_demo_dataframe(
        getattr(dem, "kills", None),
        ["attacker_steamid", "victim_steamid", "assister_steamid", "killer_steamid"],
    )
    kills_df = kills_df if kills_df is not None else pd.DataFrame()
    attacker_col = (
        _pick_existing(kills_df, ["attacker_steamid", "killer_steamid", "attackerSteamID", "killerSteamID"])
        if not kills_df.empty
        else None
    )
    attacker_side_col = (
        _pick_existing(kills_df, ["attacker_side", "attacker_team", "attackerSide", "attackerTeam"])
        if not kills_df.empty
        else None
    )
    victim_side_col = (
        _pick_existing(kills_df, ["victim_side", "victim_team", "victimSide", "victimTeam"])
        if not kills_df.empty
        else None
    )
    kx = _pick_existing(kills_df, ["attacker_X", "attacker_x"]) if not kills_df.empty else None
    ky = _pick_existing(kills_df, ["attacker_Y", "attacker_y"]) if not kills_df.empty else None
    round_kill_col = (
        demo_events._pick_column(kills_df, demo_events.ROUND_COL_CANDIDATES) if not kills_df.empty else None
    )
    def _build_points_with_time(
        df: pd.DataFrame,
        x_col: str | None,
        y_col: str | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        pts = _world_to_pixel(_to_points_xy(df, x_col, y_col), map_meta, radar_size) if x_col and y_col else _empty_points()
        pts_time = _empty_points_time()
        if df.empty or not x_col or not y_col:
            return pts, pts_time
        points_rows = []
        points_t_round = []
        for _, row in df.iterrows():
            round_number = demo_events._safe_int(row.get(round_kill_col)) if round_kill_col else None
            t_round = demo_events._round_time_seconds(row, round_number, round_start_ticks, round_start_times, tick_rate)
            if t_round is None:
                continue
            points_rows.append(row)
            points_t_round.append(float(t_round))
        if points_rows:
            points_df = pd.DataFrame(points_rows)
            points_df["__t_round"] = points_t_round
            pts_time = _world_to_pixel_with_time(
                np.column_stack(
                    [
                        pd.to_numeric(points_df[x_col], errors="coerce").to_numpy(),
                        pd.to_numeric(points_df[y_col], errors="coerce").to_numpy(),
                        pd.to_numeric(points_df["__t_round"], errors="coerce").to_numpy(),
                    ]
                ).astype(np.float32, copy=False),
                map_meta,
                radar_size,
            )
        return pts, pts_time

    kills_my = _filter_by_steamid_numeric(kills_df, attacker_col, steamid64) if attacker_col else kills_df.iloc[0:0]
    kill_pts, kills_pxt = _build_points_with_time(kills_my, kx, ky)
    kills_ct_pts = _empty_points()
    kills_t_pts = _empty_points()
    kills_ct_pxt = _empty_points_time()
    kills_t_pxt = _empty_points_time()
    if attacker_side_col and not kills_my.empty:
        side_norm = kills_my[attacker_side_col].map(_normalize_side_value)
        kills_ct_pts, kills_ct_pxt = _build_points_with_time(kills_my[side_norm == "CT"], kx, ky)
        kills_t_pts, kills_t_pxt = _build_points_with_time(kills_my[side_norm == "T"], kx, ky)
    if AUTO_OFFSET and (auto_dx or auto_dy):
        kill_pts = _apply_shift(kill_pts, auto_dx, auto_dy, radar_size)
        kills_ct_pts = _apply_shift(kills_ct_pts, auto_dx, auto_dy, radar_size)
        kills_t_pts = _apply_shift(kills_t_pts, auto_dx, auto_dy, radar_size)
        if kills_pxt.size:
            kills_pxt[:, :2] = _apply_shift(kills_pxt[:, :2], auto_dx, auto_dy, radar_size)
        if kills_ct_pxt.size:
            kills_ct_pxt[:, :2] = _apply_shift(kills_ct_pxt[:, :2], auto_dx, auto_dy, radar_size)
        if kills_t_pxt.size:
            kills_t_pxt[:, :2] = _apply_shift(kills_t_pxt[:, :2], auto_dx, auto_dy, radar_size)

    victim_col = _pick_existing(kills_df, ["victim_steamid", "victimSteamID"]) if not kills_df.empty else None
    dx = _pick_existing(kills_df, ["victim_X", "victim_x"]) if not kills_df.empty else None
    dy = _pick_existing(kills_df, ["victim_Y", "victim_y"]) if not kills_df.empty else None
    deaths_my = _filter_by_steamid_numeric(kills_df, victim_col, steamid64) if victim_col else kills_df.iloc[0:0]
    death_pts, deaths_pxt = _build_points_with_time(deaths_my, dx, dy)
    deaths_ct_pts = _empty_points()
    deaths_t_pts = _empty_points()
    deaths_ct_pxt = _empty_points_time()
    deaths_t_pxt = _empty_points_time()
    if victim_side_col and not deaths_my.empty:
        side_norm = deaths_my[victim_side_col].map(_normalize_side_value)
        deaths_ct_pts, deaths_ct_pxt = _build_points_with_time(deaths_my[side_norm == "CT"], dx, dy)
        deaths_t_pts, deaths_t_pxt = _build_points_with_time(deaths_my[side_norm == "T"], dx, dy)
    if AUTO_OFFSET and (auto_dx or auto_dy):
        death_pts = _apply_shift(death_pts, auto_dx, auto_dy, radar_size)
        deaths_ct_pts = _apply_shift(deaths_ct_pts, auto_dx, auto_dy, radar_size)
        deaths_t_pts = _apply_shift(deaths_t_pts, auto_dx, auto_dy, radar_size)
        if deaths_pxt.size:
            deaths_pxt[:, :2] = _apply_shift(deaths_pxt[:, :2], auto_dx, auto_dy, radar_size)
        if deaths_ct_pxt.size:
            deaths_ct_pxt[:, :2] = _apply_shift(deaths_ct_pxt[:, :2], auto_dx, auto_dy, radar_size)
        if deaths_t_pxt.size:
            deaths_t_pxt[:, :2] = _apply_shift(deaths_t_pxt[:, :2], auto_dx, auto_dy, radar_size)

    points = {
        "presence_all_px": pts_px,
        "presence_ct_px": ct_pts,
        "presence_t_px": t_pts,
        "presence_all_pxt": pts_pxt,
        "presence_ct_pxt": ct_pxt,
        "presence_t_pxt": t_pxt,
        "kills_px": kill_pts,
        "kills_ct_px": kills_ct_pts,
        "kills_t_px": kills_t_pts,
        "deaths_px": death_pts,
        "deaths_ct_px": deaths_ct_pts,
        "deaths_t_px": deaths_t_pts,
        "kills_pxt": kills_pxt,
        "kills_ct_pxt": kills_ct_pxt,
        "kills_t_pxt": kills_t_pxt,
        "deaths_pxt": deaths_pxt,
        "deaths_ct_pxt": deaths_ct_pxt,
        "deaths_t_pxt": deaths_t_pxt,
    }
    debug = {
        "auto_offset_px": [int(auto_dx), int(auto_dy)],
        "auto_offset_score": float(auto_score),
        "ticks_target_rows": int(ticks_my.shape[0]),
        "ticks_target_total": ticks_target_total,
        "ticks_missing_t_round": int(missing_t_round),
    }
    return points, debug


def _limit_points(points: np.ndarray, max_points: int, seed: int = 7) -> np.ndarray:
    if points.shape[0] <= max_points:
        return points
    rng = np.random.RandomState(seed)
    idx = rng.choice(points.shape[0], max_points, replace=False)
    return points[idx]


# ----------------------------
# Public API
# ----------------------------
def build_heatmaps(dem_path: Path, out_dir: Path, steamid64: str) -> dict:
    out_dir = Path(out_dir)
    _ensure_dir(out_dir)

    dem = Demo(str(dem_path), verbose=False)
    dem.parse()
    map_name = dem.header.get("map_name", "unknown")

    radar, meta, radar_name = load_radar_and_meta(map_name)
    w, h = radar.size
    map_mask_L = build_map_mask(radar)

    points, debug = _extract_points_from_demo(dem_path, steamid64, meta, (w, h), map_mask_L, dem=dem)
    pts_px = points["presence_all_px"]
    ct_pts = points["presence_ct_px"]
    t_pts = points["presence_t_px"]
    kill_pts = points["kills_px"]
    death_pts = points["deaths_px"]
    auto_dx, auto_dy = debug["auto_offset_px"]
    auto_score = debug["auto_offset_score"]

    heat_all = _density_to_heat_rgba_pixel(
        pts_px, (w, h), cmap_name=CMAP_ALL, sigma_px=PRESENCE_SIGMA_PX, map_mask_L=map_mask_L
    )
    presence_png = out_dir / "presence_heatmap.png"
    _save_composited(radar, heat_all, presence_png)

    # CT/T split
    presence_ct_png = out_dir / "presence_heatmap_ct.png"
    presence_t_png = out_dir / "presence_heatmap_t.png"
    ct_count = int(ct_pts.shape[0])
    t_count = int(t_pts.shape[0])

    if ct_count:
        heat_ct = _density_to_heat_rgba_pixel(
            ct_pts, (w, h), cmap_name=CMAP_CT, sigma_px=PRESENCE_SIGMA_PX, map_mask_L=map_mask_L
        )
        _save_composited(radar, heat_ct, presence_ct_png)
    else:
        _make_placeholder_png(presence_ct_png, (w, h))

    if t_count:
        heat_t = _density_to_heat_rgba_pixel(
            t_pts, (w, h), cmap_name=CMAP_T, sigma_px=PRESENCE_SIGMA_PX, map_mask_L=map_mask_L
        )
        _save_composited(radar, heat_t, presence_t_png)
    else:
        _make_placeholder_png(presence_t_png, (w, h))

    # ---- KILLS / DEATHS ----
    kills_png = out_dir / "kills_heatmap.png"
    if kill_pts.shape[0]:
        heat_k = _density_to_heat_rgba_pixel(
            kill_pts, (w, h), cmap_name=CMAP_KILLS, sigma_px=KD_SIGMA_PX, map_mask_L=map_mask_L
        )
        _save_composited(radar, heat_k, kills_png)
    else:
        _make_placeholder_png(kills_png, (w, h))

    deaths_png = out_dir / "deaths_heatmap.png"
    if death_pts.shape[0]:
        heat_d = _density_to_heat_rgba_pixel(
            death_pts, (w, h), cmap_name=CMAP_DEATHS, sigma_px=KD_SIGMA_PX, map_mask_L=map_mask_L
        )
        _save_composited(radar, heat_d, deaths_png)
    else:
        _make_placeholder_png(deaths_png, (w, h))

    return {
        "steamid64": str(steamid64),
        "map": map_name,
        "counts": {
            "kills": int(kill_pts.shape[0]),
            "deaths": int(death_pts.shape[0]),
            "presence_points": int(pts_px.shape[0]),
            "presence_ct_points": int(ct_count),
            "presence_t_points": int(t_count),
        },
        "analyzer_version": ANALYZER_VERSION,
        "files": {
            "presence": _small_preview_path(presence_png).name,
            "presence_ct": _small_preview_path(presence_ct_png).name,
            "presence_t": _small_preview_path(presence_t_png).name,
            "kills": _small_preview_path(kills_png).name,
            "deaths": _small_preview_path(deaths_png).name,
        },
        "debug": {
            "radar_used": radar_name,
            "manual_offset_px": [int(OFFSET_X_PX), int(OFFSET_Y_PX)],
            "auto_offset_enabled": bool(AUTO_OFFSET),
            "auto_offset_px": [int(auto_dx), int(auto_dy)],
            "auto_offset_score": float(auto_score),
            "time_step_ticks": int(TIME_STEP_TICKS),
            "presence_sigma_px": float(PRESENCE_SIGMA_PX),
        },
    }


def build_heatmaps_aggregate(
    steamid64: str,
    map_name: str,
    limit: int,
    demos_dir: Path,
    out_dir: Path,
    cache_dir: Path,
    *,
    force: bool = False,
) -> dict:
    demos_dir = Path(demos_dir)
    out_dir = Path(out_dir)
    cache_dir = Path(cache_dir)

    demo_paths = sorted(demos_dir.glob("*.dem"), key=lambda p: p.stat().st_mtime, reverse=True)
    demo_paths = demo_paths[: max(int(limit), 1)]
    if not demo_paths:
        raise FileNotFoundError(f"No .dem files found in {demos_dir}")

    radar, meta, radar_name = load_radar_and_meta(map_name)
    w, h = radar.size
    map_mask_L = build_map_mask(radar)

    out_files = [
        out_dir / "presence_heatmap.png",
        out_dir / "presence_heatmap_ct.png",
        out_dir / "presence_heatmap_t.png",
        out_dir / "kills_heatmap.png",
        out_dir / "deaths_heatmap.png",
    ]
    out_smalls = [_small_preview_path(p) for p in out_files]
    expected_outputs = out_files + out_smalls

    latest_demo_mtime = max(p.stat().st_mtime for p in demo_paths)
    meta_path = out_dir / "aggregate_meta.json"
    if not force and all(p.exists() for p in expected_outputs) and meta_path.exists():
        try:
            meta_payload = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            meta_payload = {}
        meta_matches = (
            meta_payload.get("limit") == int(limit)
            and meta_payload.get("demos") == [p.name for p in demo_paths]
            and meta_payload.get("map") == map_name
        )
        if meta_matches and min(p.stat().st_mtime for p in expected_outputs) >= latest_demo_mtime:
            return {
                "steamid64": str(steamid64),
                "map": map_name,
                "files": {
                    "presence": _small_preview_path(out_files[0]).name,
                    "presence_ct": _small_preview_path(out_files[1]).name,
                    "presence_t": _small_preview_path(out_files[2]).name,
                    "kills": _small_preview_path(out_files[3]).name,
                    "deaths": _small_preview_path(out_files[4]).name,
                },
                "cached": True,
                "analyzer_version": ANALYZER_VERSION,
            }

    presence_all = []
    presence_ct = []
    presence_t = []
    kills = []
    deaths = []
    cache_hits = 0

    cache_root = cache_dir / str(steamid64) / map_name
    _ensure_dir(cache_root)

    required_px_keys = {
        "presence_all_px": _empty_points(),
        "presence_ct_px": _empty_points(),
        "presence_t_px": _empty_points(),
        "kills_px": _empty_points(),
        "kills_ct_px": _empty_points(),
        "kills_t_px": _empty_points(),
        "deaths_px": _empty_points(),
        "deaths_ct_px": _empty_points(),
        "deaths_t_px": _empty_points(),
    }
    required_pxt_keys = {
        "presence_all_pxt": _empty_points_time(),
        "presence_ct_pxt": _empty_points_time(),
        "presence_t_pxt": _empty_points_time(),
        "kills_pxt": _empty_points_time(),
        "kills_ct_pxt": _empty_points_time(),
        "kills_t_pxt": _empty_points_time(),
        "deaths_pxt": _empty_points_time(),
        "deaths_ct_pxt": _empty_points_time(),
        "deaths_t_pxt": _empty_points_time(),
    }

    for dem_path in demo_paths:
        demo_hash = _demo_cache_hash(dem_path, radar_name, (w, h))
        cache_path = cache_root / f"{demo_hash}.npz"
        cache_needs_rebuild = force or not cache_path.exists()
        demo_points = None
        if cache_path.exists() and not force:
            with np.load(cache_path) as cached:
                required_time_keys = set(required_pxt_keys.keys())
                if not required_time_keys.issubset(set(cached.files)):
                    cache_needs_rebuild = True
                else:
                    demo_points = {
                        "presence_all_px": cached.get("presence_all_px", _empty_points()),
                        "presence_ct_px": cached.get("presence_ct_px", _empty_points()),
                        "presence_t_px": cached.get("presence_t_px", _empty_points()),
                        "presence_all_pxt": cached.get("presence_all_pxt", _empty_points_time()),
                        "presence_ct_pxt": cached.get("presence_ct_pxt", _empty_points_time()),
                        "presence_t_pxt": cached.get("presence_t_pxt", _empty_points_time()),
                        "kills_px": cached.get("kills_px", _empty_points()),
                        "kills_ct_px": cached.get("kills_ct_px", _empty_points()),
                        "kills_t_px": cached.get("kills_t_px", _empty_points()),
                        "deaths_px": cached.get("deaths_px", _empty_points()),
                        "deaths_ct_px": cached.get("deaths_ct_px", _empty_points()),
                        "deaths_t_px": cached.get("deaths_t_px", _empty_points()),
                        "kills_pxt": cached.get("kills_pxt", _empty_points_time()),
                        "kills_ct_pxt": cached.get("kills_ct_pxt", _empty_points_time()),
                        "kills_t_pxt": cached.get("kills_t_pxt", _empty_points_time()),
                        "deaths_pxt": cached.get("deaths_pxt", _empty_points_time()),
                        "deaths_ct_pxt": cached.get("deaths_ct_pxt", _empty_points_time()),
                        "deaths_t_pxt": cached.get("deaths_t_pxt", _empty_points_time()),
                    }
                    if (
                        (demo_points["kills_px"].shape[0] > 0 and demo_points["kills_pxt"].shape[0] == 0)
                        or (demo_points["deaths_px"].shape[0] > 0 and demo_points["deaths_pxt"].shape[0] == 0)
                        or (
                            demo_points["presence_all_px"].shape[0] > 0
                            and demo_points["presence_all_pxt"].shape[0] == 0
                        )
                    ):
                        cache_needs_rebuild = True
                        demo_points = None
        if demo_points is not None and not cache_needs_rebuild:
            cache_hits += 1
        if cache_needs_rebuild:
            demo_points, debug = _extract_points_from_demo(dem_path, steamid64, meta, (w, h), map_mask_L)
            for key, default_value in {**required_px_keys, **required_pxt_keys}.items():
                if key not in demo_points or demo_points[key] is None:
                    demo_points[key] = default_value
            if debug.get("ticks_target_total", 0) and demo_points["presence_all_px"].shape[0] == 0:
                logger.warning(
                    "Presence heatmap empty for %s despite %s ticks for steamid=%s",
                    dem_path.name,
                    debug.get("ticks_target_total"),
                    steamid64,
                )
            if demo_points["kills_px"].shape[0] > 0 and demo_points["kills_pxt"].shape[0] == 0:
                logger.warning(
                    "Kill heatmap time data missing for %s (kills_px=%s, kills_pxt=0) steamid=%s",
                    dem_path.name,
                    demo_points["kills_px"].shape[0],
                    steamid64,
                )
            np.savez_compressed(
                cache_path,
                presence_all_px=demo_points["presence_all_px"],
                presence_ct_px=demo_points["presence_ct_px"],
                presence_t_px=demo_points["presence_t_px"],
                presence_all_pxt=demo_points["presence_all_pxt"],
                presence_ct_pxt=demo_points["presence_ct_pxt"],
                presence_t_pxt=demo_points["presence_t_pxt"],
                kills_px=demo_points["kills_px"],
                kills_ct_px=demo_points["kills_ct_px"],
                kills_t_px=demo_points["kills_t_px"],
                deaths_px=demo_points["deaths_px"],
                deaths_ct_px=demo_points["deaths_ct_px"],
                deaths_t_px=demo_points["deaths_t_px"],
                kills_pxt=demo_points["kills_pxt"],
                kills_ct_pxt=demo_points["kills_ct_pxt"],
                kills_t_pxt=demo_points["kills_t_pxt"],
                deaths_pxt=demo_points["deaths_pxt"],
                deaths_ct_pxt=demo_points["deaths_ct_pxt"],
                deaths_t_pxt=demo_points["deaths_t_pxt"],
            )
        if demo_points is None:
            demo_points = {
                "presence_all_px": _empty_points(),
                "presence_ct_px": _empty_points(),
                "presence_t_px": _empty_points(),
                "presence_all_pxt": _empty_points_time(),
                "presence_ct_pxt": _empty_points_time(),
                "presence_t_pxt": _empty_points_time(),
                "kills_px": _empty_points(),
                "kills_ct_px": _empty_points(),
                "kills_t_px": _empty_points(),
                "deaths_px": _empty_points(),
                "deaths_ct_px": _empty_points(),
                "deaths_t_px": _empty_points(),
                "kills_pxt": _empty_points_time(),
                "kills_ct_pxt": _empty_points_time(),
                "kills_t_pxt": _empty_points_time(),
                "deaths_pxt": _empty_points_time(),
                "deaths_ct_pxt": _empty_points_time(),
                "deaths_t_pxt": _empty_points_time(),
            }

        presence_all.append(_limit_points(demo_points["presence_all_px"], 15000))
        presence_ct.append(_limit_points(demo_points["presence_ct_px"], 15000))
        presence_t.append(_limit_points(demo_points["presence_t_px"], 15000))
        kills.append(demo_points["kills_px"])
        deaths.append(demo_points["deaths_px"])

    def _concat(parts: list[np.ndarray]) -> np.ndarray:
        if not parts:
            return _empty_points()
        non_empty = [p for p in parts if p.size]
        if not non_empty:
            return _empty_points()
        return np.concatenate(non_empty, axis=0).astype(np.float32)

    presence_all_pts = _concat(presence_all)
    presence_ct_pts = _concat(presence_ct)
    presence_t_pts = _concat(presence_t)
    kills_pts = _concat(kills)
    deaths_pts = _concat(deaths)

    out_dir.mkdir(parents=True, exist_ok=True)

    heat_all = _density_to_heat_rgba_pixel(
        presence_all_pts, (w, h), cmap_name=CMAP_ALL, sigma_px=PRESENCE_SIGMA_PX, map_mask_L=map_mask_L
    )
    _save_composited(radar, heat_all, out_files[0])

    if presence_ct_pts.size:
        heat_ct = _density_to_heat_rgba_pixel(
            presence_ct_pts, (w, h), cmap_name=CMAP_CT, sigma_px=PRESENCE_SIGMA_PX, map_mask_L=map_mask_L
        )
        _save_composited(radar, heat_ct, out_files[1])
    else:
        _make_placeholder_png(out_files[1], (w, h))

    if presence_t_pts.size:
        heat_t = _density_to_heat_rgba_pixel(
            presence_t_pts, (w, h), cmap_name=CMAP_T, sigma_px=PRESENCE_SIGMA_PX, map_mask_L=map_mask_L
        )
        _save_composited(radar, heat_t, out_files[2])
    else:
        _make_placeholder_png(out_files[2], (w, h))

    if kills_pts.size:
        heat_k = _density_to_heat_rgba_pixel(
            kills_pts, (w, h), cmap_name=CMAP_KILLS, sigma_px=KD_SIGMA_PX, map_mask_L=map_mask_L
        )
        _save_composited(radar, heat_k, out_files[3])
    else:
        _make_placeholder_png(out_files[3], (w, h))

    if deaths_pts.size:
        heat_d = _density_to_heat_rgba_pixel(
            deaths_pts, (w, h), cmap_name=CMAP_DEATHS, sigma_px=KD_SIGMA_PX, map_mask_L=map_mask_L
        )
        _save_composited(radar, heat_d, out_files[4])
    else:
        _make_placeholder_png(out_files[4], (w, h))

    meta_payload = {
        "map": map_name,
        "limit": int(limit),
        "demos": [p.name for p in demo_paths],
        "analyzer_version": ANALYZER_VERSION,
    }
    meta_path.write_text(json.dumps(meta_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "steamid64": str(steamid64),
        "map": map_name,
        "files": {
            "presence": _small_preview_path(out_files[0]).name,
            "presence_ct": _small_preview_path(out_files[1]).name,
            "presence_t": _small_preview_path(out_files[2]).name,
            "kills": _small_preview_path(out_files[3]).name,
            "deaths": _small_preview_path(out_files[4]).name,
        },
        "counts": {
            "presence_points": int(presence_all_pts.shape[0]),
            "presence_ct_points": int(presence_ct_pts.shape[0]),
            "presence_t_points": int(presence_t_pts.shape[0]),
            "kills": int(kills_pts.shape[0]),
            "deaths": int(deaths_pts.shape[0]),
        },
        "cache_hits": cache_hits,
        "analyzer_version": ANALYZER_VERSION,
    }
