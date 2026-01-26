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
import json
import math
import os

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter
from scipy.ndimage import gaussian_filter, label, binary_closing, binary_opening
from matplotlib import cm
from awpy import Demo

ANALYZER_VERSION = "radar-smooth-v8-aligned"


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


# ----------------------------
# Public API
# ----------------------------
def build_heatmaps(dem_path: Path, out_dir: Path, steamid64: str) -> dict:
    out_dir = Path(out_dir)
    _ensure_dir(out_dir)

    dem = Demo(str(dem_path), verbose=False)
    dem.parse()
    map_name = dem.header.get("map_name", "unknown")

    radar, meta, radar_name = _prep_radar(map_name)
    w, h = radar.size
    map_mask_L = _build_map_mask(radar)

    # ---- PRESENCE ----
    ticks_df = dem.ticks.to_pandas()
    sid_col = _pick_existing(ticks_df, ["steamid", "steamID", "player_steamid", "playerSteamID"])
    xcol = _pick_existing(ticks_df, ["X", "x", "player_X", "player_x"])
    ycol = _pick_existing(ticks_df, ["Y", "y", "player_Y", "player_y"])
    tick_col = _pick_existing(ticks_df, ["tick", "ticks", "tick_num"])

    ticks_my = _filter_by_steamid_numeric(ticks_df, sid_col, steamid64)

    low = {c.lower(): c for c in ticks_my.columns}
    if "health" in low:
        hc = low["health"]
        alive = ticks_my[pd.to_numeric(ticks_my[hc], errors="coerce").fillna(0) > 0]
        if len(alive) > 0:
            ticks_my = alive

    ticks_my = _downsample_by_tick(ticks_my, tick_col)

    pts_px = _world_to_pixel(_to_points_xy(ticks_my, xcol, ycol), meta, (w, h))

    auto_dx = 0
    auto_dy = 0
    auto_score = 0.0
    if AUTO_OFFSET and pts_px.shape[0] > 200:
        auto_dx, auto_dy, auto_score = _auto_offset_shift(pts_px, map_mask_L)
        pts_px = _apply_shift(pts_px, auto_dx, auto_dy, (w, h))

    heat_all = _density_to_heat_rgba_pixel(
        pts_px, (w, h), cmap_name=CMAP_ALL, sigma_px=PRESENCE_SIGMA_PX, map_mask_L=map_mask_L
    )
    presence_png = out_dir / "presence_heatmap.png"
    _save_composited(radar, heat_all, presence_png)

    # CT/T split
    presence_ct_png = out_dir / "presence_heatmap_ct.png"
    presence_t_png = out_dir / "presence_heatmap_t.png"
    ct_count = 0
    t_count = 0

    if "side" in low:
        side_col = low["side"]
        side_norm = ticks_my[side_col].map(_normalize_side_value)

        ct_df = ticks_my[side_norm == "CT"]
        t_df = ticks_my[side_norm == "T"]

        ct_pts = _world_to_pixel(_to_points_xy(ct_df, xcol, ycol), meta, (w, h))
        t_pts = _world_to_pixel(_to_points_xy(t_df, xcol, ycol), meta, (w, h))

        # apply same auto shift to keep everything consistent
        if AUTO_OFFSET and (auto_dx or auto_dy):
            ct_pts = _apply_shift(ct_pts, auto_dx, auto_dy, (w, h))
            t_pts = _apply_shift(t_pts, auto_dx, auto_dy, (w, h))

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
    else:
        _make_placeholder_png(presence_ct_png, (w, h))
        _make_placeholder_png(presence_t_png, (w, h))

    # ---- KILLS / DEATHS ----
    kills_df = dem.kills.to_pandas()

    attacker_col = _pick_existing(kills_df, ["attacker_steamid", "killer_steamid", "attackerSteamID", "killerSteamID"])
    kills_my = _filter_by_steamid_numeric(kills_df, attacker_col, steamid64)
    kx = _pick_existing(kills_df, ["attacker_X", "attacker_x"])
    ky = _pick_existing(kills_df, ["attacker_Y", "attacker_y"])
    kill_pts = _world_to_pixel(_to_points_xy(kills_my, kx, ky), meta, (w, h))
    if AUTO_OFFSET and (auto_dx or auto_dy):
        kill_pts = _apply_shift(kill_pts, auto_dx, auto_dy, (w, h))

    kills_png = out_dir / "kills_heatmap.png"
    if kill_pts.shape[0]:
        heat_k = _density_to_heat_rgba_pixel(
            kill_pts, (w, h), cmap_name=CMAP_KILLS, sigma_px=KD_SIGMA_PX, map_mask_L=map_mask_L
        )
        _save_composited(radar, heat_k, kills_png)
    else:
        _make_placeholder_png(kills_png, (w, h))

    victim_col = _pick_existing(kills_df, ["victim_steamid", "victimSteamID"])
    deaths_my = _filter_by_steamid_numeric(kills_df, victim_col, steamid64)
    dx = _pick_existing(kills_df, ["victim_X", "victim_x"])
    dy = _pick_existing(kills_df, ["victim_Y", "victim_y"])
    death_pts = _world_to_pixel(_to_points_xy(deaths_my, dx, dy), meta, (w, h))
    if AUTO_OFFSET and (auto_dx or auto_dy):
        death_pts = _apply_shift(death_pts, auto_dx, auto_dy, (w, h))

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
            "kills": int(len(kills_my)),
            "deaths": int(len(deaths_my)),
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
