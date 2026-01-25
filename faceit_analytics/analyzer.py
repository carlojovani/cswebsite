from __future__ import annotations

"""
CS2 Smooth Radar Heatmaps (NO hexes) — analyzer_v4

Fixes compared to previous:
- Radar no longer gets "over-darkened". We auto-detect darkness and only brighten if needed.
- Near-black background pixels are made transparent (so the card/background doesn't become a big dark square).
- Heatmap uses anti-hotspot normalization (log1p + percentile clip) and heavier smoothing for soft zones.

Outputs:
  presence_heatmap.png        (all sides)
  presence_heatmap_ct.png     (CT only, optional)
  presence_heatmap_t.png      (T only, optional)
  kills_heatmap.png
  deaths_heatmap.png
"""

from pathlib import Path
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter

from scipy.ndimage import gaussian_filter
from awpy import Demo

ANALYZER_VERSION = "radar-smooth-v4"


# ----------------------------
# Tuning (safe defaults)
# ----------------------------

# Heatmap look (soft zones)
HEAT_ALPHA_ALL = 0.88
HEAT_ALPHA_SIDE = 0.82

# Presence (most smooth)
PRESENCE_BINS  = 850
PRESENCE_SIGMA = 8.0
PRESENCE_PCTL  = 99.9
PRESENCE_GAMMA = 1.20

# K/D (slightly tighter)
KD_BINS  = 520
KD_SIGMA = 20.0
KD_PCTL  = 99.0
KD_GAMMA = 0.90

# Colormaps
CMAP_ALL = "inferno"
CMAP_KILLS = "magma"
CMAP_DEATHS = "plasma"
CMAP_CT = "Blues"
CMAP_T  = "Reds"

# Performance
MAX_PRESENCE_POINTS = 200_000

# Radar cleanup
# Make pixels darker than this transparent (removes big dark square)
RADAR_BG_THRESHOLD = 20          # 0..255, raise if your radar still has dark background
RADAR_ALPHA_BLUR = 0.8           # softens the transparency edge
# Auto-brighten only if radar is dark:
RADAR_DARK_MEAN_THRESHOLD = 45.0 # if mean brightness below -> brighten
RADAR_BRIGHTNESS = 1.50
RADAR_CONTRAST = 1.10
RADAR_GAMMA = 0.85               # <1 brightens mids


# ----------------------------
# Utils
# ----------------------------
def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _awpy_maps_dir() -> Path:
    return Path(os.path.expanduser("~")) / ".awpy" / "maps"


def _load_map_data(map_name: str) -> dict:
    maps_dir = _awpy_maps_dir()
    p = maps_dir / "map-data.json"
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


def _apply_gamma_rgba(im: Image.Image, gamma: float) -> Image.Image:
    if gamma == 1.0:
        return im.convert("RGBA")
    im = im.convert("RGBA")
    rgb = im.convert("RGB")
    arr = np.asarray(rgb).astype(np.float32) / 255.0
    arr = np.clip(arr, 0, 1) ** gamma
    rgb2 = Image.fromarray((arr * 255).astype(np.uint8), "RGB")
    out = Image.merge("RGBA", (*rgb2.split(), im.getchannel("A")))
    return out


def _radar_make_bg_transparent(radar: Image.Image) -> Image.Image:
    """Make near-black pixels transparent to avoid a big dark square behind the map."""
    r = radar.convert("RGBA")
    arr = np.array(r)
    rgb = arr[:, :, :3].astype(np.int16)
    bright = rgb.mean(axis=2)

    # alpha mask: keep where bright > threshold
    alpha = np.where(bright > RADAR_BG_THRESHOLD, 255, 0).astype(np.uint8)
    alpha_img = Image.fromarray(alpha, mode="L").filter(ImageFilter.GaussianBlur(RADAR_ALPHA_BLUR))
    arr[:, :, 3] = np.array(alpha_img)
    return Image.fromarray(arr, mode="RGBA")


def _radar_auto_brighten(radar: Image.Image) -> Image.Image:
    """Only brighten if radar is actually dark (prevents making good radars worse)."""
    r = radar.convert("RGBA")
    # compute mean brightness excluding fully transparent pixels
    arr = np.array(r)
    a = arr[:, :, 3].astype(np.float32) / 255.0
    rgb = arr[:, :, :3].astype(np.float32)
    bright = (rgb.mean(axis=2) * a)
    denom = (a > 0.05).sum()
    mean_bright = float(bright.sum() / max(1, denom))

    if mean_bright >= RADAR_DARK_MEAN_THRESHOLD:
        return r  # already bright enough

    # brighten
    r = ImageEnhance.Brightness(r).enhance(RADAR_BRIGHTNESS)
    r = ImageEnhance.Contrast(r).enhance(RADAR_CONTRAST)
    r = _apply_gamma_rgba(r, RADAR_GAMMA)
    return r


def _prep_radar(map_name: str) -> tuple[Image.Image, dict, str]:
    radar_path = _pick_radar_image(map_name)
    radar = Image.open(radar_path).convert("RGBA")

    radar = _radar_make_bg_transparent(radar)
    radar = _radar_auto_brighten(radar)

    meta = _load_map_data(map_name)
    return radar, meta, radar_path.name


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


def _to_points_xy(df: pd.DataFrame, xcol: str, ycol: str) -> list[tuple[float, float]]:
    xs = pd.to_numeric(df[xcol], errors="coerce")
    ys = pd.to_numeric(df[ycol], errors="coerce")
    pts: list[tuple[float, float]] = []
    for x, y in zip(xs, ys):
        if pd.notna(x) and pd.notna(y):
            fx, fy = float(x), float(y)
            if not (math.isnan(fx) or math.isnan(fy)):
                pts.append((fx, fy))
    return pts


def _sample(points: list[tuple[float, float]], max_points: int) -> list[tuple[float, float]]:
    if len(points) <= max_points:
        return points
    step = max(1, len(points) // max_points)
    return points[::step]


def _world_to_pixel(points_xy: list[tuple[float, float]], map_meta: dict, radar_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    w, h = radar_size
    pos_x = float(map_meta["pos_x"])
    pos_y = float(map_meta["pos_y"])
    scale = float(map_meta["scale"])

    xs = np.array([p[0] for p in points_xy], dtype=float)
    ys = np.array([p[1] for p in points_xy], dtype=float)

    px = (xs - pos_x) / scale
    py = (pos_y - ys) / scale

    m = (px >= 0) & (px < w) & (py >= 0) & (py < h)
    return px[m], py[m]


def _density_map(px: np.ndarray, py: np.ndarray, w: int, h: int, *, bins: int, sigma: float, pctl: float, gamma: float) -> np.ndarray:
    """histogram -> gaussian -> log1p -> percentile clip -> normalize -> gamma"""
    H, _, _ = np.histogram2d(px, py, bins=bins, range=[[0, w], [0, h]])
    H = gaussian_filter(H, sigma=sigma)
    H = np.log1p(H)

    vmax = np.percentile(H, pctl) if H.max() > 0 else 1.0
    vmax = float(vmax) if vmax and vmax > 0 else 1.0

    H = np.clip(H, 0, vmax) / vmax
    H = np.clip(H, 0, 1) ** gamma
    return H


def _save_overlay(
    radar: Image.Image,
    radar_name: str,
    map_name: str,
    meta: dict,
    points_xy: list[tuple[float, float]],
    out_path: Path,
    *,
    bins: int,
    sigma: float,
    pctl: float,
    gamma: float,
    cmap: str,
    alpha: float,
) -> dict:
    if not points_xy:
        raise RuntimeError("No points to plot")

    w, h = radar.size
    px, py = _world_to_pixel(points_xy, meta, (w, h))
    if px.size == 0:
        raise RuntimeError("All points outside radar bounds")

    H = _density_map(px, py, w, h, bins=bins, sigma=sigma, pctl=pctl, gamma=gamma)

    fig = plt.figure(figsize=(w / 220, h / 220), dpi=220, facecolor="none")
    ax = fig.add_subplot(111)
    ax.set_facecolor("none")

    ax.imshow(radar, extent=[0, w, h, 0])
    ax.imshow(H.T, extent=[0, w, h, 0], cmap=cmap, alpha=alpha, interpolation="bilinear")

    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_axis_off()

    _ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=240, bbox_inches="tight", pad_inches=0, transparent=True)
    plt.close(fig)

    return {
        "radar_used": radar_name,
        "bins": bins,
        "sigma": sigma,
        "pctl": pctl,
        "gamma": gamma,
        "points_in_bounds": int(px.size),
    }


def _make_placeholder_png(out_path: Path, text: str) -> None:
    _ensure_dir(out_path.parent)
    fig = plt.figure(facecolor="none")
    ax = fig.add_subplot(111)
    ax.set_facecolor("none")
    ax.text(0.5, 0.5, text, ha="center", va="center")
    ax.set_axis_off()
    fig.savefig(out_path, dpi=220, bbox_inches="tight", pad_inches=0, transparent=True)
    plt.close(fig)


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
# Public API (used by Django view)
# ----------------------------
def build_heatmaps(
    dem_path: Path,
    out_dir: Path,
    steamid64: str,
) -> dict:
    out_dir = Path(out_dir)
    _ensure_dir(out_dir)

    dem = Demo(str(dem_path), verbose=False)
    dem.parse()
    map_name = dem.header.get("map_name", "unknown")

    radar, meta, radar_name = _prep_radar(map_name)

    # ---- PRESENCE ----
    ticks_df = dem.ticks.to_pandas()
    sid_col = _pick_existing(ticks_df, ["steamid", "steamID", "player_steamid", "playerSteamID"])
    xcol = _pick_existing(ticks_df, ["X", "x", "player_X", "player_x"])
    ycol = _pick_existing(ticks_df, ["Y", "y", "player_Y", "player_y"])

    ticks_my = _filter_by_steamid_numeric(ticks_df, sid_col, steamid64)

    # only alive ticks if health exists
    lower = {c.lower(): c for c in ticks_my.columns}
    if "health" in lower:
        hcol = lower["health"]
        alive = ticks_my[pd.to_numeric(ticks_my[hcol], errors="coerce").fillna(0) > 0]
        if len(alive) > 0:
            ticks_my = alive

    presence_points = _sample(_to_points_xy(ticks_my, xcol, ycol), MAX_PRESENCE_POINTS)

    presence_png = out_dir / "presence_heatmap.png"
    presence_dbg = _save_overlay(
        radar, radar_name, map_name, meta, presence_points, presence_png,
        bins=PRESENCE_BINS, sigma=PRESENCE_SIGMA, pctl=PRESENCE_PCTL, gamma=PRESENCE_GAMMA,
        cmap=CMAP_ALL, alpha=HEAT_ALPHA_ALL,
    )

    # CT/T split presence for toggles
    presence_ct_png = out_dir / "presence_heatmap_ct.png"
    presence_t_png = out_dir / "presence_heatmap_t.png"
    presence_ct_dbg = {"note": "side column missing"}
    presence_t_dbg = {"note": "side column missing"}

    if "side" in lower:
        side_col = lower["side"]
        side_norm = ticks_my[side_col].map(_normalize_side_value)

        ct_df = ticks_my[side_norm == "CT"]
        t_df = ticks_my[side_norm == "T"]

        ct_pts = _sample(_to_points_xy(ct_df, xcol, ycol), MAX_PRESENCE_POINTS)
        t_pts = _sample(_to_points_xy(t_df, xcol, ycol), MAX_PRESENCE_POINTS)

        if ct_pts:
            presence_ct_dbg = _save_overlay(
                radar, radar_name, map_name, meta, ct_pts, presence_ct_png,
                bins=PRESENCE_BINS, sigma=PRESENCE_SIGMA, pctl=PRESENCE_PCTL, gamma=PRESENCE_GAMMA,
                cmap=CMAP_CT, alpha=HEAT_ALPHA_SIDE,
            )
        else:
            _make_placeholder_png(presence_ct_png, "No CT rounds for this player in this demo")
            presence_ct_dbg = {"note": "no CT points"}

        if t_pts:
            presence_t_dbg = _save_overlay(
                radar, radar_name, map_name, meta, t_pts, presence_t_png,
                bins=PRESENCE_BINS, sigma=PRESENCE_SIGMA, pctl=PRESENCE_PCTL, gamma=PRESENCE_GAMMA,
                cmap=CMAP_T, alpha=HEAT_ALPHA_SIDE,
            )
        else:
            _make_placeholder_png(presence_t_png, "No T rounds for this player in this demo")
            presence_t_dbg = {"note": "no T points"}

    # ---- KILLS ----
    kills_df = dem.kills.to_pandas()
    attacker_col = _pick_existing(kills_df, ["attacker_steamid", "killer_steamid", "attackerSteamID", "killerSteamID"])
    kills_my = _filter_by_steamid_numeric(kills_df, attacker_col, steamid64)

    kx = _pick_existing(kills_df, ["attacker_X", "attacker_x"])
    ky = _pick_existing(kills_df, ["attacker_Y", "attacker_y"])
    kill_points = _to_points_xy(kills_my, kx, ky)

    kills_png = out_dir / "kills_heatmap.png"
    if kill_points:
        kills_dbg = _save_overlay(
            radar, radar_name, map_name, meta, kill_points, kills_png,
            bins=KD_BINS, sigma=KD_SIGMA, pctl=KD_PCTL, gamma=KD_GAMMA,
            cmap=CMAP_KILLS, alpha=0.90,
        )
    else:
        _make_placeholder_png(kills_png, "No kills for this player in this demo")
        kills_dbg = {"note": "no kills"}

    # ---- DEATHS ----
    victim_col = _pick_existing(kills_df, ["victim_steamid", "victimSteamID"])
    deaths_my = _filter_by_steamid_numeric(kills_df, victim_col, steamid64)

    dx = _pick_existing(kills_df, ["victim_X", "victim_x"])
    dy = _pick_existing(kills_df, ["victim_Y", "victim_y"])
    death_points = _to_points_xy(deaths_my, dx, dy)

    deaths_png = out_dir / "deaths_heatmap.png"
    if death_points:
        deaths_dbg = _save_overlay(
            radar, radar_name, map_name, meta, death_points, deaths_png,
            bins=KD_BINS, sigma=KD_SIGMA, pctl=KD_PCTL, gamma=KD_GAMMA,
            cmap=CMAP_DEATHS, alpha=0.90,
        )
    else:
        _make_placeholder_png(deaths_png, "No deaths for this player in this demo")
        deaths_dbg = {"note": "no deaths"}

    return {
        "steamid64": str(steamid64),
        "map": map_name,
        "counts": {
            "kills": int(len(kills_my)),
            "deaths": int(len(deaths_my)),
            "presence_points": int(len(presence_points)),
        },
        "analyzer_version": ANALYZER_VERSION,
        "files": {
            "presence": presence_png.name,
            "presence_ct": presence_ct_png.name,
            "presence_t": presence_t_png.name,
            "kills": kills_png.name,
            "deaths": deaths_png.name,
        },
        "debug": {
            "presence": presence_dbg,
            "presence_ct": presence_ct_dbg,
            "presence_t": presence_t_dbg,
            "kills": kills_dbg,
            "deaths": deaths_dbg,
        },
    }
