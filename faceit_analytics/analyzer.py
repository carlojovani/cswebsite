from __future__ import annotations

from pathlib import Path
import math

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from awpy import Demo
from awpy.plot import heatmap as awpy_heatmap

ANALYZER_VERSION = "radar-awpy-heatmap-v1"


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _pick_existing(df: pd.DataFrame, candidates: list[str]) -> str:
    low = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in low:
            return low[c.lower()]
    raise KeyError(f"None of {candidates} found. Have: {list(df.columns)[:80]}...")


def _filter_by_steamid_numeric(df: pd.DataFrame, col: str, steamid64: str) -> pd.DataFrame:
    # attacker_steamid может быть float64 -> сравниваем числом
    target = int(str(steamid64))
    s = pd.to_numeric(df[col], errors="coerce")
    return df[s.eq(target)]


def _to_points_xyz(df: pd.DataFrame, xcol: str, ycol: str, z_value: float = 0.0) -> list[tuple[float, float, float]]:
    xs = pd.to_numeric(df[xcol], errors="coerce")
    ys = pd.to_numeric(df[ycol], errors="coerce")
    pts: list[tuple[float, float, float]] = []
    for x, y in zip(xs, ys):
        if pd.notna(x) and pd.notna(y):
            fx, fy = float(x), float(y)
            if not (math.isnan(fx) or math.isnan(fy)):
                pts.append((fx, fy, float(z_value)))
    return pts


def _sample(points: list[tuple[float, float, float]], max_points: int) -> list[tuple[float, float, float]]:
    if len(points) <= max_points:
        return points
    step = max(1, len(points) // max_points)
    return points[::step]


def _save_awpy_heatmap(map_name: str, points_xyz: list[tuple[float, float, float]], out_path: Path, title: str) -> dict:
    if not points_xyz:
        raise RuntimeError(f"No points to plot for {title}")

    _ensure_dir(out_path.parent)

    fig, ax = awpy_heatmap(
        map_name=map_name,
        points=points_xyz,
        method="hex",  # "hex" / "hist" / "kde"
        size=12,
        alpha=0.7,
    )
    ax.set_title(title)

    fig.savefig(out_path, dpi=220, bbox_inches="tight", pad_inches=0, transparent=True)
    plt.close(fig)

    xs = [p[0] for p in points_xyz]
    ys = [p[1] for p in points_xyz]
    return {"x_min": float(min(xs)), "x_max": float(max(xs)), "y_min": float(min(ys)), "y_max": float(max(ys))}


def _make_placeholder_png(out_path: Path, text: str) -> None:
    _ensure_dir(out_path.parent)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.text(0.5, 0.5, text, ha="center", va="center")
    ax.set_axis_off()
    fig.savefig(out_path, dpi=220, bbox_inches="tight", pad_inches=0, transparent=True)
    plt.close(fig)


def build_heatmaps(
    dem_path: Path,
    out_dir: Path,
    steamid64: str,
    max_presence_points: int = 120_000,
) -> dict:
    out_dir = Path(out_dir)
    _ensure_dir(out_dir)

    dem = Demo(str(dem_path), verbose=False)
    dem.parse()

    map_name = dem.header.get("map_name", "unknown")  # должно быть de_*
    # ---------- PRESENCE ----------
    ticks_df = dem.ticks.to_pandas()
    sid_col = _pick_existing(ticks_df, ["steamid", "steamID", "player_steamid", "playerSteamID"])
    xcol = _pick_existing(ticks_df, ["X", "x", "player_X", "player_x"])
    ycol = _pick_existing(ticks_df, ["Y", "y", "player_Y", "player_y"])

    ticks_my = _filter_by_steamid_numeric(ticks_df, sid_col, steamid64)

    # живые тики (если есть health)
    lower_cols = {c.lower(): c for c in ticks_my.columns}
    if "health" in lower_cols:
        hcol = lower_cols["health"]
        alive = ticks_my[pd.to_numeric(ticks_my[hcol], errors="coerce").fillna(0) > 0]
        if len(alive) > 0:
            ticks_my = alive

    presence_pts = _to_points_xyz(ticks_my, xcol, ycol, z_value=0.0)
    presence_pts = _sample(presence_pts, max_presence_points)

    presence_png = out_dir / "presence_heatmap.png"
    presence_mm = _save_awpy_heatmap(map_name, presence_pts, presence_png, f"Presence ({map_name})")

    # ---------- KILLS (where you stood when killing) ----------
    kills_df = dem.kills.to_pandas()
    attacker_col = _pick_existing(kills_df, ["attacker_steamid", "killer_steamid", "attackerSteamID", "killerSteamID"])
    kills_my = _filter_by_steamid_numeric(kills_df, attacker_col, steamid64)

    kx = _pick_existing(kills_df, ["attacker_X", "attacker_x"])
    ky = _pick_existing(kills_df, ["attacker_Y", "attacker_y"])
    kill_pts = _to_points_xyz(kills_my, kx, ky, z_value=0.0)

    kills_png = out_dir / "kills_heatmap.png"
    if kill_pts:
        kills_mm = _save_awpy_heatmap(map_name, kill_pts, kills_png, f"Kills (you) ({map_name})")
    else:
        _make_placeholder_png(kills_png, "No kills for this player in this demo")
        kills_mm = {"note": "no kills"}

    # ---------- DEATHS (where you died) ----------
    victim_col = _pick_existing(kills_df, ["victim_steamid", "victimSteamID"])
    deaths_my = _filter_by_steamid_numeric(kills_df, victim_col, steamid64)

    dx = _pick_existing(kills_df, ["victim_X", "victim_x"])
    dy = _pick_existing(kills_df, ["victim_Y", "victim_y"])
    death_pts = _to_points_xyz(deaths_my, dx, dy, z_value=0.0)

    deaths_png = out_dir / "deaths_heatmap.png"
    if death_pts:
        deaths_mm = _save_awpy_heatmap(map_name, death_pts, deaths_png, f"Deaths ({map_name})")
    else:
        _make_placeholder_png(deaths_png, "No deaths for this player in this demo")
        deaths_mm = {"note": "no deaths"}

    return {
        "analyzer_version": ANALYZER_VERSION,
        "map": map_name,
        "steamid64": str(steamid64),
        "counts": {
            "kills": int(len(kills_my)),
            "deaths": int(len(deaths_my)),
            "presence_points": int(len(presence_pts)),
        },
        "debug": {
            "presence": {"sid_col": sid_col, "xcol": xcol, "ycol": ycol, **presence_mm},
            "kills": {"attacker_col": attacker_col, "xcol": kx, "ycol": ky, **kills_mm},
            "deaths": {"victim_col": victim_col, "xcol": dx, "ycol": dy, **deaths_mm},
        },
        "files": {
            "presence": presence_png.name,
            "kills": kills_png.name,
            "deaths": deaths_png.name,
        },
    }
