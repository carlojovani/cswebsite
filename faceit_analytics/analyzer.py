from pathlib import Path

import pandas as pd
from awpy import Demo
from awpy.plot import heatmap


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str:
    cols = list(df.columns)
    low = {c.lower(): c for c in cols}
    for c in candidates:
        if c in cols:
            return c
        if c.lower() in low:
            return low[c.lower()]
    raise KeyError(f"Missing columns. Tried: {candidates}. Have: {cols[:50]}...")


def _xyz(df: pd.DataFrame, x: str, y: str, z: str):
    return list(zip(df[x].astype(float), df[y].astype(float), df[z].astype(float)))


def build_heatmaps(
    dem_path: Path,
    out_dir: Path,
    steamid64: str,
    max_presence_points: int = 120_000,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    dem = Demo(str(dem_path), verbose=False)
    dem.parse()

    map_name = dem.header.get("map_name")
    if not map_name:
        raise RuntimeError("Could not detect map_name from demo header")

    kills = dem.kills.to_pandas()
    attacker_id_col = _pick_col(
        kills,
        ["attacker_steamid", "killer_steamid", "attackerSteamID", "killerSteamID"],
    )
    my_kills = kills[kills[attacker_id_col].astype(str) == str(steamid64)]

    vx = _pick_col(my_kills, ["victim_X", "victim_x"])
    vy = _pick_col(my_kills, ["victim_Y", "victim_y"])
    vz = _pick_col(my_kills, ["victim_Z", "victim_z"])
    pts_kills = _xyz(my_kills, vx, vy, vz)

    fig, _ = heatmap(map_name=map_name, points=pts_kills, method="kde")
    kills_png = out_dir / "kills_heatmap.png"
    fig.savefig(kills_png, dpi=160, bbox_inches="tight", transparent=True)

    ticks = dem.ticks.to_pandas()

    try:
        tick_id_col = _pick_col(
            ticks,
            ["steamid", "steamID", "player_steamid", "playerSteamID"],
        )
        ticks = ticks[ticks[tick_id_col].astype(str) == str(steamid64)]
    except Exception:
        pass

    tx = _pick_col(ticks, ["X", "x", "player_X", "player_x"])
    ty = _pick_col(ticks, ["Y", "y", "player_Y", "player_y"])
    tz = _pick_col(ticks, ["Z", "z", "player_Z", "player_z"])

    if len(ticks) > max_presence_points:
        step = max(1, len(ticks) // max_presence_points)
        ticks_s = ticks.iloc[::step].copy()
    else:
        ticks_s = ticks.iloc[::16].copy()

    pts_presence = _xyz(ticks_s, tx, ty, tz)

    fig, _ = heatmap(map_name=map_name, points=pts_presence, method="hist")
    presence_png = out_dir / "presence_heatmap.png"
    fig.savefig(presence_png, dpi=160, bbox_inches="tight", transparent=True)

    return {
        "map": map_name,
        "steamid64": str(steamid64),
        "kills": int(len(my_kills)),
        "kills_points": int(len(pts_kills)),
        "presence_points": int(len(pts_presence)),
        "files": {
            "kills": str(kills_png.name),
            "presence": str(presence_png.name),
        },
    }
