from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from awpy import Demo


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str:
    cols = list(df.columns)
    low = {c.lower(): c for c in cols}
    for c in candidates:
        if c in cols:
            return c
        if c.lower() in low:
            return low[c.lower()]
    raise KeyError(f"Missing columns. Tried: {candidates}. Have: {cols[:50]}...")


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def _save_hex_heatmap(points_xy, out_path: Path, title: str):
    """
    Рисуем heatmap без радара: просто плотность точек (XY).
    """
    _ensure_dir(out_path.parent)

    xs = [p[0] for p in points_xy]
    ys = [p[1] for p in points_xy]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    # hexbin хорошо работает на больших массивах
    ax.hexbin(xs, ys, gridsize=70)  # без цветов по твоим правилам
    ax.set_aspect("equal", adjustable="box")

    fig.savefig(out_path, dpi=160, bbox_inches="tight", transparent=True)
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

    map_name = dem.header.get("map_name", "unknown")

    # ---- KILLS (где ты убиваешь): координаты жертвы
    kills = dem.kills.to_pandas()
    attacker_id_col = _pick_col(
        kills, ["attacker_steamid", "killer_steamid", "attackerSteamID", "killerSteamID"]
    )
    my_kills = kills[kills[attacker_id_col].astype(str) == str(steamid64)]

    vx = _pick_col(my_kills, ["victim_X", "victim_x"])
    vy = _pick_col(my_kills, ["victim_Y", "victim_y"])

    kill_points = list(zip(my_kills[vx].astype(float), my_kills[vy].astype(float)))
    kills_png = out_dir / "kills_heatmap.png"
    _save_hex_heatmap(kill_points, kills_png, f"Kills heatmap ({map_name})")

    # ---- PRESENCE (где находишься): тики + семпл
    ticks = dem.ticks.to_pandas()

    # если есть steamid в тиках — фильтруем
    try:
        tick_id_col = _pick_col(ticks, ["steamid", "steamID", "player_steamid", "playerSteamID"])
        ticks = ticks[ticks[tick_id_col].astype(str) == str(steamid64)]
    except Exception:
        pass

    tx = _pick_col(ticks, ["X", "x", "player_X", "player_x"])
    ty = _pick_col(ticks, ["Y", "y", "player_Y", "player_y"])

    if len(ticks) > max_presence_points:
        step = max(1, len(ticks) // max_presence_points)
        ticks_s = ticks.iloc[::step].copy()
    else:
        ticks_s = ticks.iloc[::16].copy()

    presence_points = list(zip(ticks_s[tx].astype(float), ticks_s[ty].astype(float)))
    presence_png = out_dir / "presence_heatmap.png"
    _save_hex_heatmap(presence_points, presence_png, f"Presence heatmap ({map_name})")

    return {
        "map": map_name,
        "steamid64": str(steamid64),
        "kills": int(len(my_kills)),
        "kills_points": int(len(kill_points)),
        "presence_points": int(len(presence_points)),
        "files": {"kills": kills_png.name, "presence": presence_png.name},
    }
