import os
from pathlib import Path
import numpy as np

from django.conf import settings
from users.models import PlayerProfile
from faceit_analytics.services import demo_events

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROFILE_ID = int(os.environ.get("PROFILE_ID", "2"))
MAP_NAME   = os.environ.get("MAP_NAME", "de_mirage")
DEMO_NAME  = os.environ.get("DEMO_NAME", "match5.dem")
RES        = int(os.environ.get("RES", "1024"))
PHASE      = os.environ.get("PHASE", "other")  # e.g. other, ct_hold, ct_push, t_execute...

def get_df(parsed, names):
    for n in names:
        if hasattr(parsed, n):
            return n, getattr(parsed, n)
    # sometimes ParsedDemoEvents stores dict-like data
    for n in names:
        try:
            if hasattr(parsed, "data") and isinstance(parsed.data, dict) and n in parsed.data:
                return f"data['{n}']", parsed.data[n]
        except Exception:
            pass
    return None, None

def _pick_col(df, names):
    cols = set(getattr(df, "columns", []))
    for n in names:
        if n in cols:
            return n
    return None

def main():
    profile = PlayerProfile.objects.get(id=PROFILE_ID)
    steam64 = str(profile.steam_id)

    media = Path(settings.MEDIA_ROOT)
    dem_path = media / "local_demos" / steam64 / MAP_NAME / DEMO_NAME
    print("Demo:", dem_path, "exists:", dem_path.exists())

    parsed = demo_events.parse_demo_events(dem_path, target_steam_id=steam64)

    # --- locate kills + bomb plants robustly ---
    kills_name, kills = get_df(parsed, ["kills_df","kills","df_kills","kills_pl","kills_polars"])
    bomb_name, bomb   = get_df(parsed, ["bomb_df","bomb","df_bomb","bomb_pl","bomb_polars"])

    print("Parsed type:", type(parsed))
    print("Kills df source:", kills_name, "type:", type(kills))
    print("Bomb df source:", bomb_name, "type:", type(bomb))

    if kills is None:
        # dump available attrs to help
        attrs = [a for a in dir(parsed) if a.endswith("_df") or "kill" in a.lower() or "bomb" in a.lower()]
        raise RuntimeError(f"Cannot find kills df on parsed. Candidate attrs: {attrs}")

    # filter by attacker steam
    steam_col = _pick_col(kills, ["attacker_steamid64","attackerSteamid64","attacker_steamid","attackerSteamid","attackerSteamId"])
    if steam_col is None:
        raise RuntimeError(f"Cannot find attacker steamid column. kills cols sample: {list(getattr(kills,'columns',[]))[:40]}")

    k = kills.filter(kills[steam_col].cast(str) == steam64)

    # coords + tick + round
    axc  = _pick_col(k, ["attacker_X","attacker_x","attackerX","attackerXPos","attacker_x_pos","X","x"])
    ayc  = _pick_col(k, ["attacker_Y","attacker_y","attackerY","attackerYPos","attacker_y_pos","Y","y"])
    tickc= _pick_col(k, ["tick","Tick"])
    rndc = _pick_col(k, ["round_num","round","roundNum"])

    if axc is None or ayc is None or tickc is None or rndc is None:
        raise RuntimeError(
            f"Missing required columns. ax={axc} ay={ayc} tick={tickc} round={rndc}. "
            f"kills cols sample: {list(k.columns)[:60]}"
        )

    # plants lookup: round -> (plant_tick, plant_x, plant_y)
    plant_by_round = {}
    if bomb is not None and hasattr(bomb, "height") and bomb.height > 0 and "event" in bomb.columns:
        plants = bomb.filter(bomb["event"] == "plant")
        if plants.height > 0:
            prnd = _pick_col(plants, ["round_num","round","roundNum"])
            ptick= _pick_col(plants, ["tick","Tick"])
            px   = _pick_col(plants, ["X","x","plant_X","plant_x","plantX"])
            py   = _pick_col(plants, ["Y","y","plant_Y","plant_y","plantY"])
            if prnd and ptick and px and py:
                for rn, tk, x, y in plants.select([prnd, ptick, px, py]).iter_rows():
                    plant_by_round[int(rn)] = (int(tk), float(x), float(y))

    # phase assignment function
    assign_fn = getattr(demo_events, "_assign_kill_phase", None)
    if assign_fn is None:
        # some versions might name it differently
        assign_fn = getattr(demo_events, "assign_kill_phase", None)
    if assign_fn is None:
        raise RuntimeError("Cannot find kill phase assignment function in demo_events (expected _assign_kill_phase or assign_kill_phase).")

    # map config (optional)
    config = getattr(demo_events, "MAP_ZONES", {}).get(MAP_NAME) if hasattr(demo_events, "MAP_ZONES") else None

    pts = []
    for rn, tk, x, y in k.select([rndc, tickc, axc, ayc]).iter_rows():
        rn = int(rn); tk = int(tk)
        x = float(x); y = float(y)

        plant = plant_by_round.get(rn)
        plant_tick = plant[0] if plant else None
        plant_x = plant[1] if plant else None
        plant_y = plant[2] if plant else None

        # try both signatures
        try:
            phase = assign_fn(
                side=None,
                kill_tick=tk,
                x=x, y=y,
                plant_tick=plant_tick, plant_x=plant_x, plant_y=plant_y,
                map_name=MAP_NAME, config=config,
            )
        except TypeError:
            phase = assign_fn(tk, x, y, plant_tick, plant_x, plant_y, MAP_NAME, config)

        if phase == PHASE:
            pts.append((x, y))

    print("phase:", PHASE, "kills:", len(pts))
    if not pts:
        print("No points. Exiting.")
        return

    # radar image
    candidates = [
        Path(settings.BASE_DIR) / "faceit_analytics" / "static" / "faceit_analytics" / "radar" / f"{MAP_NAME}.png",
        Path(settings.BASE_DIR) / "static" / "radar" / f"{MAP_NAME}.png",
        Path(settings.BASE_DIR) / "static" / "radars" / f"{MAP_NAME}.png",
    ]
    radar_path = next((p for p in candidates if p.exists()), None)
    if radar_path is None:
        raise RuntimeError("Radar image not found. Add correct path candidates in script.")

    import matplotlib.image as mpimg
    radar = mpimg.imread(radar_path)

    # world -> radar px
    w2p = getattr(demo_events, "world_to_radar_px", None) or getattr(demo_events, "_world_to_radar_px", None)
    if w2p is None:
        raise RuntimeError("world->radar conversion helper not found in demo_events.")

    xs, ys = [], []
    for x, y in pts:
        px, py = w2p(MAP_NAME, x, y, RES)
        xs.append(px); ys.append(py)

    plt.figure(figsize=(8,8), dpi=max(120, RES//8))
    plt.imshow(radar)
    plt.scatter(xs, ys, s=22, alpha=0.9)
    plt.title(f"{MAP_NAME} — {PHASE} kills (demo={DEMO_NAME})")
    plt.axis("off")

    out_dir = Path(settings.MEDIA_ROOT) / "debug"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"debug_{MAP_NAME}_{PHASE}_{DEMO_NAME.replace('.dem','')}.png"
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    plt.close()

    print("Saved:", out_path)

print("DEBUG SCRIPT START")
try:
    main()
except Exception:
    import traceback
    traceback.print_exc()