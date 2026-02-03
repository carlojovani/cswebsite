import os, csv, math
from pathlib import Path
import numpy as np

from django.conf import settings
from users.models import PlayerProfile
from faceit_analytics.services import demo_events
from faceit_analytics import analyzer

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROFILE_ID  = int(os.environ.get("PROFILE_ID","2"))
MAP_NAME    = os.environ.get("MAP_NAME","de_mirage")
LIMIT_DEMOS = int(os.environ.get("LIMIT_DEMOS","5"))
RES         = int(os.environ.get("RES","1024"))

PHASES = [
    ("ct_hold", "CT Hold", "o", "#1F77B4"),
    ("ct_push", "CT Push", "^", "#FF7F0E"),
    ("ct_roam", "CT Roam", "x", "#2CA02C"),
]

def _sig_accepts(fn, name: str) -> bool:
    try:
        import inspect
        return name in inspect.signature(fn).parameters
    except Exception:
        return False

def _dist(a, b):
    if a is None or b is None:
        return None
    ax, ay = a
    bx, by = b
    return float(math.hypot(ax - bx, ay - by))

def main():
    print("CT HOLD/PUSH/ROAM OVERLAY + CSV START")
    profile = PlayerProfile.objects.get(id=PROFILE_ID)
    steam64 = str(profile.steam_id)
    target_id = demo_events.normalize_steamid64(steam64)

    demos_dir = Path(settings.MEDIA_ROOT) / "local_demos" / steam64 / MAP_NAME
    dem_paths = sorted([p for p in demos_dir.glob("*.dem")])
    if LIMIT_DEMOS > 0:
        dem_paths = dem_paths[:LIMIT_DEMOS]

    parsed_list = []
    for p in dem_paths:
        try:
            parsed = demo_events.parse_demo_events(p, target_steam_id=steam64)
            parsed_list.append((p, parsed))
        except Exception as e:
            print("parse failed:", p.name, e)

    cfg0 = demo_events._load_zone_config(MAP_NAME)
    plant_centers = demo_events._compute_site_centers_from_plants([pp for _, pp in parsed_list], MAP_NAME)
    centers = plant_centers or demo_events._site_centers(MAP_NAME, cfg0)

    pts_by_phase = {k: [] for k, _, _, _ in PHASES}

    assign_fn = demo_events._assign_kill_phase
    accept_place = _sig_accepts(assign_fn, "kill_place")
    accept_tround = _sig_accepts(assign_fn, "t_round")

    rows = []  # for csv

    for demo_path, parsed in parsed_list:
        map_name = getattr(parsed, "map_name", None) or MAP_NAME
        config = demo_events._load_zone_config(map_name)

        kills = getattr(parsed, "kills", None) or []
        ticks_by_round = getattr(parsed, "tick_positions_by_round", None) or {}
        bomb_plants = getattr(parsed, "bomb_plants_by_round", None) or {}

        kills_by_round = {}
        for k in kills:
            if k.get("attacker_steamid64") == target_id:
                rn = k.get("round")
                if isinstance(rn, int):
                    kills_by_round.setdefault(rn, []).append(k)

        rounds = set(kills_by_round.keys()) | set(ticks_by_round.keys())
        for rn in sorted([r for r in rounds if isinstance(r, int)]):
            round_kills = kills_by_round.get(rn, [])

            target_side = demo_events._target_side_for_round(
                rn,
                getattr(parsed, "target_round_sides", None),
                round_kills,
                target_id,
                None,
            )

            plant_info = bomb_plants.get(rn, {}) or {}
            plant_tick = plant_info.get("tick")
            plant_time = plant_info.get("time")

            objective_site = plant_info.get("site") if plant_info else None
            if objective_site not in {"A", "B"}:
                round_positions = ticks_by_round.get(rn, []) or []
                target_positions = [pos for pos in round_positions if pos.get("is_target")]
                objective_site = demo_events._infer_objective_site_from_ticks(
                    target_positions,
                    round_kills,
                    map_name,
                    config,
                    centers,
                    plant_tick,
                    plant_time,
                )

            plant_xy = None
            if isinstance(plant_info, dict) and plant_info.get("x") is not None and plant_info.get("y") is not None:
                plant_xy = (float(plant_info["x"]), float(plant_info["y"]))

            centerA = centers.get("A")
            centerB = centers.get("B")

            for k in round_kills:
                x = k.get("attacker_x")
                y = k.get("attacker_y")
                if x is None or y is None:
                    continue

                kill_side = target_side or demo_events._normalize_side(k.get("attacker_side"))
                if kill_side != "CT":
                    continue

                kwargs = {}
                if accept_place:
                    kwargs["kill_place"] = k.get("place")
                if accept_tround:
                    kwargs["t_round"] = k.get("time")

                phase = assign_fn(k, kill_side, map_name, config, centers, plant_info, objective_site, **kwargs)

                if phase in pts_by_phase:
                    pts_by_phase[phase].append((float(x), float(y)))

                # distances for debugging
                xy = (float(x), float(y))
                distA = _dist(xy, centerA) if isinstance(centerA, (tuple, list)) else None
                distB = _dist(xy, centerB) if isinstance(centerB, (tuple, list)) else None
                distPlant = _dist(xy, plant_xy) if plant_xy else None

                postplant = False
                if plant_tick is not None and k.get("tick") is not None:
                    try:
                        postplant = int(k["tick"]) >= int(plant_tick)
                    except Exception:
                        postplant = False

                rows.append({
                    "demo": demo_path.name,
                    "round": rn,
                    "tick": k.get("tick"),
                    "t_round": k.get("time"),
                    "place": k.get("place"),
                    "objective_site": objective_site,
                    "postplant": int(postplant),
                    "phase": phase,
                    "x": float(x),
                    "y": float(y),
                    "distA": distA,
                    "distB": distB,
                    "distPlant": distPlant,
                })

    # Radar projection + plot
    radar_img, meta, radar_name = analyzer.load_radar_and_meta(MAP_NAME)
    w, h = radar_img.size
    radar_arr = np.array(radar_img)

    plt.figure(figsize=(9, 9), dpi=max(120, RES // 8))
    plt.imshow(radar_arr)
    plt.axis("off")

    any_points = False
    for key, label, marker, color in PHASES:
        pts = pts_by_phase.get(key, [])
        if not pts:
            continue
        any_points = True
        xy = np.array(pts, dtype=np.float32)
        px = analyzer._world_to_pixel(xy, meta, (w, h))
        if px.size == 0:
            continue
        plt.scatter(px[:, 0], px[:, 1], s=28, alpha=0.92, marker=marker, c=color, label=f"{label} ({len(pts)})")

    out_dir = Path(settings.MEDIA_ROOT) / "debug"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_png = out_dir / f"debug_{MAP_NAME}_ct_hold_push_roam_demos{len(parsed_list)}.png"
    if any_points:
        plt.title(f"{MAP_NAME} — CT Hold / CT Push / CT Roam (demos={len(parsed_list)})")
        plt.legend(loc="lower right", framealpha=0.85)
        plt.savefig(out_png, bbox_inches="tight", pad_inches=0.05)
    plt.close()

    out_csv = out_dir / f"debug_{MAP_NAME}_ct_hold_push_roam_rows_demos{len(parsed_list)}.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["demo","round","tick","t_round","place","objective_site","postplant","phase","x","y","distA","distB","distPlant"]
        wri = csv.DictWriter(f, fieldnames=fieldnames)
        wri.writeheader()
        for r in rows:
            wri.writerow(r)

    print("Saved PNG:", out_png)
    print("Saved CSV:", out_csv)
    print("Rows:", len(rows))

print("RUN MAIN NOW")
try:
    main()
except Exception:
    import traceback
    traceback.print_exc()