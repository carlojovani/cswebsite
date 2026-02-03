import os
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

# phases to plot (must match your _assign_kill_phase outputs)
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

def main():
    print("CT HOLD/PUSH/ROAM OVERLAY START")
    print("PROFILE_ID:", PROFILE_ID, "MAP_NAME:", MAP_NAME, "LIMIT_DEMOS:", LIMIT_DEMOS, "RES:", RES)

    profile = PlayerProfile.objects.get(id=PROFILE_ID)
    steam64 = str(profile.steam_id)
    target_id = demo_events.normalize_steamid64(steam64)
    if target_id is None:
        raise RuntimeError(f"normalize_steamid64 failed for {steam64}")

    demos_dir = Path(settings.MEDIA_ROOT) / "local_demos" / steam64 / MAP_NAME
    if not demos_dir.exists():
        raise RuntimeError(f"demos_dir not found: {demos_dir}")

    dem_paths = sorted([p for p in demos_dir.glob("*.dem")])
    if LIMIT_DEMOS > 0:
        dem_paths = dem_paths[:LIMIT_DEMOS]

    print("demos_dir:", demos_dir)
    print("demo files:", [p.name for p in dem_paths])

    parsed_list = []
    for p in dem_paths:
        try:
            parsed = demo_events.parse_demo_events(p, target_steam_id=steam64)
            parsed_list.append((p, parsed))
        except Exception as e:
            print("parse failed:", p.name, e)

    if not parsed_list:
        raise RuntimeError("No demos parsed successfully")

    # centers (same as your previous working debug script)
    cfg0 = demo_events._load_zone_config(MAP_NAME)
    plant_centers = demo_events._compute_site_centers_from_plants([pp for _, pp in parsed_list], MAP_NAME)
    centers = plant_centers or demo_events._site_centers(MAP_NAME, cfg0)
    print("centers:", centers)

    pts_by_phase = {k: [] for k, _, _, _ in PHASES}
    counts_by_phase = {k: 0 for k, _, _, _ in PHASES}

    assign_fn = demo_events._assign_kill_phase

    for demo_path, parsed in parsed_list:
        map_name = getattr(parsed, "map_name", None) or MAP_NAME
        config = demo_events._load_zone_config(map_name)

        kills = getattr(parsed, "kills", None) or []
        ticks_by_round = getattr(parsed, "tick_positions_by_round", None) or {}
        bomb_plants = getattr(parsed, "bomb_plants_by_round", None) or {}

        # group target kills by round
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

            for k in round_kills:
                x = k.get("attacker_x")
                y = k.get("attacker_y")
                if x is None or y is None:
                    continue

                kill_side = target_side or demo_events._normalize_side(k.get("attacker_side"))
                if kill_side != "CT":
                    continue

                kwargs = {}
                if _sig_accepts(assign_fn, "kill_place"):
                    kwargs["kill_place"] = k.get("place")
                if _sig_accepts(assign_fn, "t_round"):
                    kwargs["t_round"] = k.get("time")

                phase = assign_fn(k, kill_side, map_name, config, centers, plant_info, objective_site, **kwargs)

                if phase in pts_by_phase:
                    pts_by_phase[phase].append((float(x), float(y)))
                    counts_by_phase[phase] += 1

    # Radar projection
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
            print("phase", key, "-> 0 points")
            continue
        any_points = True
        xy = np.array(pts, dtype=np.float32)
        px = analyzer._world_to_pixel(xy, meta, (w, h))
        if px.size == 0:
            print("phase", key, "-> all clipped out")
            continue

        plt.scatter(
            px[:, 0], px[:, 1],
            s=28, alpha=0.92,
            marker=marker,
            c=color,
            label=f"{label} ({len(pts)})"
        )

    if not any_points:
        print("No CT points to plot for hold/push/roam.")
        return

    plt.title(f"{MAP_NAME} — CT Hold / CT Push / CT Roam (demos={len(parsed_list)})")
    plt.legend(loc="lower right", framealpha=0.85)

    out_dir = Path(settings.MEDIA_ROOT) / "debug"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"debug_{MAP_NAME}_ct_hold_push_roam_demos{len(parsed_list)}.png"
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    plt.close()

    print("Saved:", out_path)

print("RUN MAIN NOW")
try:
    main()
except Exception:
    import traceback
    traceback.print_exc()