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
PHASE       = os.environ.get("PHASE","other")
LIMIT_DEMOS = int(os.environ.get("LIMIT_DEMOS","5"))

print("DEBUG SCRIPT START")
print("PROFILE_ID:", PROFILE_ID, "MAP_NAME:", MAP_NAME, "PHASE:", PHASE, "LIMIT_DEMOS:", LIMIT_DEMOS)

def main():
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

    # 1) parse all demos (v7 returns ParsedDemoEvents with list-of-dicts)
    parsed_list = []
    for p in dem_paths:
        try:
            parsed = demo_events.parse_demo_events(p, target_steam_id=steam64)
            parsed_list.append((p, parsed))
        except Exception as e:
            print("parse failed:", p.name, e)

    if not parsed_list:
        raise RuntimeError("No demos parsed successfully")

    # 2) compute map centers exactly like compute_kill_output_by_phase does
    config = demo_events._load_zone_config(MAP_NAME)
    plant_centers = demo_events._compute_site_centers_from_plants([pp for _, pp in parsed_list], MAP_NAME)
    centers = plant_centers or demo_events._site_centers(MAP_NAME, config)
    print("centers:", centers)

    # 3) collect kills classified into PHASE using the same helper used in metrics
    pts = []   # world coords
    info = []  # (demo, round, time, x, y, side)

    for demo_path, parsed in parsed_list:
        map_name = parsed.map_name or MAP_NAME
        config = demo_events._load_zone_config(map_name)

        # group target kills by round
        kills_by_round = {}
        for k in parsed.kills:
            if k.get("attacker_steamid64") == target_id:
                kills_by_round.setdefault(k.get("round"), []).append(k)

        rounds = set(kills_by_round.keys()) | set((parsed.tick_positions_by_round or {}).keys())
        for rn in sorted([r for r in rounds if isinstance(r,int)]):
            round_kills = kills_by_round.get(rn, [])

            target_side = demo_events._target_side_for_round(
                rn,
                parsed.target_round_sides,
                round_kills,
                target_id,
                None,
            )

            plant_info = (parsed.bomb_plants_by_round or {}).get(rn, {}) or {}
            plant_tick = plant_info.get("tick")
            plant_time = plant_info.get("time")

            # objective site inference (same as compute_kill_output_by_phase)
            objective_site = plant_info.get("site") if plant_info else None
            if objective_site not in {"A","B"}:
                round_positions = (parsed.tick_positions_by_round or {}).get(rn, [])
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
                phase = demo_events._assign_kill_phase(k, kill_side, map_name, config, centers, plant_info, objective_site)

                if phase == PHASE:
                    pts.append((float(x), float(y)))
                    info.append((demo_path.name, rn, k.get("time"), float(x), float(y), kill_side))

    print("phase:", PHASE, "kills:", len(pts))
    if info[:5]:
        print("samples:")
        for s in info[:5]:
            print("  ", s)

    if not pts:
        print("No points to plot. Exiting.")
        return

    # 4) project to radar pixels using analyzer meta (exactly your pipeline coords)
    radar_img, meta, radar_name = analyzer.load_radar_and_meta(MAP_NAME)
    w, h = radar_img.size
    xy = np.array(pts, dtype=np.float32)
    px = analyzer._world_to_pixel(xy, meta, (w, h))  # shape (n,2), already clipped to bounds
    if px.size == 0:
        print("All points clipped out of radar bounds. Exiting.")
        return

    # 5) draw overlay
    radar_arr = np.array(radar_img)

    plt.figure(figsize=(8,8), dpi=150)
    plt.imshow(radar_arr)
    plt.scatter(px[:,0], px[:,1], s=22, alpha=0.90)
    plt.title(f"{MAP_NAME} вЂ” {PHASE} kills (demos={len(parsed_list)})")
    plt.axis("off")

    out_dir = Path(settings.MEDIA_ROOT) / "debug"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"debug_{MAP_NAME}_{PHASE}_demos{len(parsed_list)}.png"
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    plt.close()

    print("Saved:", out_path)
print("RUN MAIN NOW")
try:
    main()
except Exception:
    import traceback
    traceback.print_exc()