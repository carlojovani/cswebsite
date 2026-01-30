from __future__ import annotations

from pathlib import Path

import numpy as np

from faceit_analytics import analyzer
from faceit_analytics.services.demo_events import normalize_steamid64, parse_demo_events
from faceit_analytics.services.paths import get_demos_dir
from users.models import PlayerProfile


def main() -> None:
    profile = PlayerProfile.objects.order_by("-id").first()
    if not profile:
        print("No PlayerProfile found.")
        return

    map_name = "de_mirage"
    steamid64 = str(profile.steam_id)
    target_id = normalize_steamid64(profile.steam_id)
    demos_dir = Path(get_demos_dir(profile, map_name))
    demo_paths = sorted(demos_dir.glob("*.dem"))

    if not demo_paths:
        print(f"No demos found in {demos_dir}")
        return

    total_kills = 0
    total_target_kills = 0
    total_missing_time = 0

    print(f"Profile id={profile.id} steamid={steamid64} demos={len(demo_paths)} map={map_name}")
    for demo_path in demo_paths:
        parsed = parse_demo_events(demo_path, target_steam_id=steamid64)
        demo_kills = len(parsed.kills)
        target_kills = sum(
            1 for kill in parsed.kills if kill.get("attacker_steamid64") == target_id
        )
        total_kills += demo_kills
        total_target_kills += target_kills
        total_missing_time += parsed.missing_time_kills
        print(
            f"{demo_path.name}: kills={demo_kills} target_kills={target_kills} "
            f"tick_rate={parsed.tick_rate} missing_time={parsed.missing_time_kills}"
        )

    media_root = Path("media")
    out_dir = media_root / "heatmaps_local" / steamid64 / "aggregate" / map_name
    cache_root = media_root / "heatmaps_cache"
    analyzer.build_heatmaps_aggregate(
        steamid64=steamid64,
        map_name=map_name,
        limit=len(demo_paths),
        demos_dir=demos_dir,
        out_dir=out_dir,
        cache_dir=cache_root,
        force=True,
    )

    cache_dir = cache_root / steamid64 / map_name
    npz_paths = sorted(cache_dir.glob("*.npz"))
    required_keys = {"kills_pxt", "deaths_pxt", "presence_all_pxt", "presence_ct_pxt", "presence_t_pxt"}
    union_keys: set[str] = set()
    npz_with_any_pxt = 0
    for path in npz_paths:
        with np.load(path) as cached:
            union_keys.update(cached.files)
            if any(key in cached.files for key in required_keys):
                npz_with_any_pxt += 1

    print(f"npz_count={len(npz_paths)} npz_with_any_pxt={npz_with_any_pxt}")
    print(f"pxt_keys_present={sorted(required_keys & union_keys)}")


if __name__ == "__main__":
    main()
