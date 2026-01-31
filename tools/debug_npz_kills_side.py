import os
from pathlib import Path

import numpy as np
from django.conf import settings

from users.models import PlayerProfile
from faceit_analytics import analyzer

PROFILE_ID = int(os.environ.get("PROFILE_ID", "2"))
MAP_NAME = os.environ.get("MAP_NAME", "de_mirage")
DEMO_NAME = os.environ.get("DEMO_NAME", "match5.dem")

profile = PlayerProfile.objects.get(id=PROFILE_ID)
steam = str(profile.steam_id)

media = Path(settings.MEDIA_ROOT)
demos_dir = media / "local_demos" / steam / MAP_NAME
dem_path = demos_dir / DEMO_NAME

out_dir = media / "heatmaps_local" / steam / "aggregate" / MAP_NAME
cache_root = media / "heatmaps_cache"

print("Demo:", dem_path, "exists:", dem_path.exists())

res = analyzer.build_heatmaps_aggregate(
    steamid64=steam,
    map_name=MAP_NAME,
    limit=20,
    demos_dir=demos_dir,
    out_dir=out_dir,
    cache_dir=cache_root,
    force=True,
)
print("build_heatmaps_aggregate:", res)

radar, meta, radar_name = analyzer.load_radar_and_meta(MAP_NAME)
radar_size = radar.size

cache_dir = cache_root / steam / MAP_NAME
cache_name = analyzer._demo_cache_hash(dem_path, radar_name, radar_size)
npz_path = cache_dir / f"{cache_name}.npz"
print("NPZ:", npz_path, "exists:", npz_path.exists())

z = np.load(npz_path)
print("NPZ keys:", sorted(list(z.files)))


def dump(metric_key: str) -> None:
    if metric_key not in z.files:
        print(f"\n{metric_key}: MISSING")
        return
    arr = z[metric_key]
    print(f"\n{metric_key}: shape={arr.shape} dtype={arr.dtype}")
    if arr.size == 0 or arr.shape[0] == 0:
        return
    head = arr[:5]
    print("  head(5):", head.tolist())


def dump_group(label: str, px_key: str, pxt_key: str) -> None:
    print(f"\n== {label} ==")
    dump(px_key)
    dump(pxt_key)


for label, px_key, pxt_key in [
    ("Kills (all)", "kills_px", "kills_pxt"),
    ("Kills (CT)", "kills_ct_px", "kills_ct_pxt"),
    ("Kills (T)", "kills_t_px", "kills_t_pxt"),
    ("Deaths (all)", "deaths_px", "deaths_pxt"),
    ("Deaths (CT)", "deaths_ct_px", "deaths_ct_pxt"),
    ("Deaths (T)", "deaths_t_px", "deaths_t_pxt"),
]:
    dump_group(label, px_key, pxt_key)

z.close()
print("\nDONE")
