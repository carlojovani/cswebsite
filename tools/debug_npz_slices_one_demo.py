import os, numpy as np
from pathlib import Path
from django.conf import settings
from users.models import PlayerProfile
from faceit_analytics import analyzer

PROFILE_ID = int(os.environ.get("PROFILE_ID","2"))
MAP_NAME = os.environ.get("MAP_NAME","de_mirage")
DEMO_NAME = os.environ.get("DEMO_NAME","match5.dem")

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

def dump(metric_key):
    if metric_key not in z.files:
        print("\n{}: MISSING".format(metric_key))
        return
    arr = z[metric_key]
    print("\n{}: shape={} dtype={}".format(metric_key, arr.shape, arr.dtype))
    if arr.size == 0 or arr.shape[0] == 0:
        return
    head = arr[:5]
    print("  head(5):", head.tolist())

    # only pxt has time col
    if arr.shape[1] >= 3:
        t = arr[:,2]
        t_valid = t[~np.isnan(t)]
        if t_valid.size:
            print("  t_round_min/max:", float(np.min(t_valid)), float(np.max(t_valid)))
        else:
            print("  t_round_min/max: all NaN")
        for label, tmax in [("0-15",15),("0-30",30),("0-45",45),("0-60",60),("0-75",75),("0-90",90)]:
            cnt = int((t <= tmax).sum())
            print("  {}: {}".format(label, cnt))
        print("  0+: {}".format(arr.shape[0]))

for k in [
    "presence_all_px","presence_all_pxt",
    "presence_ct_px","presence_ct_pxt",
    "presence_t_px","presence_t_pxt",
    "kills_px","kills_pxt",
    "deaths_px","deaths_pxt",
]:
    dump(k)

z.close()
print("\nDONE")