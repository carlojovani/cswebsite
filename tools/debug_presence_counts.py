import os
from pathlib import Path

from awpy import Demo
from django.conf import settings
from users.models import PlayerProfile

from faceit_analytics import analyzer
from faceit_analytics.services import demo_events

PROFILE_ID = int(os.environ.get("PROFILE_ID", "2"))
MAP_NAME = os.environ.get("MAP_NAME", "de_mirage")
DEMO_NAME = os.environ.get("DEMO_NAME", "match5.dem")

profile = PlayerProfile.objects.get(id=PROFILE_ID)
steam = str(profile.steam_id)

media = Path(settings.MEDIA_ROOT)
demos_dir = media / "local_demos" / steam / MAP_NAME
dem_path = demos_dir / DEMO_NAME

print("Demo:", dem_path, "exists:", dem_path.exists())

dem = Demo(str(dem_path), verbose=False)
dem.parse()

rounds_df = dem.rounds.to_pandas() if getattr(dem, "rounds", None) is not None else None
round_start_ticks, round_start_times, _round_winners, _rounds = demo_events._build_round_meta(rounds_df)
tick_rate = demo_events._tick_rate_from_demo(dem)

ticks_df = demo_events._load_demo_dataframe(
    getattr(dem, "ticks", None),
    ["steamid", "player_steamid", "playerSteamID"],
)
if ticks_df is None or ticks_df.empty:
    print("No ticks data available.")
    raise SystemExit(0)

sid_col = demo_events._pick_column(ticks_df, ["steamid", "steamID", "player_steamid", "playerSteamID"])
round_col = demo_events._pick_column(ticks_df, ["round", "round_num", "round_number"])

ticks_my = analyzer._filter_by_steamid_numeric(ticks_df, sid_col, steam)
tick_total = int(ticks_my.shape[0])
tick_with_time = 0
for _, row in ticks_my.iterrows():
    round_number = demo_events._safe_int(row.get(round_col)) if round_col else None
    t_round = demo_events._round_time_seconds(row, round_number, round_start_ticks, round_start_times, tick_rate)
    if t_round is not None:
        tick_with_time += 1

radar, meta, radar_name = analyzer.load_radar_and_meta(MAP_NAME)
radar_size = radar.size
map_mask_L = analyzer.build_map_mask(radar)
points, debug = analyzer._extract_points_from_demo(dem_path, steam, meta, radar_size, map_mask_L, dem=dem)
presence_points = int(points["presence_all_px"].shape[0])

print("Tick rows for steamid:", tick_total)
print("Tick rows with t_round:", tick_with_time)
print("Presence points:", presence_points)
print("Missing t_round:", debug.get("ticks_missing_t_round"))
