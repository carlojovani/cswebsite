import os
from pathlib import Path
from django.conf import settings
from users.models import PlayerProfile

from awpy import Demo
from faceit_analytics.services import demo_events

PROFILE_ID = int(os.environ.get("PROFILE_ID","2"))
MAP_NAME = os.environ.get("MAP_NAME","de_mirage")
DEMO_NAME = os.environ.get("DEMO_NAME","match5.dem")

profile = PlayerProfile.objects.get(id=PROFILE_ID)
steam = str(profile.steam_id)

media = Path(settings.MEDIA_ROOT)
demos_dir = media / "local_demos" / steam / MAP_NAME
dem_path = demos_dir / DEMO_NAME

print("Demo:", dem_path, "exists:", dem_path.exists())

# load via awpy directly
demo = Demo(str(dem_path))
demo.parse()

parsed = demo_events.parse_demo_events(demo)

ticks = parsed.ticks_df
rounds = parsed.rounds_df

print("tick_rate:", parsed.tick_rate)

# show columns
try:
    print("ticks columns (sample):", list(ticks.columns)[:25])
    print("rounds columns (sample):", list(rounds.columns)[:25])
except Exception as e:
    print("columns error:", e)

# detect round columns
t_round_col = "round_num" if "round_num" in ticks.columns else ("round" if "round" in ticks.columns else None)
r_round_col = "round_num" if "round_num" in rounds.columns else ("round" if "round" in rounds.columns else None)
print("ticks round col:", t_round_col, "rounds round col:", r_round_col)

# show samples
try:
    t_sample = ticks.select(t_round_col).head(20).to_series().to_list()
    r_sample = rounds.select(r_round_col).head(20).to_series().to_list()
    print("ticks round sample:", t_sample)
    print("rounds round sample:", r_sample)
except Exception as e:
    print("sample error:", e)

print("round_start_ticks keys head:", sorted(list(parsed.round_start_ticks.keys()))[:20])
print("round_start_times keys head:", sorted(list(parsed.round_start_times.keys()))[:20])

# test _round_time_seconds on first 10 tick rows
ok = 0
bad = 0

cols = ticks.columns
tick_col = "tick" if "tick" in cols else None
rcol = t_round_col

df10 = ticks.head(10)
for i in range(10):
    row = df10.row(i)
    d = dict(zip(cols, row))
    rn = d.get(rcol)
    tk = d.get(tick_col)
    tr = demo_events._round_time_seconds(rn, tk, parsed.round_start_ticks, parsed.round_start_times, parsed.tick_rate)
    if tr is None:
        bad += 1
    else:
        ok += 1
    print("i", i, "| rn", rn, "| tick", tk, "| t_round", tr)

print("t_round ok:", ok, "bad:", bad)

# also test rn-1 / rn+1 existence
if t_round_col is not None:
    try:
        rn0 = df10.select(t_round_col).row(0)[0]
        print("rn0:", rn0, "exists in start_ticks:", rn0 in parsed.round_start_ticks,
              "rn0-1:", (rn0-1) in parsed.round_start_ticks if isinstance(rn0,int) else None,
              "rn0+1:", (rn0+1) in parsed.round_start_ticks if isinstance(rn0,int) else None)
    except Exception as e:
        print("rn0 test error:", e)

print("DONE")