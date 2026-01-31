import os
from pathlib import Path
from django.conf import settings
from users.models import PlayerProfile
from faceit_analytics.services import demo_events
from faceit_analytics import analyzer

PROFILE_ID = int(os.environ.get("PROFILE_ID","2"))
MAP_NAME = os.environ.get("MAP_NAME","de_mirage")
DEMO_NAME = os.environ.get("DEMO_NAME","match5.dem")

profile = PlayerProfile.objects.get(id=PROFILE_ID)
steam = str(profile.steam_id)

media = Path(settings.MEDIA_ROOT)
demos_dir = media / "local_demos" / steam / MAP_NAME
dem_path = demos_dir / DEMO_NAME

demo = analyzer.load_demo_cached(dem_path)
parsed = demo_events.parse_demo_events(demo)

ticks = parsed.ticks_df
rounds = parsed.rounds_df

print("ticks type:", type(ticks))
print("rounds type:", type(rounds))

# show basic columns
print("ticks cols:", list(getattr(ticks, "columns", []))[:20])
print("rounds cols:", list(getattr(rounds, "columns", []))[:20])

# sample round numbers
try:
    t_round_col = "round_num" if "round_num" in ticks.columns else "round"
    r_round_col = "round_num" if "round_num" in rounds.columns else "round"
    print("ticks round col:", t_round_col, "rounds round col:", r_round_col)
    t_sample = ticks.select(t_round_col).head(20).to_series().to_list()
    r_sample = rounds.select(r_round_col).head(20).to_series().to_list()
    print("ticks round sample:", t_sample)
    print("rounds round sample:", r_sample)
except Exception as e:
    print("sample error:", e)

# show start tick/time keys used
print("round_start_ticks keys:", sorted(list(parsed.round_start_ticks.keys()))[:30], "...")
print("round_start_times keys:", sorted(list(parsed.round_start_times.keys()))[:30], "...")

# test _round_time_seconds on first 10 ticks
ok = 0
bad = 0
try:
    # pull first 10 tick rows
    df10 = ticks.head(10)
    # extract columns safely
    cols = df10.columns
    tick_col = "tick" if "tick" in cols else None
    xcol = "X" if "X" in cols else ("x" if "x" in cols else None)
    ycol = "Y" if "Y" in cols else ("y" if "y" in cols else None)
    rcol = "round_num" if "round_num" in cols else ("round" if "round" in cols else None)

    for i in range(10):
        row = df10.row(i)
        # map row->dict
        d = dict(zip(cols, row))
        rn = d.get(rcol)
        tk = d.get(tick_col)
        tr = demo_events._round_time_seconds(rn, tk, parsed.round_start_ticks, parsed.round_start_times, parsed.tick_rate)
        if tr is None:
            bad += 1
        else:
            ok += 1
        print("i", i, "rn", rn, "tick", tk, "t_round", tr)
except Exception as e:
    print("t_round test error:", e)

print("t_round ok:", ok, "bad:", bad)