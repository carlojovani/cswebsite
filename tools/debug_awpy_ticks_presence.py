import os
from pathlib import Path
from django.conf import settings
from users.models import PlayerProfile
from awpy import Demo

PROFILE_ID = int(os.environ.get("PROFILE_ID","2"))
MAP_NAME = os.environ.get("MAP_NAME","de_mirage")
DEMO_NAME = os.environ.get("DEMO_NAME","match5.dem")

profile = PlayerProfile.objects.get(id=PROFILE_ID)
steam = int(profile.steam_id)

media = Path(settings.MEDIA_ROOT)
dem_path = media / "local_demos" / str(profile.steam_id) / MAP_NAME / DEMO_NAME
print("demo:", dem_path, "exists:", dem_path.exists())

demo = Demo(str(dem_path), verbose=False)
demo.parse()

ticks = getattr(demo, "ticks", None)
print("demo.ticks is None:", ticks is None)
if ticks is not None:
    try:
        print("ticks type:", type(ticks))
        cols = list(getattr(ticks, "columns", []))
        print("ticks cols head:", cols[:25])
        print("ticks height:", getattr(ticks, "height", None))

        # steamid columns differ between awpy versions
        sid_col = None
        for c in ["steamid", "steamID", "steam_id", "steamid64", "steamID64"]:
            if c in cols:
                sid_col = c
                break
        print("steamid col:", sid_col)

        if sid_col:
            # count rows with target steam
            try:
                cnt = ticks.filter(ticks[sid_col] == steam).height
                print("rows with target steam:", cnt)
            except Exception as e:
                print("filter error:", e)
        else:
            print("No steamid column found in ticks")

        # also check null X/Y ratio
        for c in ["X","Y","x","y"]:
            if c in cols:
                try:
                    nulls = ticks.select(ticks[c].is_null().sum()).item()
                    print("null", c, ":", nulls)
                except Exception:
                    pass
    except Exception as e:
        print("ticks inspect error:", e)

print("DONE")