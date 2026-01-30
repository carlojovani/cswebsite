"""
Quick check: does profile.steam_id appear in kill events (after parsing) or is it off by float rounding?

Run:
  python manage.py shell -c "exec(open('tools/debug_demo_steamid_precision.py','r',encoding='utf-8').read())"

Env:
  PROFILE_ID=2
  MAP_NAME=de_mirage
"""
from __future__ import annotations
import os
from pathlib import Path

from users.models import PlayerProfile
from faceit_analytics.services.paths import get_demos_dir
from faceit_analytics.services.demo_events import parse_demo_events

profile_id = int(os.environ.get("PROFILE_ID", "2"))
map_name = os.environ.get("MAP_NAME", "de_mirage")

p = PlayerProfile.objects.filter(id=profile_id).first()
if not p:
    raise SystemExit(f"PlayerProfile id={profile_id} not found")
steam = str(p.steam_id)

demo_dir = Path(get_demos_dir(p, map_name))
demos = sorted(demo_dir.glob("*.dem"))
if not demos:
    raise SystemExit("No demos found")
demo = demos[-1]

print("profile_id:", p.id)
print("steam_id:", steam)
print("demo:", demo.name)

ev = parse_demo_events(demo, target_steam_id=steam)
kills = ev.kills or []
ids = sorted({str(k.get("attacker_steamid64")) for k in kills if k.get("attacker_steamid64") is not None})
print("unique attacker ids:", len(ids))
print("steam_id in attacker ids:", steam in ids)

# show nearest ids by numeric distance
try:
    steam_i = int(steam)
    nums = []
    for s in ids:
        try:
            nums.append(int(s))
        except Exception:
            pass
    nums_sorted = sorted(nums, key=lambda x: abs(x - steam_i))[:5]
    print("closest ids:", nums_sorted)
    if nums_sorted:
        print("closest deltas:", [n - steam_i for n in nums_sorted])
except Exception as e:
    print("cannot compute deltas:", e)

# count target kills
target_kills = [k for k in kills if str(k.get("attacker_steamid64")) == steam]
print("target_kills:", len(target_kills), "of total kills:", len(kills))
