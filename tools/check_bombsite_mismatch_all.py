from pathlib import Path
from users.models import PlayerProfile
from faceit_analytics.services.paths import get_demos_dir
from awpy import Demo

PROFILE_ID = int(__import__("os").environ.get("PROFILE_ID", "2"))
MAP_NAME = __import__("os").environ.get("MAP_NAME", "de_mirage")

p = PlayerProfile.objects.get(id=PROFILE_ID)
steam = int(p.steam_id)

demos_dir = Path(get_demos_dir(p, MAP_NAME))
demos = sorted(demos_dir.glob("*.dem"))

print("PROFILE:", p.id, "steam:", steam)
print("DIR:", demos_dir, "count:", len(demos))
print()

def norm_round_bomb_site(x):
    if x is None:
        return None
    s = str(x).lower()
    if "a" in s and "bombsite" in s:
        return "A"
    if "b" in s and "bombsite" in s:
        return "B"
    if s.endswith("_a"):
        return "A"
    if s.endswith("_b"):
        return "B"
    return s

def norm_event_bombsite(x):
    if x is None:
        return None
    s = str(x).lower()
    if "a" in s:
        return "A"
    if "b" in s:
        return "B"
    return s

total_mismatch = 0
total_plants = 0

for dem_path in demos:
    dem = Demo(str(dem_path), verbose=False)
    dem.parse()

    rounds = dem.rounds.to_pandas()
    bomb = dem.bomb.to_pandas() if getattr(dem, "bomb", None) is not None else None
    if bomb is None:
        print(dem_path.name, "NO bomb table")
        continue

    plants = bomb[bomb["event"].astype(str).str.contains("plant", case=False, na=False)].copy()
    if plants.empty:
        print(dem_path.name, "plants=0")
        continue

    total_plants += len(plants)

    # map: round_num -> bombsite from bomb events (use first plant per round)
    plant_site_by_round = {}
    for _, r in plants.iterrows():
        rn = int(r.get("round_num"))
        if rn not in plant_site_by_round:
            plant_site_by_round[rn] = norm_event_bombsite(r.get("bombsite"))

    mism = []
    for rn, event_site in sorted(plant_site_by_round.items()):
        rr = rounds[rounds["round_num"] == rn]
        if rr.empty:
            continue
        round_site = norm_round_bomb_site(rr.iloc[0].get("bomb_site"))
        if event_site is None or round_site is None:
            continue
        if str(event_site) != str(round_site):
            mism.append((rn, event_site, round_site))

    if mism:
        total_mismatch += len(mism)
        print("==", dem_path.name, "MISMATCHES:", len(mism), "==")
        for rn, es, rs in mism[:20]:
            print(f"  round {rn}: bomb.event={es} vs rounds.bomb_site={rs}")
        if len(mism) > 20:
            print("  ...")
    else:
        print(dem_path.name, "OK (plants:", len(plants), ")")

print()
print("TOTAL plants:", total_plants, "TOTAL mismatches:", total_mismatch)
if total_mismatch:
    print("CONCLUSION: do NOT trust dem.rounds.bomb_site. Use dem.bomb plant rows (bombsite + coords) as source of truth.")
