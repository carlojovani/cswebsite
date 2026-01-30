import json
from collections import Counter, defaultdict
from pathlib import Path

from users.models import PlayerProfile
from faceit_analytics.models import AnalyticsAggregate
from faceit_analytics.services.paths import get_demos_dir
from faceit_analytics.services.demo_events import parse_demo_events

PROFILE_ID = 2
MAP_NAME = "de_mirage"
PERIOD = "last_20"

def j(v, n=380):
    try:
        return json.dumps(v, ensure_ascii=False)[:n]
    except Exception:
        return str(v)[:n]

def get_nested(d, path):
    cur = d
    for k in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
        if cur is None:
            return None
    return cur

print("=== VALIDATE AFTER PIPELINE ===")
p = PlayerProfile.objects.get(id=PROFILE_ID)
steam = str(p.steam_id)
print("profile_id:", p.id)
print("steam_id:", steam)
print("map:", MAP_NAME, "period:", PERIOD)

agg = AnalyticsAggregate.objects.filter(profile=p).order_by("-updated_at", "-id").first()
print("\n=== DB: AnalyticsAggregate (latest) ===")
if not agg:
    print("NO AnalyticsAggregate row!")
    raise SystemExit(1)

mj = agg.metrics_json or {}
print("agg_id:", agg.id)
print("period:", getattr(agg, "period", None))
print("analytics_version:", getattr(agg, "analytics_version", None))
print("updated_at:", getattr(agg, "updated_at", None))
print("top keys:", sorted(list(mj.keys()))[:40])

# Печатаем “подозрительные” блоки, которые у тебя на UI нули/—/странные
print("\n=== DB: suspicious blocks ===")
for keypath in [
    "multikill",
    "playstyle.entry_breakdown",
    "kill_support",
    "timing_slices",
    "utility_iq",
]:
    val = get_nested(mj, keypath)
    print(f"{keypath} =", j(val))

# Парсим демки и считаем базовые факты (киллы/эйсы/раунды)
demos_dir = Path(get_demos_dir(p, MAP_NAME))
demo_paths = sorted(demos_dir.glob("*.dem"))
print("\n=== DEMOS ===")
print("dir:", demos_dir)
print("count:", len(demo_paths))
if not demo_paths:
    raise SystemExit("No demos found")

total_target_kills = 0
total_kills_events = 0
ace_rounds_global = set()
round_kills_global = Counter()
attacker_counts = Counter()

for dem in demo_paths:
    ev = parse_demo_events(dem, target_steam_id=steam)
    kills = ev.kills or []
    total_kills_events += len(kills)

    # attacker_steamid64 у тебя в событиях как int/str — приводим к str
    for k in kills:
        a = k.get("attacker_steamid64")
        if a is not None:
            attacker_counts[str(a)] += 1

    target_kills = [k for k in kills if str(k.get("attacker_steamid64")) == steam]
    total_target_kills += len(target_kills)

    # ace rounds: в рамках одной демки считаем раунды где target сделал >=5 киллов
    by_round = Counter()
    for k in target_kills:
        r = k.get("round")
        if r is None:
            continue
        by_round[int(r)] += 1

    for r, c in by_round.items():
        round_kills_global[r] += c
        if c >= 5:
            ace_rounds_global.add((dem.name, r))

print("\n=== DEMO FACTS ===")
print("kill_events_total:", total_kills_events)
print("target_kills_total:", total_target_kills)
print("top attackers:", attacker_counts.most_common(10))
print("ace_rounds (demo, round) count:", len(ace_rounds_global))
if ace_rounds_global:
    print("ace_rounds sample:", list(sorted(ace_rounds_global))[:12])

# Сравнение с БД (если там есть)
db_aces = get_nested(mj, "multikill.aces")
db_mk = get_nested(mj, "multikill.phases")
print("\n=== COMPARE DB vs DEMO ===")
print("DB multikill.aces:", db_aces, "| DEMO ace_rounds_count:", len(ace_rounds_global))
print("DB multikill.phases:", j(db_mk, 260))

# Entry breakdown buckets (последовательные 0-15 0-30 ...)
solo_by = get_nested(mj, "playstyle.entry_breakdown.solo_by_bucket_pct")
assist_by = get_nested(mj, "playstyle.entry_breakdown.assisted_by_bucket_pct")
print("\n=== ENTRY BUCKETS IN DB ===")
print("solo_by_bucket_pct:", j(solo_by))
print("assisted_by_bucket_pct:", j(assist_by))

print("\nDONE")
