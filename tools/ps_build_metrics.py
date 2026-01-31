# tools/ps_build_metrics.py
import json
from collections import Counter, defaultdict
from pathlib import Path

from users.models import PlayerProfile
from faceit_analytics.models import AnalyticsAggregate
from faceit_analytics.services.paths import get_demos_dir
from faceit_analytics.services.demo_events import parse_demo_events


PROFILE_ID = 2
MAP_NAME = "de_mirage"
PERIOD = "last_20"  # только для сравнения с DB


def _dump(x, n=600):
    return json.dumps(x, ensure_ascii=False)[:n]


def main():
    p = PlayerProfile.objects.get(id=PROFILE_ID)
    steam = str(p.steam_id)

    print("=== PROFILE ===")
    print("profile_id:", p.id)
    print("steam_id:", steam)
    print("map:", MAP_NAME)

    # DB snapshot
    agg = AnalyticsAggregate.objects.filter(profile=p).order_by("-id").first()
    print("\n=== DB ===")
    if not agg:
        print("NO AnalyticsAggregate for profile")
        db = {}
    else:
        db = agg.metrics_json or {}
        print("agg_id:", agg.id)
        print("period:", getattr(agg, "period", None), "version:", getattr(agg, "analytics_version", None))
        print("updated_at:", getattr(agg, "updated_at", None))
        print("db.multikill.aces:", (db.get("multikill") or {}).get("aces"))
        print("db.kill_support:", _dump(db.get("kill_support")))
        print("db.playstyle.entry_breakdown:", _dump((db.get("playstyle") or {}).get("entry_breakdown")))

    # demos
    demos_dir = Path(get_demos_dir(p, MAP_NAME))
    demo_paths = sorted(demos_dir.glob("*.dem"))
    print("\n=== DEMOS ===")
    print("dir:", demos_dir)
    print("count:", len(demo_paths))
    if not demo_paths:
        print("NO DEMOS FOUND")
        return

    # accumulate
    total_kills = 0
    total_deaths = 0
    kill_events_total = 0
    attacker_counter = Counter()
    target_kills_by_round = defaultdict(int)
    target_kills_by_demo_round = []  # (demo, round, count)
    ace_rounds = []  # (demo, round, kills)
    multikill_rounds = []  # (demo, round, kills>=2)
    parse_errors = 0

    for dem in demo_paths:
        try:
            ev = parse_demo_events(dem, target_steam_id=steam)
        except Exception as e:
            parse_errors += 1
            print(f"\n--- {dem.name} ---")
            print("PARSE ERROR:", repr(e))
            continue

        kills = ev.kills or []
        deaths = [k for k in kills if str(k.get("victim_steamid64")) == steam]
        target_kills = [k for k in kills if str(k.get("attacker_steamid64")) == steam]

        kill_events_total += len(kills)
        total_kills += len(target_kills)
        total_deaths += len(deaths)

        for k in kills:
            a = k.get("attacker_steamid64")
            if a is not None:
                attacker_counter[str(a)] += 1

        # per-round target kills
        per_round = Counter()
        for k in target_kills:
            r = k.get("round")
            if r is None:
                continue
            per_round[int(r)] += 1

        for r, c in sorted(per_round.items()):
            target_kills_by_round[r] += c
            target_kills_by_demo_round.append((dem.name, r, c))
            if c >= 5:
                ace_rounds.append((dem.name, r, c))
            if c >= 2:
                multikill_rounds.append((dem.name, r, c))

        print(f"\n--- {dem.name} ---")
        print("tick_rate:", ev.tick_rate, "approx:", ev.tick_rate_approx)
        print("kills_total_events:", len(kills))
        print("target_kills:", len(target_kills), "target_deaths:", len(deaths))
        print("rounds_in_demo:", len(ev.rounds_in_demo), "missing_time_kills:", ev.missing_time_kills)

    print("\n=== SUMMARY (from DEMOS) ===")
    print("parse_errors:", parse_errors, "/", len(demo_paths))
    print("kill_events_total:", kill_events_total)
    print("target_total_kills:", total_kills)
    print("target_total_deaths:", total_deaths)
    print("rounds_with_target_kills:", len(target_kills_by_round))
    print("multikill_rounds_count(k>=2):", len(multikill_rounds))
    print("ace_rounds_count(k>=5):", len(ace_rounds))

    if ace_rounds:
        print("\nACE ROUNDS:")
        for dem, r, c in ace_rounds[:20]:
            print(f"{dem} round={r} kills={c}")

    print("\nTop attackers overall:")
    print(attacker_counter.most_common(12))

    # Compare with DB if exists
    if agg:
        db_aces = (db.get("multikill") or {}).get("aces")
        print("\n=== COMPARE DB vs DEMO ===")
        print("DB multikill.aces:", db_aces, "| DEMO ace_rounds_count:", len(ace_rounds))

        # Support sanity (just check if unknown dominates)
        ks = db.get("kill_support") or {}
        cat = (ks.get("categories") or {})
        unknown = cat.get("unknown")
        if unknown is not None:
            print("DB kill_support unknown:", unknown, "/", ks.get("total_kills"))

    print("\nDONE")


if __name__ == '__main__':
    main()
