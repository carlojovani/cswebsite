# tools/check_round_collisions.py
from collections import Counter, defaultdict
from pathlib import Path
import json

from users.models import PlayerProfile
from faceit_analytics.services.paths import get_demos_dir
from faceit_analytics.services.demo_events import parse_demo_events

PROFILE_ID = 2
MAP_NAME = "de_mirage"

def main():
    p = PlayerProfile.objects.get(id=PROFILE_ID)
    steam = str(p.steam_id)

    demos_dir = Path(get_demos_dir(p, MAP_NAME))
    demos = sorted(demos_dir.glob("*.dem"))

    print("profile_id:", p.id)
    print("steam_id:", steam)
    print("demos:", len(demos), "dir:", demos_dir)

    # ВАЖНО:
    # correct_key = (demo_name, round)
    # bad_key = (round)  <-- если код так делает, появятся "эйсы" из суммы разных демок
    per_demo_round = Counter()
    per_round_aggregated = Counter()

    # посчитаем киллы игрока
    total_kills = 0

    for dem in demos:
        ev = parse_demo_events(dem, target_steam_id=steam)
        kills = ev.kills or []
        target = [k for k in kills if str(k.get("attacker_steamid64")) == steam]

        total_kills += len(target)

        by_round = Counter()
        for k in target:
            r = k.get("round")
            if r is None:
                continue
            by_round[int(r)] += 1

        # сохраняем обе агрегации
        for r, c in by_round.items():
            per_demo_round[(dem.name, r)] += c
            per_round_aggregated[r] += c

    print("\n=== TARGET TOTAL KILLS ===")
    print(total_kills)

    print("\n=== BAD AGGREGATION (round only) top ===")
    top_bad = per_round_aggregated.most_common(15)
    print(top_bad)

    fake_aces = [(r, c) for r, c in per_round_aggregated.items() if c >= 5]
    print("\nFake 'aces' if you group by ROUND ONLY (c>=5):", len(fake_aces))
    print("sample:", fake_aces[:20])

    real_aces = [(k, v) for k, v in per_demo_round.items() if v >= 5]
    print("\nReal aces by (demo, round) (v>=5):", len(real_aces))
    print("sample:", real_aces[:20])

    if fake_aces and not real_aces:
        print("\nDIAGNOSIS:")
        print("* If your pipeline groups by round number only across demos -> it will invent 'aces' and inflate multikills.")
        print("* Fix: every per-round metric must key by (demo_id, round) not just round.")

if __name__ == '__main__':
    main()
