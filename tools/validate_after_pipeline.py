from __future__ import annotations

import sys
from pathlib import Path

from faceit_analytics.models import AnalyticsAggregate
from faceit_analytics.services.demo_events import discover_demo_files, normalize_steamid64, parse_demo_events
from users.models import PlayerProfile


def _count_ace_rounds(kills_by_round: dict[int, int]) -> int:
    return sum(1 for count in kills_by_round.values() if count >= 5)


def main() -> None:
    profile = PlayerProfile.objects.order_by("-id").first()
    if not profile:
        print("No PlayerProfile found.")
        return

    map_name = "de_mirage"
    period = "last_20"
    steamid64 = str(profile.steam_id)
    target_id = normalize_steamid64(profile.steam_id)
    if target_id is None:
        print("Invalid steamid on profile.")
        sys.exit(1)

    demo_paths = discover_demo_files(profile, period=period, map_name=map_name)
    if not demo_paths:
        print("No demos found for validation.")
        return

    demo_kills = 0
    kills_by_round: dict[int, int] = {}
    for demo_path in demo_paths:
        parsed = parse_demo_events(Path(demo_path), target_steam_id=steamid64)
        for kill in parsed.kills:
            if kill.get("attacker_steamid64") != target_id:
                continue
            demo_kills += 1
            round_number = kill.get("round")
            if round_number is not None:
                kills_by_round[round_number] = kills_by_round.get(round_number, 0) + 1

    demo_ace_count = _count_ace_rounds(kills_by_round)

    aggregate = (
        AnalyticsAggregate.objects.filter(profile=profile, map_name=map_name, period=period)
        .order_by("-id")
        .first()
    )
    if not aggregate:
        print("No AnalyticsAggregate found for validation.")
        return

    metrics = aggregate.metrics_json or {}
    db_kills = (metrics.get("kda") or {}).get("kills")
    db_ace = (metrics.get("multikill") or {}).get("ace_rounds")

    errors = []
    if db_kills != demo_kills:
        errors.append(f"total_kills mismatch: db={db_kills} demo={demo_kills}")
    if db_ace != demo_ace_count:
        errors.append(f"ace_count mismatch: db={db_ace} demo={demo_ace_count}")

    proximity_cov = (metrics.get("demo_features_debug") or {}).get("proximity_coverage_pct")
    entry_breakdown = metrics.get("entry_breakdown") or {}
    assisted_by_bucket_pct = entry_breakdown.get("assisted_by_bucket_pct") or {}
    solo_by_bucket_pct = entry_breakdown.get("solo_by_bucket_pct") or {}
    all_bucket_values = list(assisted_by_bucket_pct.values()) + list(solo_by_bucket_pct.values())
    all_buckets_null = bool(all_bucket_values) and all(val is None for val in all_bucket_values)

    if proximity_cov is not None and proximity_cov > 50 and all_buckets_null:
        errors.append("entry buckets are all null despite high proximity coverage")

    print(f"db_kills={db_kills} demo_kills={demo_kills}")
    print(f"db_ace={db_ace} demo_ace={demo_ace_count}")
    print(f"proximity_coverage_pct={proximity_cov}")

    if errors:
        print("Validation errors:")
        for err in errors:
            print(f"- {err}")
        sys.exit(1)

    print("Validation ok.")


if __name__ == "__main__":
    main()
