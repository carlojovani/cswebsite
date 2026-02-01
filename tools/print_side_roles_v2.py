import argparse
import os
import sys

import django


def main() -> None:
    parser = argparse.ArgumentParser(description="Print side_roles_v2 summary for a profile.")
    parser.add_argument("--profile-id", type=int, required=True)
    parser.add_argument("--period", default="last_20")
    parser.add_argument("--map-name", default="de_mirage")
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(__file__))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backend.settings")
    django.setup()

    from faceit_analytics.models import AnalyticsAggregate

    aggregate = (
        AnalyticsAggregate.objects.filter(
            profile_id=args.profile_id,
            period=args.period,
            map_name=args.map_name,
        )
        .order_by("-updated_at")
        .first()
    )
    if not aggregate or not aggregate.metrics_json:
        raise SystemExit("No analytics aggregate found.")

    side_roles_v2 = (aggregate.metrics_json or {}).get("side_roles_v2") or {}
    ct = side_roles_v2.get("ct") or {}
    t = side_roles_v2.get("t") or {}
    print("CT:", ct.get("label"), "confidence=", ct.get("confidence"), "shares=", ct.get("position_shares"))
    print("T:", t.get("label"), "confidence=", t.get("confidence"), "shares=", t.get("role_shares"))


if __name__ == "__main__":
    main()
