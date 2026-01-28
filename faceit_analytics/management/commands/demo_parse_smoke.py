from __future__ import annotations

import json

from django.core.management.base import BaseCommand

from faceit_analytics.constants import ANALYTICS_VERSION
from faceit_analytics.services.demo_events import get_or_build_demo_features
from users.models import PlayerProfile


class Command(BaseCommand):
    help = "Parse local CS2 demos and print debug counts + metrics for a profile."

    def add_arguments(self, parser):
        parser.add_argument("--profile-id", type=int, required=True)
        parser.add_argument("--period", type=str, default="last_20")
        parser.add_argument("--force", action="store_true")

    def handle(self, *args, **options):
        profile_id = options["profile_id"]
        period = options["period"]
        force = options["force"]

        profile = PlayerProfile.objects.get(id=profile_id)
        payload = get_or_build_demo_features(
            profile,
            period=period,
            analytics_version=ANALYTICS_VERSION,
            force_rebuild=force,
        )
        debug = payload.get("debug") or {}
        debug["steamid64_eq_counts"] = {
            "kills": debug.get("player_kills", 0),
            "deaths": debug.get("player_deaths", 0),
            "assists": debug.get("player_assists", 0),
        }

        output = {
            "profile_id": profile_id,
            "period": period,
            "debug": debug,
            "role_fingerprint": payload.get("role_fingerprint"),
            "utility_iq": payload.get("utility_iq"),
            "timing_slices": payload.get("timing_slices"),
        }
        self.stdout.write(json.dumps(output, ensure_ascii=False, indent=2))
