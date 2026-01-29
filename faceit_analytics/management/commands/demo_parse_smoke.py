from __future__ import annotations

import json

from django.core.management.base import BaseCommand, CommandError

from faceit_analytics.constants import ANALYTICS_VERSION
from faceit_analytics.services.demo_events import get_or_build_demo_features, safe_json
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
        timing_slices = payload.get("timing_slices")
        if timing_slices is None:
            raise CommandError("timing_slices is missing from demo features payload")
        approx_expected = bool(
            debug.get("demos_count") == 0
            or debug.get("rounds_count") == 0
            or debug.get("tickrate_assumed")
            or debug.get("approx_time_kills")
            or debug.get("missing_time_kills")
        )
        approx_actual = bool(timing_slices.get("approx"))
        if approx_expected != approx_actual:
            self.stderr.write(
                "Warning: timing_slices approx flag does not match expected "
                f"(expected={approx_expected}, actual={approx_actual})"
            )
        output = {
            "profile_id": profile_id,
            "period": period,
            "demos_count": payload.get("demos_count"),
            "rounds_count": debug.get("rounds_count"),
            "kills_events_count": debug.get("kills_events_count"),
            "flash_events_count": debug.get("flash_events_count"),
            "util_damage_events_count": debug.get("util_damage_events_count"),
            "missing_time_kills": debug.get("missing_time_kills"),
            "target_kills": debug.get("player_kills"),
            "target_deaths": debug.get("player_deaths"),
            "target_assists": debug.get("player_assists"),
            "target_name": debug.get("target_name"),
            "debug": debug,
            "role_fingerprint": payload.get("role_fingerprint"),
            "utility_iq": payload.get("utility_iq"),
            "timing_slices": timing_slices,
            "sample_event": debug.get("kill_event_sample"),
        }
        self.stdout.write(json.dumps(safe_json(output), ensure_ascii=False, indent=2))
