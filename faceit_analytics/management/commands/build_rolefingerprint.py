import json

from django.core.management.base import BaseCommand, CommandError

from faceit_analytics.constants import ANALYTICS_VERSION
from faceit_analytics.services.aggregates import build_metrics, enrich_metrics_with_role_features
from users.models import PlayerProfile


class Command(BaseCommand):
    help = "Build role fingerprint analytics for a profile and print metrics_json."

    def add_arguments(self, parser):
        parser.add_argument("--profile", type=int, required=True, help="PlayerProfile ID")
        parser.add_argument("--period", type=str, default="last_20", help="Analytics period (default: last_20)")

    def handle(self, *args, **options):
        profile_id = options["profile"]
        period = options["period"]

        try:
            profile = PlayerProfile.objects.get(id=profile_id)
        except PlayerProfile.DoesNotExist as exc:
            raise CommandError(f"PlayerProfile {profile_id} not found") from exc

        aggregates = build_metrics(profile, period=period, analytics_version=ANALYTICS_VERSION)
        if not aggregates:
            raise CommandError("No aggregates built.")

        aggregate = enrich_metrics_with_role_features(aggregates[0], profile, period)
        self.stdout.write(json.dumps(aggregate.metrics_json, ensure_ascii=False, indent=2))
