from django.utils import timezone

from faceit_analytics.models import AnalyticsAggregate

DEFAULT_MAP_NAME = "all"
DEFAULT_SIDE = AnalyticsAggregate.SIDE_ALL


def build_metrics(profile, period: str, analytics_version: str = "v1") -> list[AnalyticsAggregate]:
    metrics = {
        "win_rate": profile.win_rate,
        "average_kd": profile.average_kd,
        "average_hs": profile.average_hs,
        "elo": profile.elo,
        "matches": profile.matches,
        "wins": profile.wins,
        "current_win_streak": profile.current_win_streak,
        "longest_win_streak": profile.longest_win_streak,
        "source": "player_profile",
        "updated_at": timezone.now().isoformat(),
    }

    aggregate, _ = AnalyticsAggregate.objects.update_or_create(
        profile=profile,
        map_name=DEFAULT_MAP_NAME,
        side=DEFAULT_SIDE,
        period=period,
        analytics_version=analytics_version,
        defaults={"metrics_json": metrics},
    )

    return [aggregate]
