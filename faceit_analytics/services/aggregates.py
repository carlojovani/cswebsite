from django.utils import timezone

from faceit_analytics.models import AnalyticsAggregate
from faceit_analytics.services.adapters import adapt_feature_inputs
from faceit_analytics.services.features import (
    compute_role_fingerprint,
    compute_timing_slices,
    compute_utility_iq,
)

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


def enrich_metrics_with_role_features(
    aggregate: AnalyticsAggregate,
    profile,
    period: str,
) -> AnalyticsAggregate:
    events, positions, meta = adapt_feature_inputs(profile, period)
    timing_slices = compute_timing_slices(events, meta)
    meta = {**meta, "timing_slices": timing_slices}
    role_fingerprint = compute_role_fingerprint(events, positions, meta)
    utility_iq = compute_utility_iq(events, meta)

    metrics = aggregate.metrics_json or {}
    metrics["role_fingerprint"] = role_fingerprint
    metrics["utility_iq"] = utility_iq
    metrics["timing_slices"] = timing_slices
    aggregate.metrics_json = metrics
    aggregate.save(update_fields=["metrics_json", "updated_at"])
    return aggregate
