from django.utils import timezone

from faceit_analytics.models import AnalyticsAggregate
from faceit_analytics.services.adapters import adapt_feature_inputs
from faceit_analytics.services.features import (
    compute_role_fingerprint,
    compute_timing_slices,
    compute_utility_iq,
)
from faceit_analytics.utils import to_jsonable

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
    *,
    demo_features: dict | None = None,
) -> AnalyticsAggregate:
    demo_features_debug = None
    demo_features_approx = False
    if demo_features:
        timing_slices = demo_features.get("timing_slices")
        role_fingerprint = demo_features.get("role_fingerprint")
        utility_iq = demo_features.get("utility_iq")
        demo_features_debug = demo_features.get("debug")
        if demo_features.get("insufficient_rounds"):
            demo_features_approx = True
            if role_fingerprint is not None:
                role_fingerprint["approx"] = True
            if utility_iq is not None:
                utility_iq["approx"] = True
            if timing_slices is not None:
                timing_slices["approx"] = True
    else:
        events, positions, meta = adapt_feature_inputs(profile, period)
        timing_slices = compute_timing_slices(events, meta)
        meta = {**meta, "timing_slices": timing_slices}
        role_fingerprint = compute_role_fingerprint(events, positions, meta)
        utility_iq = compute_utility_iq(events, meta)
        demo_features_debug = {"reason": "demo_features_missing"}
        demo_features_approx = True

    metrics = aggregate.metrics_json or {}
    metrics["role_fingerprint"] = role_fingerprint
    metrics["utility_iq"] = utility_iq
    metrics["timing_slices"] = timing_slices
    metrics["demo_features_debug"] = demo_features_debug
    metrics["demo_features_approx"] = demo_features_approx
    aggregate.metrics_json = to_jsonable(metrics)
    aggregate.save(update_fields=["metrics_json", "updated_at"])
    return aggregate
