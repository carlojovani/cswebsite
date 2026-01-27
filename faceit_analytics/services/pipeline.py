from django.utils import timezone

from faceit_analytics.models import AnalyticsAggregate
from faceit_analytics.services.aggregates import build_metrics
from faceit_analytics.services.heatmaps import DEFAULT_MAPS, upsert_heatmap_aggregate
from users.faceit import fetch_faceit_profile_details
from users.models import PlayerProfile


def sync_faceit_profile(profile: PlayerProfile) -> None:
    nickname = profile.user.faceit_nickname
    if not nickname:
        return

    faceit_data = fetch_faceit_profile_details(nickname)
    if not faceit_data:
        return

    profile.faceit_player_id = faceit_data.get("player_id", profile.faceit_player_id)
    profile.country = faceit_data.get("country", profile.country)
    profile.level = faceit_data.get("skill_level", profile.level)
    profile.elo = faceit_data.get("faceit_elo", profile.elo)
    profile.skill_level = faceit_data.get("skill_level", profile.skill_level)
    profile.faceit_url = faceit_data.get("faceit_url", profile.faceit_url)
    profile.avatar = faceit_data.get("avatar", profile.avatar)
    profile.steam_id = (
        faceit_data.get("steam_id_64")
        or faceit_data.get("game_player_id")
        or profile.steam_id
    )
    profile.matches = faceit_data.get("matches", profile.matches)
    profile.wins = faceit_data.get("wins", profile.wins)
    profile.win_rate = faceit_data.get("win_rate", profile.win_rate)
    profile.average_kd = faceit_data.get("average_kd", profile.average_kd)
    profile.average_hs = faceit_data.get("average_hs", profile.average_hs)
    profile.current_win_streak = faceit_data.get("current_win_streak", profile.current_win_streak)
    profile.longest_win_streak = faceit_data.get("longest_win_streak", profile.longest_win_streak)
    profile.last_faceit_update = timezone.now()
    profile.save()


def build_heatmaps(profile: PlayerProfile, period: str, resolution: int = 64) -> None:
    for map_name in DEFAULT_MAPS:
        for side in (
            AnalyticsAggregate.SIDE_ALL,
            AnalyticsAggregate.SIDE_CT,
            AnalyticsAggregate.SIDE_T,
        ):
            upsert_heatmap_aggregate(
                profile=profile,
                map_name=map_name,
                side=side,
                period=period,
                resolution=resolution,
            )


def run_full_pipeline(profile_id: int, requested_by_id: int | None = None, period: str = "last_20") -> None:
    profile = PlayerProfile.objects.get(id=profile_id)
    sync_faceit_profile(profile)
    build_metrics(profile, period=period)
    build_heatmaps(profile, period=period, resolution=64)
