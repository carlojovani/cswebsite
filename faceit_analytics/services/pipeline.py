import traceback

from django.core.cache import cache
from django.utils import timezone

from faceit_analytics.cache_keys import HeatmapKeyParts, heatmap_image_url_key, heatmap_meta_key, profile_metrics_key
from faceit_analytics.constants import ANALYTICS_VERSION
from faceit_analytics.models import AnalyticsAggregate, ProcessingJob
from faceit_analytics.services.aggregates import build_metrics, enrich_metrics_with_role_features
from faceit_analytics.services.heatmaps import DEFAULT_MAPS, get_or_build_heatmap
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


def build_heatmaps(
    job: ProcessingJob,
    profile: PlayerProfile,
    period: str,
    resolution: int = 64,
    progress_start: int = 70,
    progress_end: int = 100,
) -> None:
    maps = list(DEFAULT_MAPS)
    sides = [
        AnalyticsAggregate.SIDE_ALL,
        AnalyticsAggregate.SIDE_CT,
        AnalyticsAggregate.SIDE_T,
    ]
    total = max(len(maps) * len(sides), 1)
    step_size = max(progress_end - progress_start, 1) / total
    counter = 0
    _update_job(job, progress=progress_start)
    for map_name in maps:
        for side in sides:
            get_or_build_heatmap(
                profile_id=profile.id,
                map_name=map_name,
                side=side,
                period=period,
                resolution=resolution,
                version=ANALYTICS_VERSION,
            )
            counter += 1
            _update_job(job, progress=min(int(progress_start + step_size * counter), progress_end))

def _update_job(job: ProcessingJob, **fields) -> None:
    for key, value in fields.items():
        setattr(job, key, value)
    job.save(update_fields=list(fields.keys()) + ["updated_at"])


def _invalidate_cache(profile_id: int, period: str, resolution: int) -> None:
    try:
        cache.delete(profile_metrics_key(profile_id, period, ANALYTICS_VERSION))
        for map_name in DEFAULT_MAPS:
            for side in (
                AnalyticsAggregate.SIDE_ALL,
                AnalyticsAggregate.SIDE_CT,
                AnalyticsAggregate.SIDE_T,
            ):
                parts = HeatmapKeyParts(
                    profile_id=profile_id,
                    map_name=map_name,
                    side=side,
                    period=period,
                    version=ANALYTICS_VERSION,
                    resolution=resolution,
                )
                cache.delete(heatmap_meta_key(parts))
                cache.delete(heatmap_image_url_key(parts))
    except Exception:
        pass


def run_full_pipeline(
    profile_id: int,
    job_id: int,
    period: str = "last_20",
    resolution: int = 64,
) -> None:
    job = ProcessingJob.objects.select_related("profile").get(id=job_id)
    _update_job(
        job,
        status=ProcessingJob.STATUS_RUNNING,
        progress=1,
        error="",
        started_at=job.started_at or timezone.now(),
        finished_at=None,
    )

    try:
        profile = job.profile
        sync_faceit_profile(profile)
        _update_job(job, progress=20)

        aggregates = build_metrics(profile, period=period, analytics_version=ANALYTICS_VERSION)
        if aggregates:
            enrich_metrics_with_role_features(aggregates[0], profile, period)
        _update_job(job, progress=60)

        build_heatmaps(job, profile, period=period, resolution=resolution)
        _invalidate_cache(profile.id, period, resolution)
        _update_job(
            job,
            status=ProcessingJob.STATUS_SUCCESS,
            progress=100,
            finished_at=timezone.now(),
        )
    except Exception:
        _update_job(
            job,
            status=ProcessingJob.STATUS_FAILED,
            error=traceback.format_exc(),
            finished_at=timezone.now(),
        )
        raise
