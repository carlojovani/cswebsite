from __future__ import annotations

from pathlib import Path

from django.conf import settings

from faceit_analytics.models import AnalyticsAggregate
from users.models import PlayerProfile


def _profile_steamid64(profile: PlayerProfile) -> str:
    for attr in ("steamid64", "steam_id64", "steam_id"):
        value = getattr(profile, attr, None)
        if value:
            return str(value).strip()
    return ""


def get_profile_dirs(profile: PlayerProfile) -> dict[str, Path]:
    media_root = Path(getattr(settings, "MEDIA_ROOT", "media"))
    steam_id = _profile_steamid64(profile)

    dirs: dict[str, Path] = {
        "media_root": media_root,
        "heatmaps_root": media_root / "heatmaps",
        "heatmaps_profile_root": media_root / "heatmaps" / str(profile.id),
    }

    if steam_id:
        dirs.update(
            {
                "local_demos_root": media_root / "local_demos" / steam_id,
                "heatmaps_cache_root": media_root / "heatmaps_cache" / steam_id,
                "heatmaps_local_root": media_root / "heatmaps_local" / steam_id,
                "heatmaps_local_aggregate_root": media_root / "heatmaps_local" / steam_id / "aggregate",
            }
        )

    return dirs


def ensure_profile_dirs(profile: PlayerProfile) -> None:
    from faceit_analytics.services.heatmaps import DEFAULT_MAPS

    dirs = get_profile_dirs(profile)
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    for map_name in DEFAULT_MAPS:
        if "local_demos_root" in dirs:
            (dirs["local_demos_root"] / map_name).mkdir(parents=True, exist_ok=True)
            (dirs["heatmaps_cache_root"] / map_name).mkdir(parents=True, exist_ok=True)
            (dirs["heatmaps_local_aggregate_root"] / map_name).mkdir(parents=True, exist_ok=True)

        heatmaps_profile_map = dirs["heatmaps_profile_root"] / map_name
        heatmaps_profile_map.mkdir(parents=True, exist_ok=True)
        for side in (
            AnalyticsAggregate.SIDE_ALL,
            AnalyticsAggregate.SIDE_CT,
            AnalyticsAggregate.SIDE_T,
        ):
            for period in ("last_20", "last_50", "all_time"):
                (heatmaps_profile_map / side / period).mkdir(parents=True, exist_ok=True)


def get_demos_dir(profile: PlayerProfile, map_name: str) -> Path:
    media_root = Path(getattr(settings, "MEDIA_ROOT", "media"))
    steam_id = _profile_steamid64(profile)
    if not steam_id:
        return media_root / "local_demos" / "unknown" / map_name
    return media_root / "local_demos" / steam_id / map_name
