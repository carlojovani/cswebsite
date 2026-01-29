from __future__ import annotations

from dataclasses import dataclass


DEFAULT_TTL_SECONDS = 60 * 10


@dataclass(frozen=True)
class HeatmapKeyParts:
    profile_id: int
    map_name: str
    metric: str
    side: str
    period: str
    time_slice: str
    version: str
    resolution: int


def profile_metrics_key(profile_id: int, period: str, map_name: str, version: str) -> str:
    return f"profile_metrics:{profile_id}:{map_name}:{period}:{version}"


def heatmap_meta_key(parts: HeatmapKeyParts) -> str:
    return (
        "heatmap_meta:"
        f"{parts.profile_id}:{parts.map_name}:{parts.metric}:{parts.side}:"
        f"{parts.period}:{parts.time_slice}:{parts.version}:{parts.resolution}"
    )


def heatmap_image_url_key(parts: HeatmapKeyParts) -> str:
    return (
        "heatmap_image_url:"
        f"{parts.profile_id}:{parts.map_name}:{parts.metric}:{parts.side}:"
        f"{parts.period}:{parts.time_slice}:{parts.version}:{parts.resolution}"
    )

def demo_features_key(profile_id: int, period: str, map_name: str, demo_set_hash: str, version: str) -> str:
    return f"demo_features:{profile_id}:{map_name}:{period}:{version}:{demo_set_hash}"
