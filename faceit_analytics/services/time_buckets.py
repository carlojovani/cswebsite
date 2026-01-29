from __future__ import annotations

from typing import Iterable

from django.conf import settings

DEFAULT_TIME_BUCKETS: dict[str, tuple[int, int | None]] = {
    "0-15": (0, 15),
    "0-30": (0, 30),
    "0-45": (0, 45),
    "0-60": (0, 60),
    "0-75": (0, 75),
    "0-90": (0, 90),
    "0+": (0, None),
}


def _normalize_range(value: Iterable[int | None]) -> tuple[int, int | None]:
    start, end = value
    start_int = int(start)
    end_int = None if end is None else int(end)
    return start_int, end_int


def get_time_bucket_presets() -> dict[str, tuple[int, int | None]]:
    presets = getattr(settings, "HEATMAP_TIME_BUCKETS", DEFAULT_TIME_BUCKETS)
    return {str(key).lower(): _normalize_range(value) for key, value in dict(presets).items()}


def get_time_bucket_labels() -> list[str]:
    presets = get_time_bucket_presets()
    order = getattr(settings, "HEATMAP_TIME_BUCKET_ORDER", list(presets.keys()))
    labels = []
    for label in order:
        label_norm = str(label).lower()
        if label_norm in presets:
            labels.append(label_norm)
    return labels or list(presets.keys())


def normalize_time_bucket(value: str | None) -> str:
    if not value:
        return "all"
    value_str = str(value).strip().lower()
    if value_str in {"all", "any"}:
        return "all"
    presets = get_time_bucket_presets()
    if value_str in presets:
        return value_str
    return "all"


def bucket_range(label: str) -> tuple[int, int | None] | None:
    if not label:
        return None
    presets = get_time_bucket_presets()
    return presets.get(str(label).strip().lower())


def time_bucket_for_seconds(value: float | None) -> str:
    if value is None:
        return "unknown"
    presets = get_time_bucket_presets()
    for label, (start, end) in presets.items():
        if end is None:
            if value >= start:
                return label
            continue
        if start <= value < (end + 1):
            return label
    return "late"
