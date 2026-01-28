from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from django.conf import settings
from django.core.files import File
from django.db import transaction
from django.utils import timezone
from PIL import Image, ImageFilter

from faceit_analytics import analyzer
from faceit_analytics.constants import ANALYTICS_VERSION
from faceit_analytics.models import AnalyticsAggregate, HeatmapAggregate
from users.models import PlayerProfile

DEFAULT_MAPS: Iterable[str] = ("de_mirage",)
HEATMAP_OUTPUT_SIZE = int(getattr(settings, "HEATMAP_OUTPUT_SIZE", 1024))
HEATMAP_BLUR_FACTOR = float(getattr(settings, "HEATMAP_BLUR_FACTOR", 6))


def _period_to_limit(period: str) -> int:
    mapping = {
        "last_20": 20,
        "last_50": 50,
        "all_time": 200,
    }
    return mapping.get(period, 5)


def build_heatmap_grid(
    points: Sequence[Sequence[float]],
    resolution: int = 64,
    bounds: tuple[float, float, float, float] | None = None,
) -> tuple[list[list[float]], float]:
    grid = [[0.0 for _ in range(resolution)] for _ in range(resolution)]
    if not points:
        return grid, 0.0

    if bounds is None:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        minx = min(xs)
        miny = min(ys)
        maxx = max(xs)
        maxy = max(ys)
    else:
        minx, miny, maxx, maxy = bounds

    span_x = maxx - minx
    span_y = maxy - miny
    if span_x == 0 or span_y == 0:
        return grid, 0.0

    for p in points:
        x = p[0]
        y = p[1]
        weight = p[2] if len(p) > 2 else 1.0
        gx = int((x - minx) / span_x * (resolution - 1))
        gy = int((y - miny) / span_y * (resolution - 1))
        if 0 <= gx < resolution and 0 <= gy < resolution:
            grid[gy][gx] += float(weight)

    max_value = max((max(row) for row in grid), default=0.0)
    return grid, float(max_value)


def render_heatmap_png(
    grid: list[list[float]],
    output_path: Path,
    max_value: float | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    height = len(grid)
    width = len(grid[0]) if grid else 0
    if not width or not height:
        raise ValueError("Grid is empty")

    flat = [value for row in grid for value in row]
    peak = max_value if max_value is not None else max(flat) if flat else 1.0
    if peak <= 0:
        peak = 1.0
    normalized = [int(min(value / peak, 1.0) * 255) for value in flat]

    image = Image.new("L", (width, height))
    image.putdata(normalized)

    output_size = max(HEATMAP_OUTPUT_SIZE, 1)
    image = image.resize((output_size, output_size), Image.Resampling.BICUBIC)

    blur_factor = max(HEATMAP_BLUR_FACTOR, 1)
    scale = output_size / max(width, height)
    blur_radius = max(scale / blur_factor, 0.1)
    image = image.filter(ImageFilter.GaussianBlur(blur_radius))

    image.save(output_path, format="PNG")


def _collect_points_from_cache(
    steamid64: str,
    map_name: str,
    period: str,
    side: str,
) -> tuple[list[tuple[float, float, float]], tuple[int, int]]:
    media_root = Path(getattr(settings, "MEDIA_ROOT", "media"))
    demos_dir = media_root / "local_demos" / steamid64 / map_name
    cache_dir = media_root / "heatmaps_cache" / steamid64 / map_name
    out_dir = media_root / "heatmaps_local" / steamid64 / "aggregate" / map_name

    demo_paths = sorted(demos_dir.glob("*.dem"), key=lambda p: p.stat().st_mtime, reverse=True)
    demo_paths = demo_paths[: max(_period_to_limit(period), 1)]
    if not demo_paths:
        return [], (0, 0)

    radar, _meta, radar_name = analyzer.load_radar_and_meta(map_name)
    radar_size = radar.size

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_paths: list[Path] = []
    for dem_path in demo_paths:
        cache_name = analyzer._demo_cache_hash(dem_path, radar_name, radar_size)
        cache_paths.append(cache_dir / f"{cache_name}.npz")

    if not all(path.exists() for path in cache_paths):
        analyzer.build_heatmaps_aggregate(
            steamid64=steamid64,
            map_name=map_name,
            limit=_period_to_limit(period),
            demos_dir=demos_dir,
            out_dir=out_dir,
            cache_dir=media_root / "heatmaps_cache",
        )

    points: list[tuple[float, float, float]] = []
    array_key = {
        AnalyticsAggregate.SIDE_ALL: "presence_all_px",
        AnalyticsAggregate.SIDE_CT: "presence_ct_px",
        AnalyticsAggregate.SIDE_T: "presence_t_px",
    }.get(side, "presence_all_px")

    for cache_path in cache_paths:
        if not cache_path.exists():
            continue
        with np.load(cache_path) as cached:
            data = cached.get(array_key)
            if data is None:
                continue
            for x, y in data.tolist():
                points.append((float(x), float(y), 1.0))

    return points, radar_size


@transaction.atomic
def get_or_build_heatmap(
    profile_id: int,
    map_name: str,
    side: str,
    period: str,
    version: str = ANALYTICS_VERSION,
    resolution: int = 64,
) -> HeatmapAggregate:
    aggregate = HeatmapAggregate.objects.filter(
        profile_id=profile_id,
        map_name=map_name,
        side=side,
        period=period,
        analytics_version=version,
        resolution=resolution,
    ).first()
    if aggregate and aggregate.image:
        return aggregate

    profile = PlayerProfile.objects.get(id=profile_id)

    steamid64 = (profile.steam_id or "").strip()
    if not steamid64:
        raise ValueError("SteamID64 is missing on player profile")

    points, radar_size = _collect_points_from_cache(steamid64, map_name, period, side)
    bounds = (0.0, 0.0, float(radar_size[0] or 1), float(radar_size[1] or 1))
    grid, max_value = build_heatmap_grid(points, resolution=resolution, bounds=bounds)

    aggregate, _ = HeatmapAggregate.objects.update_or_create(
        profile_id=profile_id,
        map_name=map_name,
        side=side,
        period=period,
        analytics_version=version,
        resolution=resolution,
        defaults={
            "grid": grid,
            "max_value": max_value,
            "updated_at": timezone.now(),
        },
    )

    media_root = Path(getattr(settings, "MEDIA_ROOT", "media"))
    output_path = media_root / "heatmaps" / str(profile_id) / map_name / side / period
    output_path = output_path / f"{version}.png"
    render_heatmap_png(grid, output_path, max_value=max_value)

    with output_path.open("rb") as image_file:
        aggregate.image.save(output_path.name, File(image_file), save=True)

    return aggregate
