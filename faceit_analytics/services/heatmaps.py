from __future__ import annotations

from io import BytesIO
import hashlib
from pathlib import Path
import time
from typing import Iterable, Sequence

import numpy as np
from django.conf import settings
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from django.db import transaction
from django.utils import timezone
from PIL import Image, ImageFilter
from matplotlib import cm

from faceit_analytics import analyzer
from faceit_analytics.constants import ANALYTICS_VERSION
from faceit_analytics.models import AnalyticsAggregate, HeatmapAggregate
from users.models import PlayerProfile

DEFAULT_MAPS: Iterable[str] = ("de_mirage",)
HEATMAP_OUTPUT_SIZE = int(getattr(settings, "HEATMAP_OUTPUT_SIZE", 768))
HEATMAP_UPSCALE_FILTER = str(getattr(settings, "HEATMAP_UPSCALE_FILTER", "LANCZOS")).upper()
HEATMAP_BLUR_FACTOR = float(
    getattr(settings, "HEATMAP_BLUR_FACTOR", getattr(settings, "HEATMAP_BLUR_RADIUS", 1.0))
)
HEATMAP_PERCENTILE_CLIP = float(
    getattr(settings, "HEATMAP_PERCENTILE_CLIP", getattr(settings, "HEATMAP_CLIP_PCT", 99))
)
HEATMAP_GAMMA = float(getattr(settings, "HEATMAP_GAMMA", 0.85))
HEATMAP_ALPHA = float(getattr(settings, "HEATMAP_ALPHA", 0.55))


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


def render_heatmap_image(
    grid: list[list[float]],
    *,
    output_size: int | None = None,
    blur_radius: float | None = None,
    clip_pct: float | None = None,
    gamma: float | None = None,
    upscale_filter: str | None = None,
    cmap_name: str = analyzer.CMAP_ALL,
) -> Image.Image:
    height = len(grid)
    width = len(grid[0]) if grid else 0
    if not width or not height:
        raise ValueError("Grid is empty")

    arr = np.array(grid, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    positive = arr[arr > 0]
    clip_value = (
        np.percentile(positive, clip_pct if clip_pct is not None else HEATMAP_PERCENTILE_CLIP)
        if positive.size
        else float(arr.max())
    )
    if not clip_value or clip_value <= 0:
        clip_value = 1.0
    arr = np.clip(arr, 0, clip_value) / clip_value

    gamma_value = gamma if gamma is not None else HEATMAP_GAMMA
    if gamma_value and gamma_value > 0:
        arr = arr ** gamma_value

    mask_bytes = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    mask_image = Image.fromarray(mask_bytes, mode="L")
    target_size = max(output_size or HEATMAP_OUTPUT_SIZE, 1)
    if target_size != width or target_size != height:
        resample = _get_resample_filter(upscale_filter)
        mask_image = mask_image.resize((target_size, target_size), resample=resample)

    blur_radius = max(0.0, float(blur_radius if blur_radius is not None else HEATMAP_BLUR_FACTOR))
    if blur_radius:
        mask_image = mask_image.filter(ImageFilter.GaussianBlur(blur_radius))

    mask_arr = np.array(mask_image, dtype=np.float32) / 255.0
    cmap = cm.get_cmap(cmap_name)
    rgba = cmap(mask_arr)
    alpha_value = max(0.0, float(HEATMAP_ALPHA))
    rgba[:, :, 3] = np.clip(mask_arr * alpha_value, 0, 1)
    rgb_bytes = (rgba[:, :, :3] * 255).astype(np.uint8)
    alpha_bytes = (rgba[:, :, 3] * 255).astype(np.uint8)
    out = np.dstack([rgb_bytes, alpha_bytes])
    image = Image.fromarray(out, mode="RGBA")
    return image


def render_heatmap_png(
    grid: list[list[float]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = render_heatmap_image(grid)
    image.save(output_path, format="PNG")


def _get_resample_filter(filter_name: str | None) -> int:
    name = (filter_name or HEATMAP_UPSCALE_FILTER).upper()
    return {
        "NEAREST": Image.Resampling.NEAREST,
        "BILINEAR": Image.Resampling.BILINEAR,
        "BICUBIC": Image.Resampling.BICUBIC,
        "LANCZOS": Image.Resampling.LANCZOS,
    }.get(name, Image.Resampling.LANCZOS)


def _build_heatmap_filename(aggregate: HeatmapAggregate, grid_array: np.ndarray) -> str:
    digest = hashlib.sha256(grid_array.tobytes()).hexdigest()[:8]
    timestamp = int(time.time())
    return (
        f"heatmap_{aggregate.analytics_version}_res{aggregate.resolution}_"
        f"out{HEATMAP_OUTPUT_SIZE}_{digest}_{timestamp}.png"
    )


def ensure_heatmap_image(
    aggregate: HeatmapAggregate,
    *,
    radar_path: Path | None = None,
    force: bool = False,
) -> HeatmapAggregate:
    storage = aggregate.image.storage if aggregate.image else default_storage
    if aggregate.image and aggregate.image.name:
        if not storage.exists(aggregate.image.name):
            aggregate.image = None
            force = True

    if aggregate.image and not force:
        return aggregate

    if not aggregate.grid:
        return aggregate

    media_root = Path(getattr(settings, "MEDIA_ROOT", "media"))
    media_root.mkdir(parents=True, exist_ok=True)

    cmap_name = {
        AnalyticsAggregate.SIDE_CT: analyzer.CMAP_CT,
        AnalyticsAggregate.SIDE_T: analyzer.CMAP_T,
        AnalyticsAggregate.SIDE_ALL: analyzer.CMAP_ALL,
    }.get(aggregate.side, analyzer.CMAP_ALL)

    heatmap_image = render_heatmap_image(
        aggregate.grid,
        output_size=HEATMAP_OUTPUT_SIZE,
        blur_radius=HEATMAP_BLUR_FACTOR,
        clip_pct=HEATMAP_PERCENTILE_CLIP,
        gamma=HEATMAP_GAMMA,
        upscale_filter=HEATMAP_UPSCALE_FILTER,
        cmap_name=cmap_name,
    )

    radar_image = None
    if radar_path:
        try:
            radar_image = Image.open(radar_path).convert("RGBA")
        except OSError:
            radar_image = None
    if radar_image is None:
        try:
            radar_image, _meta, _radar_name = analyzer.load_radar_and_meta(aggregate.map_name)
            radar_image = radar_image.convert("RGBA")
        except Exception:
            radar_image = None

    if radar_image is not None:
        radar_image = radar_image.resize(
            (heatmap_image.width, heatmap_image.height),
            resample=_get_resample_filter(HEATMAP_UPSCALE_FILTER),
        )
        composite = Image.alpha_composite(radar_image, heatmap_image)
    else:
        composite = heatmap_image

    buffer = BytesIO()
    composite.save(buffer, format="PNG", optimize=True)
    buffer_value = buffer.getvalue()

    grid_array = np.array(aggregate.grid, dtype=np.float32)
    filename = _build_heatmap_filename(aggregate, grid_array)
    aggregate.image.save(filename, ContentFile(buffer_value), save=False)
    aggregate.updated_at = timezone.now()
    aggregate.save(update_fields=["image", "updated_at"])
    return aggregate


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
    *,
    force_rebuild: bool = False,
) -> HeatmapAggregate:
    aggregate = HeatmapAggregate.objects.filter(
        profile_id=profile_id,
        map_name=map_name,
        side=side,
        period=period,
        analytics_version=version,
        resolution=resolution,
    ).first()
    if aggregate and not force_rebuild:
        return ensure_heatmap_image(aggregate)

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

    return ensure_heatmap_image(aggregate, force=True)
