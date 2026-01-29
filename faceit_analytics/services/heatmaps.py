from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path
from typing import Iterable, Sequence
from uuid import uuid4

import numpy as np
from django.conf import settings
from django.core.files.storage import default_storage
from django.db import transaction
from django.utils import timezone
from matplotlib import cm
from PIL import Image, ImageFilter

from faceit_analytics import analyzer
from faceit_analytics.constants import ANALYTICS_VERSION
from faceit_analytics.models import AnalyticsAggregate, HeatmapAggregate, heatmap_upload_to
from faceit_analytics.utils import to_jsonable
from users.models import PlayerProfile

DEFAULT_MAPS: Iterable[str] = ("de_mirage",)

# Output image size (radar+heatmap)
HEATMAP_OUTPUT_SIZE = int(getattr(settings, "HEATMAP_OUTPUT_SIZE", 1024))

# Upscale filter for resizing
HEATMAP_UPSCALE_FILTER = str(getattr(settings, "HEATMAP_UPSCALE_FILTER", "LANCZOS")).upper()

# IMPORTANT: blur sigma is now interpreted in GRID units (cells), not pixels.
# For res=64: sigma 0.8..1.4 is typical. For "kills/deaths dots": lower.
# Backward compatible env names:
HEATMAP_BLUR_SIGMA_GRID = float(
    getattr(
        settings,
        "HEATMAP_BLUR_SIGMA_GRID",
        getattr(
            settings,
            "HEATMAP_BLUR_SIGMA",
            getattr(settings, "HEATMAP_BLUR_FACTOR", getattr(settings, "HEATMAP_BLUR_RADIUS", 0.0)),
        ),
    )
)

# Optional post-upscale blur in output pixel space (default disabled)
HEATMAP_BLUR_SIGMA_OUTPUT = float(getattr(settings, "HEATMAP_BLUR_SIGMA_OUTPUT", 0.0))

# Per-metric blur overrides (optional)
HEATMAP_BLUR_SIGMA_KILLS = float(getattr(settings, "HEATMAP_BLUR_SIGMA_KILLS", HEATMAP_BLUR_SIGMA_GRID))
HEATMAP_BLUR_SIGMA_DEATHS = float(getattr(settings, "HEATMAP_BLUR_SIGMA_DEATHS", HEATMAP_BLUR_SIGMA_GRID))
HEATMAP_BLUR_SIGMA_PRESENCE = float(getattr(settings, "HEATMAP_BLUR_SIGMA_PRESENCE", HEATMAP_BLUR_SIGMA_GRID))

# Percentile clip for normalization (higher -> less saturation)
HEATMAP_NORM_PERCENTILE = float(
    getattr(
        settings,
        "HEATMAP_NORM_PERCENTILE",
        getattr(settings, "HEATMAP_PERCENTILE_CLIP", getattr(settings, "HEATMAP_CLIP_PCT", 99.5)),
    )
)

# Gamma (lower -> brighter tails, higher -> more contrast)
HEATMAP_GAMMA = float(getattr(settings, "HEATMAP_GAMMA", 0.75))

# Global alpha multiplier
HEATMAP_ALPHA = float(getattr(settings, "HEATMAP_ALPHA", 0.98))

# Alpha curve power (lower -> more visible faint areas)
HEATMAP_ALPHA_POWER = float(getattr(settings, "HEATMAP_ALPHA_POWER", 0.55))

# Optional unsharp to increase clarity after resize (0 disables)
HEATMAP_UNSHARP_RADIUS = float(getattr(settings, "HEATMAP_UNSHARP_RADIUS", 0.0))
HEATMAP_UNSHARP_PERCENT = int(getattr(settings, "HEATMAP_UNSHARP_PERCENT", 140))
HEATMAP_UNSHARP_THRESHOLD = int(getattr(settings, "HEATMAP_UNSHARP_THRESHOLD", 2))


def _period_to_limit(period: str) -> int:
    mapping = {"last_20": 20, "last_50": 50, "all_time": 200}
    return mapping.get(period, 5)


def _get_resample_filter(filter_name: str | None) -> int:
    name = (filter_name or HEATMAP_UPSCALE_FILTER).upper()
    return {
        "NEAREST": Image.Resampling.NEAREST,
        "BILINEAR": Image.Resampling.BILINEAR,
        "BICUBIC": Image.Resampling.BICUBIC,
        "LANCZOS": Image.Resampling.LANCZOS,
    }.get(name, Image.Resampling.LANCZOS)


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


# ---------- core image ops (NO uint8 until final) ----------

def _gaussian_kernel1d(sigma: float) -> np.ndarray:
    if sigma <= 0:
        return np.array([1.0], dtype=np.float32)
    radius = int(max(1, np.ceil(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-(x * x) / (2.0 * sigma * sigma)).astype(np.float32)
    s = float(k.sum())
    return k / s if s > 0 else k


def _convolve1d_reflect(arr: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    pad = (kernel.size - 1) // 2
    if pad <= 0:
        return arr
    pad_width = [(0, 0)] * arr.ndim
    pad_width[axis] = (pad, pad)
    padded = np.pad(arr, pad_width, mode="reflect")
    # convolve along axis
    out = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="valid"), axis, padded)
    return out.astype(np.float32, copy=False)


def _gaussian_blur_grid(arr: np.ndarray, sigma_grid: float) -> np.ndarray:
    if sigma_grid <= 0:
        return arr
    k = _gaussian_kernel1d(float(sigma_grid))
    out = _convolve1d_reflect(arr, k, axis=0)
    out = _convolve1d_reflect(out, k, axis=1)
    return out


def _metric_blur_sigma(metric: str) -> float:
    if metric == HeatmapAggregate.METRIC_KILLS:
        return HEATMAP_BLUR_SIGMA_KILLS
    if metric == HeatmapAggregate.METRIC_DEATHS:
        return HEATMAP_BLUR_SIGMA_DEATHS
    return HEATMAP_BLUR_SIGMA_PRESENCE


def render_heatmap_image(
    grid: list[list[float]],
    *,
    output_size: int | None = None,
    blur_sigma_grid: float | None = None,
    blur_sigma_output: float | None = None,
    clip_pct: float | None = None,
    gamma: float | None = None,
    upscale_filter: str | None = None,
    cmap_name: str = analyzer.CMAP_ALL,
) -> Image.Image:
    h = len(grid)
    w = len(grid[0]) if grid else 0
    if not w or not h:
        raise ValueError("Grid is empty")

    target_size = int(max(output_size or HEATMAP_OUTPUT_SIZE, 1))

    arr = np.array(grid, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr[arr < 0] = 0.0

    # blur in GRID space (res units) BEFORE resizing, to keep details crisp
    sigma = float(blur_sigma_grid) if blur_sigma_grid is not None else float(HEATMAP_BLUR_SIGMA_GRID)
    if sigma > 0:
        arr = _gaussian_blur_grid(arr, sigma_grid=sigma)

    # resize float mask to target_size (still float32)
    if target_size != w or target_size != h:
        resample = _get_resample_filter(upscale_filter)
        mask_f = Image.fromarray(arr, mode="F").resize((target_size, target_size), resample=resample)
        arr = np.array(mask_f, dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        arr[arr < 0] = 0.0

    # optional blur in OUTPUT pixel space (post-upscale)
    sigma_output = float(blur_sigma_output) if blur_sigma_output is not None else float(HEATMAP_BLUR_SIGMA_OUTPUT)
    if sigma_output > 0:
        arr = _gaussian_blur_grid(arr, sigma_grid=sigma_output)

    # normalize using percentile of positive values
    pos = arr[arr > 0]
    pct = float(clip_pct) if clip_pct is not None else float(HEATMAP_NORM_PERCENTILE)
    clip_value = float(np.percentile(pos, pct)) if pos.size else float(arr.max())
    if not clip_value or clip_value <= 0:
        clip_value = 1.0
    mask = np.clip(arr / clip_value, 0.0, 1.0)

    # gamma
    g = float(gamma) if gamma is not None else float(HEATMAP_GAMMA)
    if g > 0:
        mask = mask ** g

    # colormap to RGBA
    cmap = cm.get_cmap(cmap_name)
    rgba = cmap(mask)  # float64
    rgba = rgba.astype(np.float32)

    # alpha curve (controls visibility on top of radar)
    alpha_mul = float(max(0.0, HEATMAP_ALPHA))
    alpha_pow = float(max(0.01, HEATMAP_ALPHA_POWER))
    rgba[:, :, 3] = np.clip((mask ** alpha_pow) * alpha_mul, 0.0, 1.0)

    out = (rgba * 255.0).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(out, mode="RGBA")

    # optional unsharp to enhance details without “pixelation”
    if HEATMAP_UNSHARP_RADIUS and HEATMAP_UNSHARP_RADIUS > 0:
        img = img.filter(
            ImageFilter.UnsharpMask(
                radius=float(HEATMAP_UNSHARP_RADIUS),
                percent=int(HEATMAP_UNSHARP_PERCENT),
                threshold=int(HEATMAP_UNSHARP_THRESHOLD),
            )
        )
    return img


def _build_heatmap_filename(aggregate: HeatmapAggregate, grid_array: np.ndarray) -> str:
    digest = hashlib.sha256(grid_array.tobytes()).hexdigest()[:8]
    timestamp = time.time_ns()
    return (
        f"heatmap_{aggregate.analytics_version}_{aggregate.metric}_res{aggregate.resolution}_"
        f"out{HEATMAP_OUTPUT_SIZE}_{digest}_{timestamp}.png"
    )


def _atomic_write_png(final_path: Path, render_callable) -> None:
    final_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = final_path.with_name(f"{final_path.name}.tmp.{uuid4().hex}")
    try:
        render_callable(tmp_path)
        os.replace(tmp_path, final_path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def ensure_heatmap_image(
    aggregate: HeatmapAggregate,
    *,
    radar_path: Path | None = None,
    force: bool = False,
) -> HeatmapAggregate:
    storage = aggregate.image.storage if aggregate.image else default_storage

    # if file missing -> force regenerate
    if aggregate.image and aggregate.image.name:
        try:
            exists = storage.exists(aggregate.image.name)
        except Exception:
            exists = False
        if not exists:
            aggregate.image = None
            force = True

    # if force -> remove old file if possible
    if aggregate.image and force:
        try:
            storage.delete(aggregate.image.name)
        except Exception:
            pass
        aggregate.image = None

    if aggregate.image and not force:
        return aggregate

    if not aggregate.grid:
        return aggregate

    media_root = Path(getattr(settings, "MEDIA_ROOT", "media"))
    media_root.mkdir(parents=True, exist_ok=True)

    # choose cmap by metric
    if aggregate.metric == HeatmapAggregate.METRIC_KILLS:
        cmap_name = analyzer.CMAP_KILLS
    elif aggregate.metric == HeatmapAggregate.METRIC_DEATHS:
        cmap_name = analyzer.CMAP_DEATHS
    else:
        cmap_name = {
            AnalyticsAggregate.SIDE_CT: analyzer.CMAP_CT,
            AnalyticsAggregate.SIDE_T: analyzer.CMAP_T,
            AnalyticsAggregate.SIDE_ALL: analyzer.CMAP_ALL,
        }.get(aggregate.side, analyzer.CMAP_ALL)

    # IMPORTANT: use per-metric sigma in GRID units
    blur_sigma = _metric_blur_sigma(aggregate.metric)

    heatmap_image = render_heatmap_image(
        aggregate.grid,
        output_size=HEATMAP_OUTPUT_SIZE,
        blur_sigma_grid=blur_sigma,
        blur_sigma_output=HEATMAP_BLUR_SIGMA_OUTPUT,
        clip_pct=HEATMAP_NORM_PERCENTILE,
        gamma=HEATMAP_GAMMA,
        upscale_filter=HEATMAP_UPSCALE_FILTER,
        cmap_name=cmap_name,
    )

    # load radar
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

    # composite
    if radar_image is not None:
        radar_image = radar_image.resize(
            (heatmap_image.width, heatmap_image.height),
            resample=_get_resample_filter(HEATMAP_UPSCALE_FILTER),
        )
        composite = Image.alpha_composite(radar_image, heatmap_image)
    else:
        composite = heatmap_image

    grid_array = np.array(aggregate.grid, dtype=np.float32)
    filename = _build_heatmap_filename(aggregate, grid_array)
    relative_path = heatmap_upload_to(aggregate, filename)
    final_path = media_root / relative_path

    def _render(path: Path) -> None:
        composite.save(path, format="PNG", optimize=True)

    _atomic_write_png(final_path, _render)
    aggregate.image.name = relative_path
    aggregate.updated_at = timezone.now()
    aggregate.save(update_fields=["image", "updated_at"])
    return aggregate


def _collect_points_from_cache(
    steamid64: str,
    map_name: str,
    period: str,
    side: str,
    metric: str,
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
    if metric == HeatmapAggregate.METRIC_KILLS:
        array_key = "kills_px"
    elif metric == HeatmapAggregate.METRIC_DEATHS:
        array_key = "deaths_px"
    else:
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
    metric: str,
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
        metric=metric,
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

    points, radar_size = _collect_points_from_cache(steamid64, map_name, period, side, metric)
    bounds = (0.0, 0.0, float(radar_size[0] or 1), float(radar_size[1] or 1))
    grid, max_value = build_heatmap_grid(points, resolution=resolution, bounds=bounds)

    aggregate, _ = HeatmapAggregate.objects.update_or_create(
        profile_id=profile_id,
        map_name=map_name,
        metric=metric,
        side=side,
        period=period,
        analytics_version=version,
        resolution=resolution,
        defaults={
            "grid": to_jsonable(grid),
            "max_value": to_jsonable(max_value),
            "updated_at": timezone.now(),
        },
    )

    return ensure_heatmap_image(aggregate, force=True)
