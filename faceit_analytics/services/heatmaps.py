from __future__ import annotations

from pathlib import Path
from typing import Iterable

from django.conf import settings
from django.core.files import File
from django.utils import timezone
from PIL import Image

from faceit_analytics.analyzer import build_heatmaps_aggregate
from faceit_analytics.models import AnalyticsAggregate, HeatmapAggregate

DEFAULT_MAPS: Iterable[str] = ("de_mirage",)


def _period_to_limit(period: str) -> int:
    mapping = {
        "last_20": 20,
        "last_50": 50,
        "all_time": 200,
    }
    return mapping.get(period, 5)


def build_heatmap_grid(
    profile,
    map_name: str,
    side: str,
    period: str,
    resolution: int = 64,
) -> tuple[list[list[int]], Path]:
    steamid64 = (profile.steam_id or "").strip()
    if not steamid64:
        raise ValueError("SteamID64 is missing on player profile")

    media_root = Path(getattr(settings, "MEDIA_ROOT", "media"))
    demos_dir = media_root / "local_demos" / steamid64 / map_name
    out_dir = media_root / "heatmaps_local" / steamid64 / "aggregate" / map_name
    cache_dir = media_root / "heatmaps_cache"

    limit_value = _period_to_limit(period)

    stats = build_heatmaps_aggregate(
        steamid64=steamid64,
        map_name=map_name,
        limit=limit_value,
        demos_dir=demos_dir,
        out_dir=out_dir,
        cache_dir=cache_dir,
    )

    files = stats.get("files", {})
    if side == AnalyticsAggregate.SIDE_CT:
        filename = files.get("presence_ct")
    elif side == AnalyticsAggregate.SIDE_T:
        filename = files.get("presence_t")
    else:
        filename = files.get("presence")

    if not filename:
        raise FileNotFoundError("Heatmap image not generated")

    image_path = out_dir / filename
    if not image_path.exists():
        raise FileNotFoundError(f"Heatmap image not found at {image_path}")

    with Image.open(image_path) as image:
        grid_image = image.convert("L").resize((resolution, resolution))
        pixels = list(grid_image.getdata())

    grid = [
        pixels[row * resolution : (row + 1) * resolution]
        for row in range(resolution)
    ]

    return grid, image_path


def render_heatmap_png(grid: list[list[int]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    height = len(grid)
    width = len(grid[0]) if grid else 0
    if not width or not height:
        raise ValueError("Grid is empty")

    flat = [value for row in grid for value in row]
    max_value = max(flat) if flat else 1
    if max_value == 0:
        max_value = 1
    normalized = [int(value / max_value * 255) for value in flat]

    image = Image.new("L", (width, height))
    image.putdata(normalized)
    image = image.resize((width * 4, height * 4))
    image.save(output_path, format="PNG")


def upsert_heatmap_aggregate(
    profile,
    map_name: str,
    side: str,
    period: str,
    resolution: int = 64,
    analytics_version: str = "v1",
) -> HeatmapAggregate:
    grid, image_path = build_heatmap_grid(profile, map_name, side, period, resolution)

    aggregate, _ = HeatmapAggregate.objects.update_or_create(
        profile=profile,
        map_name=map_name,
        side=side,
        period=period,
        analytics_version=analytics_version,
        resolution=resolution,
        defaults={
            "grid_json": grid,
            "updated_at": timezone.now(),
        },
    )

    with image_path.open("rb") as image_file:
        aggregate.image.save(image_path.name, File(image_file), save=True)

    return aggregate
