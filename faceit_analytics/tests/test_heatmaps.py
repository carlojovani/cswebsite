import os
from pathlib import Path
from unittest import mock

import numpy as np

import django
from django.test import override_settings

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backend.settings")
django.setup()

from faceit_analytics import analyzer  # noqa: E402
from faceit_analytics.models import HeatmapAggregate  # noqa: E402
from faceit_analytics.services.heatmaps import (  # noqa: E402
    _build_heatmap_filename,
    ensure_heatmap_image,
    render_heatmap_image,
)


def test_render_heatmap_image_output_size():
    grid = [
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 2.0, 0.0],
        [0.0, 0.0, 4.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ]
    image = render_heatmap_image(grid, output_size=1024)
    assert image.size == (1024, 1024)
    assert image.mode == "RGBA"
    pixels = np.array(image)
    assert pixels.std() > 0


def test_render_heatmap_no_blur_keeps_hotspot_visible():
    grid = [
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 5.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ]
    with override_settings(
        HEATMAP_BLUR_SIGMA_GRID=0,
        HEATMAP_BLUR_SIGMA_OUTPUT=0,
        HEATMAP_ALPHA=0.6,
        HEATMAP_GAMMA=0.85,
        HEATMAP_NORM_PERCENTILE=99,
    ):
        image = render_heatmap_image(grid, output_size=64, blur_sigma_grid=0)
    pixels = np.array(image)
    alpha = pixels[:, :, 3]
    assert alpha.max() > 120
    assert np.percentile(alpha, 90) < 200
    assert pixels[:, :, :3].max() > 40


def test_metric_affects_filename_or_url():
    grid = [[0.0, 1.0], [0.0, 0.0]]
    kills = HeatmapAggregate(
        profile_id=1,
        map_name="de_mirage",
        metric=HeatmapAggregate.METRIC_KILLS,
        side="all",
        period="last_20",
        time_slice="all",
        analytics_version="v2",
        resolution=64,
        grid=grid,
        max_value=1.0,
    )
    deaths = HeatmapAggregate(
        profile_id=1,
        map_name="de_mirage",
        metric=HeatmapAggregate.METRIC_DEATHS,
        side="all",
        period="last_20",
        time_slice="all",
        analytics_version="v2",
        resolution=64,
        grid=grid,
        max_value=1.0,
    )
    presence = HeatmapAggregate(
        profile_id=1,
        map_name="de_mirage",
        metric=HeatmapAggregate.METRIC_PRESENCE,
        side="all",
        period="last_20",
        time_slice="all",
        analytics_version="v2",
        resolution=64,
        grid=grid,
        max_value=1.0,
    )
    grid_array = np.array(grid, dtype=np.float32)
    kills_name = _build_heatmap_filename(kills, grid_array)
    deaths_name = _build_heatmap_filename(deaths, grid_array)
    presence_name = _build_heatmap_filename(presence, grid_array)
    assert kills_name != deaths_name
    assert kills_name != presence_name
    assert "kills" in kills_name
    assert "deaths" in deaths_name
    assert "presence" in presence_name


def test_force_regenerates_version(tmp_path):
    grid = [[0.0, 1.0], [0.0, 0.0]]
    aggregate = HeatmapAggregate(
        profile_id=1,
        map_name="de_mirage",
        metric=HeatmapAggregate.METRIC_KILLS,
        side="all",
        period="last_20",
        time_slice="all",
        analytics_version="v2",
        resolution=64,
        grid=grid,
        max_value=1.0,
    )
    aggregate.save = lambda *args, **kwargs: None

    with override_settings(MEDIA_ROOT=str(tmp_path), MEDIA_URL="/media/"):
        first = ensure_heatmap_image(aggregate, force=True)
        first_name = first.image.name
        first_path = Path(tmp_path) / first_name
        first_version = int(first_path.stat().st_mtime)
        first_updated = first.updated_at

        second = ensure_heatmap_image(aggregate, force=True)
        second_name = second.image.name
        second_path = Path(tmp_path) / second_name
        second_version = int(second_path.stat().st_mtime)
        second_updated = second.updated_at

    assert first_name != second_name or first_version != second_version
    assert second_updated >= first_updated


def test_force_regenerates_and_deletes_old_file(tmp_path):
    grid = [[0.0, 1.0], [0.0, 0.0]]
    aggregate = HeatmapAggregate(
        profile_id=1,
        map_name="de_mirage",
        metric=HeatmapAggregate.METRIC_KILLS,
        side="all",
        period="last_20",
        time_slice="all",
        analytics_version="v2",
        resolution=64,
        grid=grid,
        max_value=1.0,
    )
    aggregate.save = lambda *args, **kwargs: None

    with override_settings(MEDIA_ROOT=str(tmp_path), MEDIA_URL="/media/"):
        first = ensure_heatmap_image(aggregate, force=True)
        first_path = Path(tmp_path) / first.image.name
        assert first_path.exists()

        second = ensure_heatmap_image(aggregate, force=True)
        second_path = Path(tmp_path) / second.image.name

    assert second_path.exists()
    assert not first_path.exists()


def test_missing_file_regenerates_heatmap(tmp_path):
    grid = [[0.0, 1.0], [0.0, 0.0]]
    aggregate = HeatmapAggregate(
        profile_id=1,
        map_name="de_mirage",
        metric=HeatmapAggregate.METRIC_KILLS,
        side="all",
        period="last_20",
        time_slice="all",
        analytics_version="v2",
        resolution=64,
        grid=grid,
        max_value=1.0,
    )
    aggregate.save = lambda *args, **kwargs: None

    with override_settings(MEDIA_ROOT=str(tmp_path), MEDIA_URL="/media/"):
        aggregate = ensure_heatmap_image(aggregate, force=True)
        assert aggregate.image
        image_path = Path(tmp_path) / aggregate.image.name
        assert image_path.exists()
        image_path.unlink()

        aggregate = ensure_heatmap_image(aggregate)
        assert aggregate.image
        assert aggregate.image.storage.exists(aggregate.image.name)


def test_heat_overlay_not_invisible():
    grid = [
        [0.0, 2.0, 0.0, 0.0],
        [0.0, 5.0, 3.0, 0.0],
        [0.0, 3.0, 6.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]
    with override_settings(
        HEATMAP_OUTPUT_SIZE=64,
        HEATMAP_BLUR_SIGMA_GRID=0.6,
        HEATMAP_BLUR_SIGMA_OUTPUT=0.0,
        HEATMAP_GAMMA=0.6,
        HEATMAP_ALPHA=0.75,
        HEATMAP_NORM_PERCENTILE=99.5,
    ):
        image = render_heatmap_image(grid)
    pixels = np.array(image)
    alpha = pixels[:, :, 3]
    visible_ratio = (alpha > 10).sum() / alpha.size
    assert visible_ratio > 0.005


def test_atomic_write_used(tmp_path):
    grid = [[0.0, 1.0], [0.0, 0.0]]
    aggregate = HeatmapAggregate(
        profile_id=1,
        map_name="de_mirage",
        metric=HeatmapAggregate.METRIC_KILLS,
        side="all",
        period="last_20",
        time_slice="all",
        analytics_version="v2",
        resolution=64,
        grid=grid,
        max_value=1.0,
    )
    aggregate.save = lambda *args, **kwargs: None

    with override_settings(MEDIA_ROOT=str(tmp_path), MEDIA_URL="/media/"):
        with mock.patch("faceit_analytics.services.heatmaps.os.replace", wraps=os.replace) as replace_mock:
            aggregate = ensure_heatmap_image(aggregate, force=True)
            assert replace_mock.called

        tmp_files = list(Path(tmp_path).rglob("*.tmp.*"))
        assert not tmp_files
        assert aggregate.image.storage.exists(aggregate.image.name)


def test_metric_renders_different_palettes():
    grid = [
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 3.0, 6.0, 0.0],
        [0.0, 2.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ]
    with override_settings(
        HEATMAP_BLUR_SIGMA_GRID=0,
        HEATMAP_BLUR_SIGMA_OUTPUT=0,
        HEATMAP_GAMMA=1.0,
        HEATMAP_ALPHA=1.0,
        HEATMAP_NORM_PERCENTILE=100,
    ):
        kills = render_heatmap_image(grid, output_size=32, blur_sigma_grid=0, cmap_name=analyzer.CMAP_KILLS)
        deaths = render_heatmap_image(grid, output_size=32, blur_sigma_grid=0, cmap_name=analyzer.CMAP_DEATHS)
        presence = render_heatmap_image(grid, output_size=32, blur_sigma_grid=0, cmap_name=analyzer.CMAP_ALL)
    kills_pixels = np.array(kills)
    deaths_pixels = np.array(deaths)
    presence_pixels = np.array(presence)
    assert not np.array_equal(kills_pixels, deaths_pixels)
    assert not np.array_equal(kills_pixels, presence_pixels)
