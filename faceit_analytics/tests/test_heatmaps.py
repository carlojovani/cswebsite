import os

import numpy as np

import django
from django.test import override_settings

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backend.settings")
django.setup()

from faceit_analytics.models import HeatmapAggregate  # noqa: E402
from faceit_analytics.services.heatmaps import ensure_heatmap_image, render_heatmap_image  # noqa: E402


def test_render_heatmap_image_output_size():
    grid = [
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 2.0, 0.0],
        [0.0, 0.0, 4.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ]
    image = render_heatmap_image(grid, output_size=64, blur_radius=1.2)
    assert image.size == (64, 64)
    pixels = np.array(image)
    assert pixels.std() > 0


def test_missing_file_regenerates_heatmap(tmp_path):
    grid = [[0.0, 1.0], [0.0, 0.0]]
    aggregate = HeatmapAggregate(
        profile_id=1,
        map_name="de_mirage",
        side="ALL",
        period="last_20",
        analytics_version="v2",
        resolution=64,
        grid=grid,
        max_value=1.0,
        image="heatmaps/missing.png",
    )
    aggregate.save = lambda *args, **kwargs: None

    with override_settings(MEDIA_ROOT=str(tmp_path), MEDIA_URL="/media/"):
        aggregate = ensure_heatmap_image(aggregate)
        assert aggregate.image
        assert aggregate.image.storage.exists(aggregate.image.name)
