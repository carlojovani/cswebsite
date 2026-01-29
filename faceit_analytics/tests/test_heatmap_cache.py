import numpy as np
from PIL import Image
from django.test import override_settings
from unittest import mock

from faceit_analytics import analyzer
from faceit_analytics.services.heatmaps import _collect_points_from_cache, build_heatmap_grid


def _write_demo(tmp_path, steamid, map_name, demo_name="match.dem"):
    demos_dir = tmp_path / "local_demos" / steamid / map_name
    demos_dir.mkdir(parents=True, exist_ok=True)
    demo_path = demos_dir / demo_name
    demo_path.write_bytes(b"demo")
    return demo_path, demos_dir


def _write_cache(tmp_path, steamid, map_name, demo_path, data, radar_name, radar_size):
    cache_name = analyzer._demo_cache_hash(demo_path, radar_name, radar_size)
    cache_dir = tmp_path / "heatmaps_cache" / steamid / map_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{cache_name}.npz"
    np.savez_compressed(cache_path, **data)
    return radar_size


def test_slice_filters_points_changes_grid(tmp_path):
    steamid = "76561198000000099"
    map_name = "de_mirage"
    demo_path, demos_dir = _write_demo(tmp_path, steamid, map_name)
    radar_name = "fake_radar"
    radar_size = (100, 100)
    radar_image = Image.new("RGBA", radar_size)

    radar_size = _write_cache(
        tmp_path,
        steamid,
        map_name,
        demo_path,
        {
            "presence_all_px": np.array([[10.0, 10.0]], dtype=np.float32),
            "presence_all_pxt": np.array([[10.0, 10.0, 10.0], [50.0, 50.0, 20.0]], dtype=np.float32),
        },
        radar_name,
        radar_size,
    )

    with override_settings(MEDIA_ROOT=tmp_path), mock.patch(
        "faceit_analytics.analyzer.load_radar_and_meta",
        return_value=(radar_image, {}, radar_name),
    ):
        points_early, _size, _meta = _collect_points_from_cache(
            demos_dir,
            steamid,
            map_name,
            "last_20",
            "all",
            "presence",
            "0-15",
        )
        points_late, _size, _meta = _collect_points_from_cache(
            demos_dir,
            steamid,
            map_name,
            "last_20",
            "all",
            "presence",
            "0-30",
        )

    bounds = (0.0, 0.0, float(radar_size[0]), float(radar_size[1]))
    grid_early, _ = build_heatmap_grid(points_early, resolution=8, bounds=bounds)
    grid_late, _ = build_heatmap_grid(points_late, resolution=8, bounds=bounds)

    assert points_early != points_late
    assert not np.array_equal(np.array(grid_early), np.array(grid_late))


def test_slice_missing_time_data_returns_empty(tmp_path):
    steamid = "76561198000000098"
    map_name = "de_mirage"
    demo_path, demos_dir = _write_demo(tmp_path, steamid, map_name)
    radar_name = "fake_radar"
    radar_size = (100, 100)
    radar_image = Image.new("RGBA", radar_size)

    _write_cache(
        tmp_path,
        steamid,
        map_name,
        demo_path,
        {"kills_px": np.array([[10.0, 10.0]], dtype=np.float32)},
        radar_name,
        radar_size,
    )

    with override_settings(MEDIA_ROOT=tmp_path), mock.patch(
        "faceit_analytics.analyzer.load_radar_and_meta",
        return_value=(radar_image, {}, radar_name),
    ):
        points, _size, meta = _collect_points_from_cache(
            demos_dir,
            steamid,
            map_name,
            "last_20",
            "all",
            "kills",
            "0-15",
        )

    assert points == []
    assert meta["cache_has_time_data"] is False


def test_heatmap_ct_t_not_equal_when_data_diff(tmp_path):
    steamid = "76561198000000100"
    map_name = "de_mirage"
    demo_path, demos_dir = _write_demo(tmp_path, steamid, map_name)
    radar_name = "fake_radar"
    radar_size = (100, 100)
    radar_image = Image.new("RGBA", radar_size)
    radar_size = _write_cache(
        tmp_path,
        steamid,
        map_name,
        demo_path,
        {
            "presence_ct_px": np.array([[5.0, 5.0]], dtype=np.float32),
            "presence_t_px": np.array([[90.0, 90.0]], dtype=np.float32),
            "presence_ct_pxt": np.array([[5.0, 5.0, 10.0]], dtype=np.float32),
            "presence_t_pxt": np.array([[90.0, 90.0, 10.0]], dtype=np.float32),
        },
        radar_name,
        radar_size,
    )

    with override_settings(MEDIA_ROOT=tmp_path), mock.patch(
        "faceit_analytics.analyzer.load_radar_and_meta",
        return_value=(radar_image, {}, radar_name),
    ):
        points_ct, _size, _meta = _collect_points_from_cache(
            demos_dir,
            steamid,
            map_name,
            "last_20",
            "ct",
            "presence",
            "all",
        )
        points_t, _size, _meta = _collect_points_from_cache(
            demos_dir,
            steamid,
            map_name,
            "last_20",
            "t",
            "presence",
            "all",
        )

    bounds = (0.0, 0.0, float(radar_size[0]), float(radar_size[1]))
    grid_ct, _ = build_heatmap_grid(points_ct, resolution=8, bounds=bounds)
    grid_t, _ = build_heatmap_grid(points_t, resolution=8, bounds=bounds)

    assert points_ct != points_t
    assert not np.array_equal(np.array(grid_ct), np.array(grid_t))
