from __future__ import annotations

import numpy as np
from PIL import Image

from faceit_analytics import analyzer


def test_heatmap_cache_writer_includes_time_arrays(tmp_path, monkeypatch) -> None:
    demo_path = tmp_path / "local_demos" / "76561198000000001" / "de_mirage" / "match.dem"
    demo_path.parent.mkdir(parents=True, exist_ok=True)
    demo_path.write_bytes(b"demo")

    radar_image = Image.new("RGBA", (64, 64))
    monkeypatch.setattr(
        analyzer,
        "load_radar_and_meta",
        lambda map_name: (radar_image, {"dummy": True}, "fake_radar"),
    )
    monkeypatch.setattr(
        analyzer,
        "_density_to_heat_rgba_pixel",
        lambda *args, **kwargs: radar_image.copy(),
    )
    points = {
        "presence_all_px": np.array([[1.0, 2.0]], dtype=np.float32),
        "presence_ct_px": np.array([[1.0, 2.0]], dtype=np.float32),
        "presence_t_px": np.array([[1.0, 2.0]], dtype=np.float32),
        "presence_all_pxt": np.array([[1.0, 2.0, 5.0]], dtype=np.float32),
        "presence_ct_pxt": np.array([[1.0, 2.0, 5.0]], dtype=np.float32),
        "presence_t_pxt": np.array([[1.0, 2.0, 5.0]], dtype=np.float32),
        "kills_px": np.array([[3.0, 4.0]], dtype=np.float32),
        "deaths_px": np.array([[5.0, 6.0]], dtype=np.float32),
        "kills_pxt": np.array([[3.0, 4.0, 7.0]], dtype=np.float32),
        "deaths_pxt": np.array([[5.0, 6.0, 8.0]], dtype=np.float32),
    }
    monkeypatch.setattr(analyzer, "_extract_points_from_demo", lambda *args, **kwargs: (points, {}))

    analyzer.build_heatmaps_aggregate(
        steamid64="76561198000000001",
        map_name="de_mirage",
        limit=1,
        demos_dir=demo_path.parent,
        out_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
    )

    cache_name = analyzer._demo_cache_hash(demo_path, "fake_radar", radar_image.size)
    cache_path = tmp_path / "cache" / "76561198000000001" / "de_mirage" / f"{cache_name}.npz"
    with np.load(cache_path) as cached:
        assert "presence_all_pxt" in cached.files
        assert "presence_ct_pxt" in cached.files
        assert "presence_t_pxt" in cached.files
        assert "kills_pxt" in cached.files
        assert "deaths_pxt" in cached.files
