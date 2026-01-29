from datetime import datetime
from decimal import Decimal
from pathlib import Path

import numpy as np

from faceit_analytics.utils import deep_json_sanitize, to_jsonable


def test_json_sanitize_numpy_scalars():
    payload = {
        "steamid": np.uint64(76561198016259349),
        "count": np.int64(5),
        "ratio": np.float32(1.25),
        "decimal": Decimal("3.5"),
        "path": Path("/tmp/demo.dem"),
        "when": datetime(2024, 1, 1, 12, 30),
    }
    result = deep_json_sanitize(payload)
    assert result["steamid"] == 76561198016259349
    assert result["count"] == 5
    assert result["ratio"] == 1.25
    assert result["decimal"] == 3.5
    assert result["path"] == "/tmp/demo.dem"
    assert result["when"].startswith("2024-01-01T12:30")


def test_to_jsonable_converts_numpy_uint64():
    payload = {"value": np.uint64(123)}
    result = to_jsonable(payload)
    assert result["value"] == 123
