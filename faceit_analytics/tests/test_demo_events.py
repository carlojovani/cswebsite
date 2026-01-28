import tempfile
from pathlib import Path

from faceit_analytics.services import demo_events


def test_compute_demo_set_hash_stable():
    with tempfile.TemporaryDirectory() as tmp_dir:
        path_a = Path(tmp_dir) / "a.dem"
        path_b = Path(tmp_dir) / "b.dem"
        path_a.write_text("alpha")
        path_b.write_text("bravo")

        first = demo_events.compute_demo_set_hash([path_a, path_b])
        second = demo_events.compute_demo_set_hash([path_b, path_a])

        assert first == second


def test_trade_kill_window():
    target_id = 111
    target_side = "CT"
    prior_kills = [
        {
            "time": 4,
            "attacker": 222,
            "victim": 333,
            "victim_side": "CT",
        },
        {
            "time": 8,
            "attacker": 222,
            "victim": 444,
            "victim_side": "CT",
        },
    ]
    kill = {
        "time": 10,
        "attacker": target_id,
        "victim": 222,
        "victim_side": "T",
    }

    assert demo_events._is_trade_kill(kill, prior_kills, target_id, target_side) is True

    kill_late = {
        "time": 20,
        "attacker": target_id,
        "victim": 222,
        "victim_side": "T",
    }
    assert demo_events._is_trade_kill(kill_late, prior_kills, target_id, target_side) is False


def test_flash_assist_matching():
    target_id = 999
    target_side = "CT"
    kill = {
        "time": 20,
        "attacker": 111,
        "victim": 333,
        "attacker_side": "CT",
    }
    flashes = [
        {
            "thrower": target_id,
            "blinded": 333,
            "duration": 1.0,
            "time": 18,
        }
    ]

    assert demo_events._has_flash_assist(kill, flashes, target_id, target_side) is True

    flashes_short = [
        {
            "thrower": target_id,
            "blinded": 333,
            "duration": 0.1,
            "time": 18,
        }
    ]
    assert demo_events._has_flash_assist(kill, flashes_short, target_id, target_side) is False
