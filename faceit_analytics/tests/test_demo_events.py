import tempfile
from pathlib import Path

import numpy as np

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


def test_normalize_steamid64_scientific():
    assert demo_events.normalize_steamid64("76561198016259349") == 76561198016259349
    value = 7.6561198016259349e16
    assert demo_events.normalize_steamid64(value) == 76561198016259349


def test_safe_json_converts_numpy():
    payload = {"steamid": np.uint64(76561198016259349), "value": np.nan}
    result = demo_events.safe_json(payload)
    assert result["steamid"] == 76561198016259349
    assert result["value"] is None


def test_name_fallback_matching():
    target_id = 76561198016259349
    parsed = demo_events.ParsedDemoEvents(
        kills=[
            {
                "round": 1,
                "time": 10.0,
                "tick": 100,
                "attacker_steamid64": 111,
                "victim_steamid64": target_id,
                "attacker_name": "Other",
                "victim_name": "Player",
                "attacker_side": "T",
                "victim_side": "CT",
            },
            {
                "round": 1,
                "time": 20.0,
                "tick": 200,
                "attacker_steamid64": target_id + 5,
                "victim_steamid64": 222,
                "attacker_name": "Player",
                "victim_name": "Enemy",
                "attacker_side": "CT",
                "victim_side": "T",
            },
        ],
        flashes=[],
        utility_damage=[],
        flash_events_count=0,
        round_winners={},
        target_round_sides={},
        rounds_in_demo={1},
        tick_rate=128.0,
        tick_rate_approx=False,
        missing_time_kills=0,
        missing_time_flashes=0,
        missing_time_utility=0,
        approx_time_kills=0,
        attacker_none_count=0,
        attacker_id_sample={"attacker": None, "victim": None},
        debug={},
    )

    _, _, debug, _entry, _support = demo_events.aggregate_player_features([parsed], str(target_id))
    assert debug["player_kills"] == 1
    assert debug["player_deaths"] == 1
    assert debug["target_name"] == "Player"


def test_awareness_before_death_basic():
    events = [
        {"type": "damage", "round": 1, "time": 10.0},
        {"type": "death", "round": 1, "time": 12.0},
        {"type": "death", "round": 1, "time": 40.0},
    ]
    result = demo_events.compute_awareness_before_death(events)
    assert result["aware_deaths"] == 1
    assert result["total_deaths"] == 2
    assert result["awareness_before_death_rate"] == 50.0


def test_multikill_basic_case():
    events = [
        {"type": "kill", "round": 1, "time": 5.0, "attacker_place": "A Site", "phase": "t_execute"},
        {"type": "kill", "round": 1, "time": 12.0, "attacker_place": "A Site", "phase": "t_execute"},
        {"type": "kill", "round": 2, "time": 50.0, "attacker_place": "B Site", "phase": "t_post_plant"},
    ]
    result = demo_events.compute_multikill_metrics(events, "de_mirage")
    assert result["multikill_events"] == 1
    assert result["by_timing"]["early"] == 1
    assert result["by_zone"]["A"] == 1
    assert result["by_state"]["t_execute"]["k2"] == 1


def test_global_round_ids_across_demos():
    target_id = 76561198016259349
    parsed_one = demo_events.ParsedDemoEvents(
        kills=[
            {
                "round": 1,
                "time": 5.0,
                "tick": 100,
                "attacker_steamid64": target_id,
                "victim_steamid64": 111,
                "attacker_name": "Player",
                "victim_name": "Enemy",
                "attacker_side": "T",
                "victim_side": "CT",
                "attacker_x": 100.0,
                "attacker_y": 100.0,
            },
        ],
        flashes=[],
        utility_damage=[],
        flash_events_count=0,
        round_winners={},
        target_round_sides={1: "T"},
        rounds_in_demo={1},
        tick_positions_by_round={},
        bomb_plants_by_round={},
        map_name="de_mirage",
        tick_rate=128.0,
        tick_rate_approx=False,
        missing_time_kills=0,
        missing_time_flashes=0,
        missing_time_utility=0,
        approx_time_kills=0,
        attacker_none_count=0,
        attacker_id_sample={"attacker": None, "victim": None},
        debug={},
    )
    parsed_two = demo_events.ParsedDemoEvents(
        kills=[
            {
                "round": 1,
                "time": 8.0,
                "tick": 110,
                "attacker_steamid64": target_id,
                "victim_steamid64": 222,
                "attacker_name": "Player",
                "victim_name": "Enemy2",
                "attacker_side": "T",
                "victim_side": "CT",
                "attacker_x": 120.0,
                "attacker_y": 110.0,
            },
        ],
        flashes=[],
        utility_damage=[],
        flash_events_count=0,
        round_winners={},
        target_round_sides={1: "T"},
        rounds_in_demo={1},
        tick_positions_by_round={},
        bomb_plants_by_round={},
        map_name="de_mirage",
        tick_rate=128.0,
        tick_rate_approx=False,
        missing_time_kills=0,
        missing_time_flashes=0,
        missing_time_utility=0,
        approx_time_kills=0,
        attacker_none_count=0,
        attacker_id_sample={"attacker": None, "victim": None},
        debug={},
    )

    events, _, _, _, _ = demo_events.aggregate_player_features([parsed_one, parsed_two], str(target_id))
    kill_rounds = {event.get("round") for event in events if event.get("type") == "kill"}
    assert kill_rounds == {1001, 2001}
    assert {event.get("round_num") for event in events if event.get("type") == "kill"} == {1}
    assert {event.get("demo_index") for event in events if event.get("type") == "kill"} == {1, 2}


def test_entry_breakdown_assisted_by_proximity():
    target_id = 76561198016259349
    parsed = demo_events.ParsedDemoEvents(
        kills=[
            {
                "round": 1,
                "time": 5.0,
                "tick": 100,
                "attacker_steamid64": target_id,
                "victim_steamid64": 111,
                "attacker_name": "Player",
                "victim_name": "Enemy",
                "attacker_side": "T",
                "victim_side": "CT",
                "attacker_x": 100.0,
                "attacker_y": 100.0,
            },
            {
                "round": 1,
                "time": 6.0,
                "tick": 120,
                "attacker_steamid64": 222,
                "victim_steamid64": 333,
                "attacker_name": "Ally",
                "victim_name": "Enemy2",
                "attacker_side": "T",
                "victim_side": "CT",
                "attacker_x": 120.0,
                "attacker_y": 110.0,
            },
        ],
        flashes=[],
        utility_damage=[],
        flash_events_count=0,
        round_winners={},
        target_round_sides={1: "T"},
        rounds_in_demo={1},
        tick_positions_by_round={
            1: [
                {"time": 5.0, "tick": 90, "steamid": 222, "x": 120.0, "y": 110.0, "is_target": False},
            ]
        },
        bomb_plants_by_round={},
        map_name="de_mirage",
        tick_rate=128.0,
        tick_rate_approx=False,
        missing_time_kills=0,
        missing_time_flashes=0,
        missing_time_utility=0,
        approx_time_kills=0,
        attacker_none_count=0,
        attacker_id_sample={"attacker": None, "victim": None},
        debug={},
    )

    _events, _meta, _debug, entry_breakdown, _support = demo_events.aggregate_player_features(
        [parsed], str(target_id)
    )
    assert entry_breakdown["entry_attempts"] == 1
    assert entry_breakdown["assisted_entry_count"] == 1
    assert entry_breakdown["assisted_by_bucket"]["0-15"] == 1


def test_multikill_state_distribution_no_double_count():
    events = [
        {"type": "kill", "round": 1, "time": 5.0, "phase": "t_execute"},
        {"type": "kill", "round": 1, "time": 8.0, "phase": "t_execute"},
        {"type": "kill", "round": 1, "time": 15.0, "phase": "t_post_plant"},
        {"type": "kill", "round": 1, "time": 18.0, "phase": "t_post_plant"},
        {"type": "kill", "round": 1, "time": 22.0, "phase": "t_post_plant"},
    ]
    result = demo_events.compute_multikill_metrics(events, "de_mirage")
    assert result["by_state"]["t_execute"]["k2"] == 1
    assert result["by_state"]["t_post_plant"]["k3"] == 1
    assert result["by_state"]["t_execute"]["k5"] == 0
    assert result["by_state"]["t_post_plant"]["k5"] == 0
    assert result["ace_rounds"] == 1


def test_kill_support_proximity_window():
    target_id = 76561198016259349
    parsed = demo_events.ParsedDemoEvents(
        kills=[
            {
                "round": 1,
                "time": 10.0,
                "tick": 100,
                "attacker_steamid64": target_id,
                "victim_steamid64": 111,
                "attacker_name": "Player",
                "victim_name": "Enemy",
                "attacker_side": "T",
                "victim_side": "CT",
                "attacker_x": 100.0,
                "attacker_y": 100.0,
            }
        ],
        flashes=[],
        utility_damage=[],
        flash_events_count=0,
        round_winners={},
        target_round_sides={1: "T"},
        rounds_in_demo={1},
        tick_positions_by_round={
            1: [
                {"time": 9.0, "tick": 90, "steamid": 222, "x": 120.0, "y": 110.0, "is_target": False},
                {"time": 0.0, "tick": 500, "steamid": 333, "x": 2000.0, "y": 2000.0, "is_target": False},
            ]
        },
        bomb_plants_by_round={},
        map_name="de_mirage",
        tick_rate=128.0,
        tick_rate_approx=False,
        missing_time_kills=0,
        missing_time_flashes=0,
        missing_time_utility=0,
        approx_time_kills=0,
        attacker_none_count=0,
        attacker_id_sample={"attacker": None, "victim": None},
        debug={},
    )

    events, _meta, _debug, _entry_breakdown, support = demo_events.aggregate_player_features(
        [parsed], str(target_id)
    )
    kill_event = next(event for event in events if event.get("type") == "kill")
    assert kill_event["support_category"] == "partner"
    assert support["with_partner_kills"] == 1


def test_parse_demo_events_bomb_data(monkeypatch, tmp_path):
    bomb_df = np.array(
        [(1, "planted", 1000, 10.0, 20.0, "A", 76561198000000001)],
        dtype=object,
    )

    class DummyDemo:
        tickrate = 64.0
        header = {"map_name": "de_mirage"}

        def __init__(self, *_args, **_kwargs):
            self.rounds = demo_events.pd.DataFrame(
                [
                    {"round_num": 1, "start_tick": 900, "start_time": 0.0},
                ]
            )
            self.kills = demo_events.pd.DataFrame()
            self.flashes = demo_events.pd.DataFrame()
            self.util_damage = demo_events.pd.DataFrame()
            self.damages = demo_events.pd.DataFrame()
            self.ticks = demo_events.pd.DataFrame()
            self.bomb = demo_events.pd.DataFrame(
                bomb_df,
                columns=["round_num", "event", "tick", "X", "Y", "bombsite", "steamid"],
            )

        def parse(self):
            return None

    monkeypatch.setattr(demo_events, "Demo", DummyDemo)
    demo_path = tmp_path / "match.dem"
    demo_path.write_bytes(b"demo")

    parsed = demo_events.parse_demo_events(demo_path, target_steam_id="76561198000000001")
    assert parsed.bomb_plants_by_round[1]["site"] == "A"
    assert parsed.bomb_plants_by_round[1]["x"] == 10.0
    assert parsed.bomb_events_by_round[1][0]["event"] == "plant"
