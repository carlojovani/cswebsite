from faceit_analytics.services.features import compute_timing_slices


def test_timing_slices_round_time_bins():
    tick_rate = 128
    round_start_tick = 1000
    events = [
        {"type": "kill", "round": 1, "tick": round_start_tick + tick_rate * 10, "round_start_tick": round_start_tick},
        {"type": "kill", "round": 1, "tick": round_start_tick + tick_rate * 40, "round_start_tick": round_start_tick},
        {"type": "kill", "round": 1, "tick": round_start_tick + tick_rate * 70, "round_start_tick": round_start_tick},
        {"type": "kill", "round": 1, "tick": round_start_tick + tick_rate * 100, "round_start_tick": round_start_tick},
    ]
    meta = {"tick_rate": tick_rate}
    result = compute_timing_slices(events, meta)
    buckets = {bucket["label"]: bucket["value"] for bucket in result["buckets"]}

    assert result["approx"] is False
    assert buckets["0-30"] == 25.0
    assert buckets["30-60"] == 25.0
    assert buckets["60-90"] == 25.0
    assert buckets["90+"] == 25.0
