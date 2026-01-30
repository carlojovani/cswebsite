from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backend.settings")
import django
import pandas as pd

django.setup()

from faceit_analytics.services import demo_events

class _FakeTable:
    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    def to_pandas(self, *args, **kwargs) -> pd.DataFrame:
        return self._df


class _FakeDemo:
    def __init__(self, _path: str, verbose: bool = False) -> None:
        self.header = {"map_name": "de_mirage", "tickrate": 64}
        self.tickrate = 64
        self.kills = _FakeTable(
            pd.DataFrame(
                [
                    {
                        "round": 1,
                        "tick": 128,
                        "attacker_steamid": "76561198000000001",
                        "victim_steamid": "76561198000000002",
                        "attacker_name": pd.NA,
                        "victim_name": "Target",
                        "attacker_X": 10.0,
                        "attacker_Y": 20.0,
                    }
                ]
            )
        )
        self.flashes = _FakeTable(pd.DataFrame([]))
        self.damages = _FakeTable(
            pd.DataFrame(
                [
                    {
                        "round": 1,
                        "tick": 200,
                        "attacker_steamid": "76561198000000001",
                        "victim_steamid": "76561198000000002",
                        "weapon": "hegrenade",
                        "hp_damage": 10,
                    }
                ]
            )
        )
        self.rounds = _FakeTable(pd.DataFrame([{"round": 1, "start_tick": 100, "start_time": 0.0}]))
        self.ticks = _FakeTable(pd.DataFrame([]))

    def parse(self) -> None:
        return None


def test_parse_demo_events_smoke(monkeypatch, tmp_path: Path) -> None:
    demo_path = tmp_path / "demo.dem"
    demo_path.write_bytes(b"fake")
    monkeypatch.setattr(demo_events, "Demo", _FakeDemo)

    parsed = demo_events.parse_demo_events(demo_path)

    assert parsed.kills
    assert parsed.kills[0]["attacker_name"] is None
