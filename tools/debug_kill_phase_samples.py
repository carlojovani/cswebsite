"""tools/debug_kill_phase_samples.py

Usage (PowerShell):
  python manage.py shell -c "from tools.debug_kill_phase_samples import main; main()"

ENV:
  PROFILE_ID=2
  MAP_NAME=de_mirage
  DEMO_NAME=some_demo.dem
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from faceit_analytics.services.demo_events import (
    _assign_kill_phase,
    _distance_to_point,
    _infer_objective_site_from_ticks,
    _load_zone_config,
    _site_centers,
    _target_side_for_round,
    _zone_from_coords,
    parse_demo_events,
)
from faceit_analytics.services.paths import get_demos_dir
from users.models import PlayerProfile


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if not value:
        return default
    try:
        return int(value)
    except Exception:
        return default


def _pick_demo(paths: list[Path], demo_name: str | None) -> Path:
    if demo_name:
        for p in paths:
            if p.name == demo_name:
                return p
        raise SystemExit(f"Demo {demo_name} not found in {paths[0].parent if paths else 'demos dir'}")
    return paths[-1]


def _iter_demo_paths(profile: PlayerProfile, map_name: str) -> list[Path]:
    demos_dir = Path(get_demos_dir(profile, map_name))
    return sorted(demos_dir.glob("*.dem"))


def _kill_attacker_is_target(kill: dict[str, Any], target_steam: str, target_name: str | None) -> bool:
    if str(kill.get("attacker_steamid64")) == target_steam:
        return True
    return bool(target_name and kill.get("attacker_name") == target_name)


def main() -> None:
    profile_id = _env_int("PROFILE_ID", 2)
    map_name = os.environ.get("MAP_NAME", "de_mirage")
    demo_name = os.environ.get("DEMO_NAME")

    profile = PlayerProfile.objects.filter(id=profile_id).first()
    if not profile:
        raise SystemExit(f"PlayerProfile id={profile_id} not found")

    demo_paths = _iter_demo_paths(profile, map_name)
    if not demo_paths:
        raise SystemExit(f"No demos found in {get_demos_dir(profile, map_name)}")

    demo_path = _pick_demo(demo_paths, demo_name)
    print(f"Using demo: {demo_path.name}")

    target_steam = str(profile.steam_id)
    target_name = profile.name
    parsed = parse_demo_events(demo_path, target_steam_id=target_steam)
    config = _load_zone_config(parsed.map_name or map_name)
    centers = _site_centers(parsed.map_name or map_name, config)

    print("CT kill samples (first 30):")
    print("round\ttick\tx\ty\tzone\tdist_a\tdist_b\tstate")

    printed = 0
    kills_by_round: dict[int | None, list[dict[str, Any]]] = {}
    for kill in parsed.kills:
        if _kill_attacker_is_target(kill, target_steam, target_name):
            kills_by_round.setdefault(kill.get("round"), []).append(kill)

    rounds = set(kills_by_round.keys()) | set((parsed.tick_positions_by_round or {}).keys())
    for round_number in sorted(rounds):
        if round_number is None:
            continue
        round_kills = kills_by_round.get(round_number, [])
        target_side = _target_side_for_round(
            round_number,
            parsed.target_round_sides,
            round_kills,
            int(target_steam),
            target_name,
        )
        plant_info = parsed.bomb_plants_by_round.get(round_number, {}) if parsed.bomb_plants_by_round else {}
        plant_tick = plant_info.get("tick") if plant_info else None
        plant_time = plant_info.get("time") if plant_info else None
        target_positions = [
            pos
            for pos in (parsed.tick_positions_by_round or {}).get(round_number, [])
            if pos.get("is_target")
        ]
        objective_site = plant_info.get("site") if plant_info else None
        if objective_site not in {"A", "B"}:
            objective_site = _infer_objective_site_from_ticks(
                target_positions,
                round_kills,
                parsed.map_name or map_name,
                config,
                centers,
                plant_tick,
                plant_time,
            )

        for kill in round_kills:
            kill_side = target_side or kill.get("attacker_side")
            if kill_side != "CT":
                continue
            kill_x = kill.get("attacker_x")
            kill_y = kill.get("attacker_y")
            dist_a = _distance_to_point(kill_x, kill_y, centers.get("A"))
            dist_b = _distance_to_point(kill_x, kill_y, centers.get("B"))
            zone = _zone_from_coords(parsed.map_name or map_name, kill_x, kill_y, config)
            state = _assign_kill_phase(
                kill,
                kill_side,
                parsed.map_name or map_name,
                config,
                centers,
                plant_info,
                objective_site,
            )
            print(
                f"{round_number}\t{kill.get('tick')}\t{kill_x}\t{kill_y}\t{zone}"
                f"\t{dist_a}\t{dist_b}\t{state}"
            )
            printed += 1
            if printed >= 30:
                return
