from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import pandas as pd
from awpy import Demo
from django.conf import settings
from django.core.cache import cache

from faceit_analytics.cache_keys import demo_features_key
from faceit_analytics.services.features import (
    compute_role_fingerprint,
    compute_timing_slices,
    compute_utility_iq,
)

TRADE_WINDOW_SECONDS = 5
FLASH_ASSIST_WINDOW_SECONDS = 4
FIRST_CONTACT_WINDOW_SECONDS = 20
MIN_FLASH_DURATION_SECONDS = 0.2
MIN_ROUNDS_REQUIRED = 30
DEMO_FEATURES_TTL_SECONDS = 60 * 60 * 24


@dataclass
class ParsedDemoEvents:
    kills: list[dict[str, Any]]
    flashes: list[dict[str, Any]]
    utility_damage: list[dict[str, Any]]
    round_winners: dict[int, str | None]
    target_round_sides: dict[int, str]
    rounds_in_demo: set[int]
    tick_rate: float


def _period_to_limit(period: str) -> int:
    mapping = {
        "last_20": 20,
        "last_50": 50,
        "all_time": 200,
    }
    return mapping.get(period, 5)


def _normalize_side(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (int, float)):
        val = int(value)
        if val == 2:
            return "T"
        if val == 3:
            return "CT"
    value_str = str(value).strip().lower()
    if value_str in {"t", "terrorist", "2"}:
        return "T"
    if value_str in {"ct", "counterterrorist", "counter-terrorist", "counter_terrorist", "3"}:
        return "CT"
    return None


def _pick_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    if df is None or df.empty:
        return None
    lower = {col.lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    return None


def _safe_int(value: Any) -> int | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def discover_demo_files(profile, period: str, demos_dir: Path | None = None) -> list[Path]:
    root = Path(demos_dir) if demos_dir else Path(getattr(settings, "DEMOS_DIR", settings.BASE_DIR / "demos"))
    candidates: list[Path] = [root / str(profile.id)]
    nickname = getattr(profile.user, "faceit_nickname", "") or ""
    if nickname:
        candidates.append(root / nickname)

    demo_paths: list[Path] = []
    for folder in candidates:
        if not folder.exists():
            continue
        demo_paths.extend(folder.glob("*.dem"))

    demo_paths = sorted(demo_paths, key=lambda p: p.stat().st_mtime, reverse=True)
    limit = max(_period_to_limit(period), 1)
    return demo_paths[:limit]


def compute_demo_set_hash(files: Iterable[Path]) -> str:
    hasher = hashlib.sha1()
    for path in sorted(files, key=lambda p: p.name):
        stats = path.stat()
        hasher.update(path.name.encode("utf-8"))
        hasher.update(str(stats.st_size).encode("utf-8"))
        hasher.update(str(stats.st_mtime).encode("utf-8"))
    return hasher.hexdigest()


def _tick_rate_from_demo(demo: Demo) -> float:
    for attr in ("tickrate", "tick_rate", "tickRate"):
        value = getattr(demo, attr, None)
        if value:
            return float(value)
    header = getattr(demo, "header", None) or {}
    for key in ("tickrate", "tick_rate", "tickRate"):
        if key in header and header[key]:
            return float(header[key])
    return 64.0


def _build_round_meta(rounds_df: pd.DataFrame) -> tuple[dict[int, int], dict[int, float], dict[int, str | None], set[int]]:
    round_start_ticks: dict[int, int] = {}
    round_start_times: dict[int, float] = {}
    round_winners: dict[int, str | None] = {}
    rounds_in_demo: set[int] = set()

    if rounds_df is None or rounds_df.empty:
        return round_start_ticks, round_start_times, round_winners, rounds_in_demo

    round_col = _pick_column(rounds_df, ["round", "round_num", "round_number"])
    start_tick_col = _pick_column(rounds_df, ["start_tick", "round_start_tick", "freeze_end_tick"])
    start_time_col = _pick_column(rounds_df, ["start_time", "round_start_time", "freeze_end_time"])
    winner_col = _pick_column(rounds_df, ["winner", "winning_side", "round_winner"])

    for _, row in rounds_df.iterrows():
        round_number = _safe_int(row.get(round_col)) if round_col else None
        if round_number is None:
            continue
        rounds_in_demo.add(round_number)
        if start_tick_col:
            start_tick = _safe_int(row.get(start_tick_col))
            if start_tick is not None:
                round_start_ticks[round_number] = start_tick
        if start_time_col:
            start_time = _safe_float(row.get(start_time_col))
            if start_time is not None:
                round_start_times[round_number] = start_time
        if winner_col:
            round_winners[round_number] = _normalize_side(row.get(winner_col))

    return round_start_ticks, round_start_times, round_winners, rounds_in_demo


def _round_time_seconds(
    row: pd.Series,
    round_number: int | None,
    round_start_ticks: dict[int, int],
    round_start_times: dict[int, float],
    tick_rate: float,
) -> float | None:
    for key in ("round_time", "time_from_round_start", "time_from_start", "roundTime"):
        if key in row and row.get(key) is not None:
            return _safe_float(row.get(key))

    time_value = None
    for key in ("time", "seconds", "timestamp"):
        if key in row and row.get(key) is not None:
            time_value = _safe_float(row.get(key))
            break

    if round_number is not None and time_value is not None:
        start_time = round_start_times.get(round_number)
        if start_time is not None:
            return time_value - start_time

    tick_value = None
    for key in ("tick", "ticks", "tick_num"):
        if key in row and row.get(key) is not None:
            tick_value = _safe_int(row.get(key))
            break

    if round_number is not None and tick_value is not None:
        start_tick = round_start_ticks.get(round_number)
        if start_tick is not None and tick_rate:
            return (tick_value - start_tick) / tick_rate

    return None


def _extract_player_round_sides(demo: Demo, target_steam_id: str) -> dict[int, str]:
    ticks_df = demo.ticks.to_pandas()
    if ticks_df is None or ticks_df.empty:
        return {}

    steamid_col = _pick_column(ticks_df, ["steamid", "steamID", "player_steamid", "playerSteamID"])
    round_col = _pick_column(ticks_df, ["round", "round_num", "round_number"])
    side_col = _pick_column(ticks_df, ["side", "player_side", "playerSide"])
    if not steamid_col or not round_col or not side_col:
        return {}

    target_id = int(str(target_steam_id))
    sides: dict[int, str] = {}

    filtered = ticks_df[pd.to_numeric(ticks_df[steamid_col], errors="coerce").eq(target_id)]
    for _, row in filtered.iterrows():
        round_number = _safe_int(row.get(round_col))
        if round_number is None or round_number in sides:
            continue
        side = _normalize_side(row.get(side_col))
        if side:
            sides[round_number] = side

    return sides


def parse_demo_events(dem_path: Path, target_steam_id: str | None = None) -> ParsedDemoEvents:
    demo = Demo(str(dem_path), verbose=False)
    demo.parse()

    tick_rate = _tick_rate_from_demo(demo)

    rounds_df = demo.rounds.to_pandas() if getattr(demo, "rounds", None) is not None else None
    round_start_ticks, round_start_times, round_winners, rounds_in_demo = _build_round_meta(rounds_df)

    kills_df = demo.kills.to_pandas() if getattr(demo, "kills", None) is not None else None
    flashes_df = demo.flashes.to_pandas() if getattr(demo, "flashes", None) is not None else None
    damages_df = demo.damages.to_pandas() if getattr(demo, "damages", None) is not None else None

    kills: list[dict[str, Any]] = []
    if kills_df is not None and not kills_df.empty:
        round_col = _pick_column(kills_df, ["round", "round_num", "round_number"])
        attacker_col = _pick_column(kills_df, ["attacker_steamid", "killer_steamid", "attackerSteamID", "killerSteamID"])
        victim_col = _pick_column(kills_df, ["victim_steamid", "victimSteamID"])
        assister_col = _pick_column(kills_df, ["assister_steamid", "assistant_steamid", "assisterSteamID", "assistantSteamID"])
        attacker_side_col = _pick_column(kills_df, ["attacker_side", "attacker_side_name", "attacker_team", "attackerTeam"])
        victim_side_col = _pick_column(kills_df, ["victim_side", "victim_side_name", "victim_team", "victimTeam"])

        for _, row in kills_df.iterrows():
            round_number = _safe_int(row.get(round_col)) if round_col else None
            if round_number is not None:
                rounds_in_demo.add(round_number)
            t_round = _round_time_seconds(row, round_number, round_start_ticks, round_start_times, tick_rate)
            kills.append(
                {
                    "round": round_number,
                    "time": t_round,
                    "attacker": _safe_int(row.get(attacker_col)) if attacker_col else None,
                    "victim": _safe_int(row.get(victim_col)) if victim_col else None,
                    "assister": _safe_int(row.get(assister_col)) if assister_col else None,
                    "attacker_side": _normalize_side(row.get(attacker_side_col)) if attacker_side_col else None,
                    "victim_side": _normalize_side(row.get(victim_side_col)) if victim_side_col else None,
                }
            )

    flashes: list[dict[str, Any]] = []
    if flashes_df is not None and not flashes_df.empty:
        round_col = _pick_column(flashes_df, ["round", "round_num", "round_number"])
        thrower_col = _pick_column(
            flashes_df,
            ["attacker_steamid", "thrower_steamid", "flasher_steamid", "attackerSteamID"],
        )
        blinded_col = _pick_column(
            flashes_df,
            ["player_steamid", "victim_steamid", "blinded_steamid", "playerSteamID"],
        )
        duration_col = _pick_column(flashes_df, ["flash_duration", "duration", "flashDuration"])
        thrower_side_col = _pick_column(flashes_df, ["attacker_side", "thrower_side", "attacker_team", "attackerTeam"])
        blinded_side_col = _pick_column(flashes_df, ["player_side", "blinded_side", "victim_side", "playerTeam"])

        for _, row in flashes_df.iterrows():
            round_number = _safe_int(row.get(round_col)) if round_col else None
            if round_number is not None:
                rounds_in_demo.add(round_number)
            t_round = _round_time_seconds(row, round_number, round_start_ticks, round_start_times, tick_rate)
            thrower_side = _normalize_side(row.get(thrower_side_col)) if thrower_side_col else None
            blinded_side = _normalize_side(row.get(blinded_side_col)) if blinded_side_col else None
            flashes.append(
                {
                    "round": round_number,
                    "time": t_round,
                    "thrower": _safe_int(row.get(thrower_col)) if thrower_col else None,
                    "blinded": _safe_int(row.get(blinded_col)) if blinded_col else None,
                    "duration": _safe_float(row.get(duration_col)) if duration_col else None,
                    "thrower_side": thrower_side,
                    "blinded_side": blinded_side,
                    "is_teamflash": bool(thrower_side and blinded_side and thrower_side == blinded_side),
                }
            )

    utility_damage: list[dict[str, Any]] = []
    if damages_df is not None and not damages_df.empty:
        round_col = _pick_column(damages_df, ["round", "round_num", "round_number"])
        attacker_col = _pick_column(
            damages_df,
            ["attacker_steamid", "attackerSteamID", "attacker_steam_id"],
        )
        victim_col = _pick_column(damages_df, ["victim_steamid", "victimSteamID"])
        dmg_col = _pick_column(damages_df, ["hp_damage", "health_damage", "dmg_health", "damage"])
        weapon_col = _pick_column(damages_df, ["weapon", "weapon_name", "weapon_type", "weaponClass"])

        for _, row in damages_df.iterrows():
            weapon = str(row.get(weapon_col) or "").lower()
            kind = None
            if "hegrenade" in weapon or "he_grenade" in weapon:
                kind = "he"
            elif "molotov" in weapon or "incendiary" in weapon or "incgrenade" in weapon:
                kind = "molotov"
            if not kind:
                continue

            round_number = _safe_int(row.get(round_col)) if round_col else None
            if round_number is not None:
                rounds_in_demo.add(round_number)
            t_round = _round_time_seconds(row, round_number, round_start_ticks, round_start_times, tick_rate)
            damage_amount = _safe_float(row.get(dmg_col)) if dmg_col else None
            if damage_amount is None or damage_amount <= 0:
                continue
            utility_damage.append(
                {
                    "round": round_number,
                    "time": t_round,
                    "attacker": _safe_int(row.get(attacker_col)) if attacker_col else None,
                    "victim": _safe_int(row.get(victim_col)) if victim_col else None,
                    "damage": float(damage_amount),
                    "kind": kind,
                }
            )

    target_round_sides = _extract_player_round_sides(demo, target_steam_id) if target_steam_id else {}

    return ParsedDemoEvents(
        kills=kills,
        flashes=flashes,
        utility_damage=utility_damage,
        round_winners=round_winners,
        target_round_sides=target_round_sides,
        rounds_in_demo=rounds_in_demo,
        tick_rate=tick_rate,
    )


def _is_trade_kill(
    kill: dict[str, Any],
    prior_kills: list[dict[str, Any]],
    target_steam_id: int,
    target_side: str | None,
) -> bool:
    if kill.get("attacker") != target_steam_id or target_side is None:
        return False
    if kill.get("victim_side") == target_side:
        return False
    kill_time = kill.get("time")
    if kill_time is None:
        return False
    for prior in reversed(prior_kills):
        prior_time = prior.get("time")
        if prior_time is None or kill_time - prior_time > TRADE_WINDOW_SECONDS:
            break
        if prior.get("victim_side") != target_side:
            continue
        if prior.get("attacker") == kill.get("victim"):
            return True
    return False


def _is_traded_death(
    kill: dict[str, Any],
    subsequent_kills: list[dict[str, Any]],
    target_steam_id: int,
    target_side: str | None,
) -> bool:
    if kill.get("victim") != target_steam_id or target_side is None:
        return False
    if kill.get("attacker_side") == target_side:
        return False
    kill_time = kill.get("time")
    if kill_time is None:
        return False
    for later in subsequent_kills:
        later_time = later.get("time")
        if later_time is None or later_time - kill_time > TRADE_WINDOW_SECONDS:
            break
        if later.get("attacker_side") != target_side:
            continue
        if later.get("victim") == kill.get("attacker"):
            return True
    return False


def _has_flash_assist(
    kill: dict[str, Any],
    flashes: list[dict[str, Any]],
    target_steam_id: int,
    target_side: str | None,
) -> bool:
    if target_side is None:
        return False
    if kill.get("attacker") == target_steam_id:
        return False
    if kill.get("attacker_side") != target_side:
        return False
    kill_time = kill.get("time")
    if kill_time is None:
        return False
    for flash in flashes:
        if flash.get("thrower") != target_steam_id:
            continue
        if flash.get("blinded") != kill.get("victim"):
            continue
        duration = flash.get("duration") or 0
        if duration <= MIN_FLASH_DURATION_SECONDS:
            continue
        flash_time = flash.get("time")
        if flash_time is None:
            continue
        if 0 <= kill_time - flash_time <= FLASH_ASSIST_WINDOW_SECONDS:
            return True
    return False


def _target_side_for_round(
    round_number: int | None,
    target_round_sides: dict[int, str],
    round_kills: list[dict[str, Any]],
    target_steam_id: int,
) -> str | None:
    if round_number is not None and round_number in target_round_sides:
        return target_round_sides[round_number]
    for kill in round_kills:
        if kill.get("attacker") == target_steam_id:
            return kill.get("attacker_side")
        if kill.get("victim") == target_steam_id:
            return kill.get("victim_side")
    return None


def aggregate_player_features(
    parsed_demos: list[ParsedDemoEvents],
    target_steam_id: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    target_id = int(str(target_steam_id))
    events: list[dict[str, Any]] = []
    rounds_seen: set[tuple[int, int]] = set()
    tick_rate = 64.0

    for demo_index, parsed in enumerate(parsed_demos, start=1):
        tick_rate = parsed.tick_rate or tick_rate
        if parsed.target_round_sides:
            rounds_seen.update({(demo_index, r) for r in parsed.target_round_sides.keys()})

        by_round: dict[int | None, list[dict[str, Any]]] = {}
        for kill in parsed.kills:
            by_round.setdefault(kill.get("round"), []).append(kill)

        flashes_by_round: dict[int | None, list[dict[str, Any]]] = {}
        for flash in parsed.flashes:
            flashes_by_round.setdefault(flash.get("round"), []).append(flash)

        utility_by_round: dict[int | None, list[dict[str, Any]]] = {}
        for dmg in parsed.utility_damage:
            utility_by_round.setdefault(dmg.get("round"), []).append(dmg)

        round_numbers = set(by_round.keys()) | set(flashes_by_round.keys()) | set(utility_by_round.keys())
        for round_number in round_numbers:
            round_key = (demo_index, round_number or 0)
            round_kills_sorted = sorted(by_round.get(round_number, []), key=lambda k: k.get("time") or 0)
            target_side = _target_side_for_round(round_number, parsed.target_round_sides, round_kills_sorted, target_id)

            round_events: list[dict[str, Any]] = []

            if round_number is not None and not parsed.target_round_sides:
                rounds_seen.add(round_key)

            first_kill = round_kills_sorted[0] if round_kills_sorted else None

            for idx, kill in enumerate(round_kills_sorted):
                kill_time = kill.get("time")
                prior = round_kills_sorted[:idx]
                later = round_kills_sorted[idx + 1 :]
                is_first_duel = first_kill is not None and kill is first_kill

                if kill.get("attacker") == target_id:
                    round_events.append(
                        {
                            "type": "kill",
                            "round": round_number,
                            "time": kill_time,
                            "is_trade_kill": _is_trade_kill(kill, prior, target_id, target_side),
                            "is_first_duel": is_first_duel,
                            "is_first_duel_win": is_first_duel,
                        }
                    )

                if kill.get("victim") == target_id:
                    round_events.append(
                        {
                            "type": "death",
                            "round": round_number,
                            "time": kill_time,
                            "was_traded": _is_traded_death(kill, later, target_id, target_side),
                            "is_first_duel": is_first_duel,
                        }
                    )

                if kill.get("assister") == target_id:
                    round_events.append(
                        {
                            "type": "assist",
                            "round": round_number,
                            "time": kill_time,
                        }
                    )

                if _has_flash_assist(kill, flashes_by_round.get(round_number, []), target_id, target_side):
                    round_events.append(
                        {
                            "type": "flash_assist",
                            "round": round_number,
                            "time": kill_time,
                            "is_flash_assist": True,
                        }
                    )

            flashes_for_round = flashes_by_round.get(round_number, [])
            for flash in flashes_for_round:
                if flash.get("thrower") != target_id:
                    continue
                round_events.append(
                    {
                        "type": "flash",
                        "round": round_number,
                        "time": flash.get("time"),
                        "is_friendly_flash": flash.get("is_teamflash"),
                    }
                )

            for dmg in utility_by_round.get(round_number, []):
                if dmg.get("attacker") != target_id:
                    continue
                round_events.append(
                    {
                        "type": "damage",
                        "round": round_number,
                        "time": dmg.get("time"),
                        "utility_damage": dmg.get("damage"),
                    }
                )

            if round_kills_sorted and target_side is not None:
                alive_ct = 5
                alive_t = 5
                target_alive = True
                clutch_marked = False
                for kill in round_kills_sorted:
                    if kill.get("victim_side") == "CT":
                        alive_ct = max(alive_ct - 1, 0)
                    elif kill.get("victim_side") == "T":
                        alive_t = max(alive_t - 1, 0)

                    if kill.get("victim") == target_id:
                        target_alive = False

                    team_alive = alive_ct if target_side == "CT" else alive_t
                    enemy_alive = alive_t if target_side == "CT" else alive_ct
                    if target_alive and team_alive == 1 and enemy_alive >= 1 and not clutch_marked:
                        clutch_marked = True
                        round_events.append(
                            {
                                "type": "clutch_opportunity",
                                "round": round_number,
                                "time": kill.get("time"),
                                "clutch_opportunity": True,
                            }
                        )
                        winner_side = parsed.round_winners.get(round_number)
                        if winner_side and winner_side == target_side:
                            round_events.append(
                                {
                                    "type": "clutch_win",
                                    "round": round_number,
                                    "time": kill.get("time"),
                                    "clutch_win": True,
                                }
                            )

            contact_times = [
                event.get("time")
                for event in round_events
                if event.get("type") in {"kill", "death", "assist"}
                and event.get("time") is not None
            ]
            if contact_times:
                first_contact_time = min(contact_times)
                if first_contact_time <= FIRST_CONTACT_WINDOW_SECONDS:
                    for event in round_events:
                        if (
                            event.get("time") == first_contact_time
                            and event.get("type") in {"kill", "death", "assist"}
                        ):
                            event["is_first_contact"] = True
                            break

            events.extend(round_events)

    rounds_total = len(rounds_seen)

    meta = {
        "rounds": rounds_total if rounds_total else None,
        "tick_rate": tick_rate,
    }

    return events, meta


def get_or_build_demo_features(
    profile,
    period: str,
    analytics_version: str,
    *,
    force_rebuild: bool = False,
    progress_callback: Callable[[int], None] | None = None,
    progress_start: int = 10,
    progress_end: int = 40,
) -> dict[str, Any]:
    steam_id = (getattr(profile, "steam_id", None) or "").strip()
    demo_files = discover_demo_files(profile, period)
    demos_count = len(demo_files)
    demo_set_hash = compute_demo_set_hash(demo_files) if demo_files else ""

    cache_key = demo_features_key(profile.id, period, analytics_version, demo_set_hash)
    if not force_rebuild:
        cached = cache.get(cache_key)
        if cached:
            return cached

    debug = {
        "demos_count": demos_count,
        "rounds_count": 0,
        "kills_events_count": 0,
        "flash_events_count": 0,
        "util_damage_events_count": 0,
        "demo_set_hash": demo_set_hash,
        "min_rounds_required": MIN_ROUNDS_REQUIRED,
    }

    if demos_count == 0 or not steam_id:
        meta = {
            "rounds": None,
            "period": period,
            "profile_id": profile.id,
        }
        timing_slices = compute_timing_slices([], meta)
        role_fingerprint = compute_role_fingerprint([], None, {**meta, "timing_slices": timing_slices})
        utility_iq = compute_utility_iq([], meta)
        role_fingerprint["debug"] = debug
        payload = {
            "role_fingerprint": role_fingerprint,
            "utility_iq": utility_iq,
            "timing_slices": timing_slices,
            "debug": debug,
            "rounds_total": 0,
            "demos_count": demos_count,
            "demo_set_hash": demo_set_hash,
            "insufficient_rounds": True,
        }
        cache.set(cache_key, payload, DEMO_FEATURES_TTL_SECONDS)
        return payload

    parsed_demos: list[ParsedDemoEvents] = []
    for index, demo_path in enumerate(demo_files, start=1):
        parsed = parse_demo_events(demo_path, target_steam_id=steam_id)
        parsed_demos.append(parsed)
        debug["kills_events_count"] += len(parsed.kills)
        debug["flash_events_count"] += len(parsed.flashes)
        debug["util_damage_events_count"] += len(parsed.utility_damage)
        debug["rounds_count"] += len(parsed.rounds_in_demo)
        if progress_callback:
            span = max(progress_end - progress_start, 1)
            progress = progress_start + int((index / max(demos_count, 1)) * span)
            progress_callback(progress)

    events, meta = aggregate_player_features(parsed_demos, steam_id)
    rounds_total = meta.get("rounds") or 0

    if rounds_total < MIN_ROUNDS_REQUIRED:
        meta = {
            "rounds": None,
            "period": period,
            "profile_id": profile.id,
        }
        timing_slices = compute_timing_slices([], meta)
        role_fingerprint = compute_role_fingerprint([], None, {**meta, "timing_slices": timing_slices})
        utility_iq = compute_utility_iq([], meta)
        insufficient_rounds = True
    else:
        meta = {
            "rounds": rounds_total,
            "period": period,
            "profile_id": profile.id,
            "tick_rate": meta.get("tick_rate"),
        }
        timing_slices = compute_timing_slices(events, meta)
        role_fingerprint = compute_role_fingerprint(events, None, {**meta, "timing_slices": timing_slices})
        utility_iq = compute_utility_iq(events, meta)
        insufficient_rounds = False

    role_fingerprint["debug"] = debug

    payload = {
        "role_fingerprint": role_fingerprint,
        "utility_iq": utility_iq,
        "timing_slices": timing_slices,
        "debug": debug,
        "rounds_total": rounds_total,
        "demos_count": demos_count,
        "demo_set_hash": demo_set_hash,
        "insufficient_rounds": insufficient_rounds,
    }

    cache.set(cache_key, payload, DEMO_FEATURES_TTL_SECONDS)
    if progress_callback:
        progress_callback(progress_end)
    return payload
