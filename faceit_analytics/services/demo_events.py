from __future__ import annotations

import hashlib
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from awpy import Demo
from django.conf import settings
from django.core.cache import cache

from faceit_analytics.cache_keys import demo_features_key
from faceit_analytics.services.features import (
    compute_role_fingerprint,
    compute_timing_slices,
    compute_utility_iq,
)
from faceit_analytics.services.paths import get_demos_dir
from faceit_analytics.utils import to_jsonable

TRADE_WINDOW_SECONDS = 5
FLASH_ASSIST_WINDOW_SECONDS = 4
FIRST_CONTACT_WINDOW_SECONDS = 20
MIN_FLASH_DURATION_SECONDS = 0.2
MIN_ROUNDS_REQUIRED = 30
MIN_CONTACTS_REQUIRED = 10
DEMO_FEATURES_TTL_SECONDS = 60 * 60 * 24
DEATH_AWARENESS_LOOKBACK_SEC = int(getattr(settings, "DEATH_AWARENESS_LOOKBACK_SEC", 5))
MULTIKILL_WINDOW_SEC = int(getattr(settings, "MULTIKILL_WINDOW_SEC", 10))
MULTIKILL_EARLY_THRESHOLD_SEC = int(getattr(settings, "MULTIKILL_EARLY_THRESHOLD_SEC", 30))
HEATMAP_TIME_SLICES = list(getattr(settings, "HEATMAP_TIME_SLICES", [(0, 999)]))

KILLS_STEAMID_COLUMNS = ["attacker_steamid", "victim_steamid", "assister_steamid"]
UTIL_DAMAGE_STEAMID_COLUMNS = ["attacker_steamid", "victim_steamid"]


def _read_parquet_with_steamid_strings(parquet_path: Path, steam_cols: Iterable[str]) -> pd.DataFrame:
    table = pq.read_table(parquet_path)
    for col in steam_cols:
        if col in table.column_names:
            index = table.schema.get_field_index(col)
            table = table.set_column(index, col, table[col].cast(pa.string()))
    return table.to_pandas()


def _load_demo_dataframe(value: Any, steam_cols: Iterable[str]) -> pd.DataFrame | None:
    if value is None:
        return None
    if isinstance(value, pd.DataFrame):
        return value
    if isinstance(value, (str, Path)):
        parquet_path = Path(value)
        if parquet_path.suffix == ".parquet" and parquet_path.exists():
            return _read_parquet_with_steamid_strings(parquet_path, steam_cols)
    if hasattr(value, "to_pandas"):
        return value.to_pandas()
    return None


@dataclass
class ParsedDemoEvents:
    kills: list[dict[str, Any]]
    flashes: list[dict[str, Any]]
    utility_damage: list[dict[str, Any]]
    flash_events_count: int
    round_winners: dict[int, str | None]
    target_round_sides: dict[int, str]
    rounds_in_demo: set[int]
    tick_rate: float
    tick_rate_approx: bool
    missing_time_kills: int
    missing_time_flashes: int
    missing_time_utility: int
    approx_time_kills: int
    attacker_none_count: int
    attacker_id_sample: dict[str, str | None]
    debug: dict[str, Any]


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


def normalize_steamid64(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        value_str = value.strip()
        if not value_str or value_str.lower() == "nan":
            return None
        if value_str.isdigit():
            return int(value_str)
        try:
            float_value = float(value_str)
        except (TypeError, ValueError):
            return None
        if math.isnan(float_value):
            return None
        return int(round(float_value))
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        float_value = float(value)
        if math.isnan(float_value):
            return None
        return int(round(float_value))
    if hasattr(value, "item"):
        return normalize_steamid64(value.item())
    try:
        float_value = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(float_value):
        return None
    return int(round(float_value))


def steamid_eq(value: Any, target: Any) -> bool:
    normalized_value = normalize_steamid64(value)
    normalized_target = normalize_steamid64(target)
    if normalized_value is None or normalized_target is None:
        return False
    return normalized_value == normalized_target


def safe_json(obj: Any) -> Any:
    return to_jsonable(obj)


def _local_demos_root(demos_dir: Path | None = None) -> Path:
    if demos_dir is not None:
        return Path(demos_dir)
    local_root = getattr(settings, "LOCAL_DEMOS_ROOT", None)
    if local_root:
        return Path(local_root)
    media_root = Path(getattr(settings, "MEDIA_ROOT", "media"))
    return media_root / "local_demos"


def _profile_steamid64(profile) -> str:
    for attr in ("steamid64", "steam_id64", "steam_id"):
        value = getattr(profile, attr, None)
        if value:
            return str(value).strip()
    return ""


def discover_demo_files(profile, period: str, map_name: str, demos_dir: Path | None = None) -> list[Path]:
    if demos_dir is not None:
        demos_root = Path(demos_dir)
    else:
        demos_root = get_demos_dir(profile, map_name)
    steam_id = _profile_steamid64(profile)
    if not steam_id:
        return []
    if not demos_root.exists():
        return []
    demo_paths = list(demos_root.glob("*.dem"))

    demo_paths = sorted(demo_paths, key=lambda p: p.stat().st_mtime, reverse=True)
    limit = max(_period_to_limit(period), 1)
    return demo_paths[:limit]


def compute_demo_set_hash(files: Iterable[Path]) -> str:
    hasher = hashlib.sha1()
    for path in sorted(files, key=lambda p: p.name):
        stats = path.stat()
        hasher.update(str(path.resolve()).encode("utf-8"))
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

    target_id = normalize_steamid64(target_steam_id)
    if target_id is None:
        return {}
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
    tick_rate_approx = True
    if getattr(demo, "tickrate", None) or getattr(demo, "tick_rate", None) or getattr(demo, "tickRate", None):
        tick_rate_approx = False
    header = getattr(demo, "header", None) or {}
    if header.get("tickrate") or header.get("tick_rate") or header.get("tickRate"):
        tick_rate_approx = False

    rounds_df = demo.rounds.to_pandas() if getattr(demo, "rounds", None) is not None else None
    round_start_ticks, round_start_times, round_winners, rounds_in_demo = _build_round_meta(rounds_df)

    kills_df = _load_demo_dataframe(getattr(demo, "kills", None), KILLS_STEAMID_COLUMNS)
    flashes_df = _load_demo_dataframe(getattr(demo, "flashes", None), [])
    util_damage_df = _load_demo_dataframe(
        getattr(demo, "util_damage", None) or getattr(demo, "utility_damage", None),
        UTIL_DAMAGE_STEAMID_COLUMNS,
    )
    damages_df = util_damage_df or _load_demo_dataframe(getattr(demo, "damages", None), UTIL_DAMAGE_STEAMID_COLUMNS)

    kills: list[dict[str, Any]] = []
    missing_time_kills = 0
    approx_time_kills = 0
    attacker_none_count = 0
    attacker_id_sample: dict[str, str | None] = {"attacker": None, "victim": None}
    round_start_tick_map: dict[int, int] = {}
    assistedflash_kill_count = 0
    debug_payload: dict[str, Any] = {
        "tickrate": tick_rate,
        "tickrate_assumed": tick_rate_approx,
        "round_start_tick_sample": dict(list(round_start_ticks.items())[:5]) if round_start_ticks else {},
    }
    if kills_df is not None and not kills_df.empty:
        raw_kills_df = kills_df.copy()
        attacker_steam_raw_type_sample: list[str] = []
        raw_kill_columns = list(kills_df.columns)
        debug_payload["raw_kill_columns"] = raw_kill_columns
        raw_kill_row_sample: dict[str, Any] | None = None
        if raw_kill_columns:
            sample_row = kills_df.iloc[0].to_dict()
            raw_kill_row_sample = {key: sample_row.get(key) for key in list(sample_row.keys())[:20]}
            debug_payload["raw_kill_row_sample"] = raw_kill_row_sample
        time_fields = {"time", "seconds", "round_time", "tick", "game_time", "clock_time"}
        debug_payload["time_fields_present"] = [
            column for column in raw_kill_columns if column.lower() in time_fields
        ]
        round_col = _pick_column(kills_df, ["round", "round_num", "round_number"])
        tick_col = _pick_column(kills_df, ["tick", "tick_num", "ticks"])
        attacker_col = _pick_column(
            kills_df,
            [
                "killer",
                "attacker",
                "killer_steamid",
                "attacker_steamid",
                "killerSteamID",
                "killerSteamId",
                "attackerSteamID",
                "attackerSteamId",
                "attackerSteamID64",
                "killerSteamID64",
            ],
        )
        victim_col = _pick_column(
            kills_df,
            [
                "victim",
                "victim_steamid",
                "victimSteamID",
                "victimSteamId",
                "victimSteamID64",
            ],
        )
        assister_col = _pick_column(
            kills_df,
            [
                "assister",
                "assist",
                "assister_steamid",
                "assistant_steamid",
                "assisterSteamID",
                "assisterSteamId",
                "assistantSteamID",
                "assistantSteamId",
            ],
        )
        attacker_name_col = _pick_column(kills_df, ["attacker_name", "killer_name"])
        victim_name_col = _pick_column(kills_df, ["victim_name"])
        assister_name_col = _pick_column(kills_df, ["assister_name", "assistant_name"])
        assistedflash_col = _pick_column(kills_df, ["assistedflash", "assisted_flash", "assistedFlash"])
        attacker_side_col = _pick_column(kills_df, ["attacker_side", "attacker_side_name", "attacker_team", "attackerTeam"])
        victim_side_col = _pick_column(kills_df, ["victim_side", "victim_side_name", "victim_team", "victimTeam"])
        attacker_place_col = _pick_column(kills_df, ["attacker_place", "attackerPlace"])
        victim_place_col = _pick_column(kills_df, ["victim_place", "victimPlace"])
        attacker_x_col = _pick_column(kills_df, ["attacker_X", "attacker_x"])
        attacker_y_col = _pick_column(kills_df, ["attacker_Y", "attacker_y"])
        victim_x_col = _pick_column(kills_df, ["victim_X", "victim_x"])
        victim_y_col = _pick_column(kills_df, ["victim_Y", "victim_y"])

        if round_col and tick_col:
            for _, row in kills_df.iterrows():
                round_number = _safe_int(row.get(round_col))
                tick_value = _safe_int(row.get(tick_col))
                if round_number is None or tick_value is None:
                    continue
                current = round_start_tick_map.get(round_number)
                if current is None or tick_value < current:
                    round_start_tick_map[round_number] = tick_value

        tickrate_assumed_for_kills = False
        fallback_tick_rate = 128.0

        for idx, row in kills_df.iterrows():
            raw_row = raw_kills_df.loc[idx] if idx in raw_kills_df.index else row
            round_number = _safe_int(row.get(round_col)) if round_col else None
            if round_number is not None:
                rounds_in_demo.add(round_number)
            tick_value = _safe_int(row.get(tick_col)) if tick_col else None
            t_round = _round_time_seconds(row, round_number, round_start_ticks, round_start_times, tick_rate)
            time_approx = False
            start_tick_used = None
            if t_round is None and tick_value is not None:
                start_tick = round_start_ticks.get(round_number) if round_number is not None else None
                if start_tick is None and round_number is not None:
                    start_tick = round_start_tick_map.get(round_number)
                if start_tick is not None:
                    start_tick_used = start_tick
                    t_round = max((tick_value - start_tick) / tick_rate, 0.0) if tick_rate else None
                else:
                    tickrate_assumed_for_kills = True
                    time_approx = True
                    approx_time_kills += 1
                    t_round = max(tick_value / fallback_tick_rate, 0.0)
            if t_round is None:
                missing_time_kills += 1
            attacker_steam_raw = raw_row.get(attacker_col) if attacker_col else None
            victim_steam_raw = raw_row.get(victim_col) if victim_col else None
            assister_steam_raw = raw_row.get(assister_col) if assister_col else None
            attacker_steam_id = normalize_steamid64(attacker_steam_raw)
            victim_steam_id = normalize_steamid64(victim_steam_raw)
            assister_steam_id = normalize_steamid64(assister_steam_raw)
            if attacker_steam_raw is not None and len(attacker_steam_raw_type_sample) < 5:
                attacker_steam_raw_type_sample.append(type(attacker_steam_raw).__name__)
            if attacker_steam_id is None:
                attacker_none_count += 1
            if attacker_id_sample["attacker"] is None and attacker_steam_id:
                attacker_id_sample["attacker"] = str(attacker_steam_id)
            if attacker_id_sample["victim"] is None and victim_steam_id:
                attacker_id_sample["victim"] = str(victim_steam_id)
            attacker_name = str(row.get(attacker_name_col)) if attacker_name_col and row.get(attacker_name_col) else None
            victim_name = str(row.get(victim_name_col)) if victim_name_col and row.get(victim_name_col) else None
            assister_name = (
                str(row.get(assister_name_col)) if assister_name_col and row.get(assister_name_col) else None
            )
            kill_event = {
                "round": round_number,
                "time": t_round,
                "tick": tick_value,
                "round_start_tick": start_tick_used,
                "time_approx": time_approx,
                "attacker": attacker_steam_id,
                "victim": victim_steam_id,
                "assister": assister_steam_id,
                "attacker_steamid64": attacker_steam_id,
                "victim_steamid64": victim_steam_id,
                "assister_steamid64": assister_steam_id,
                "attacker_steam_raw": attacker_steam_raw,
                "victim_steam_raw": victim_steam_raw,
                "assister_steam_raw": assister_steam_raw,
                "attacker_name": attacker_name,
                "victim_name": victim_name,
                "assister_name": assister_name,
                "attacker_side": _normalize_side(row.get(attacker_side_col)) if attacker_side_col else None,
                "victim_side": _normalize_side(row.get(victim_side_col)) if victim_side_col else None,
                "assistedflash": bool(row.get(assistedflash_col)) if assistedflash_col else False,
                "attacker_place": str(row.get(attacker_place_col)) if attacker_place_col and row.get(attacker_place_col) else None,
                "victim_place": str(row.get(victim_place_col)) if victim_place_col and row.get(victim_place_col) else None,
                "attacker_x": _safe_float(row.get(attacker_x_col)) if attacker_x_col else None,
                "attacker_y": _safe_float(row.get(attacker_y_col)) if attacker_y_col else None,
                "victim_x": _safe_float(row.get(victim_x_col)) if victim_x_col else None,
                "victim_y": _safe_float(row.get(victim_y_col)) if victim_y_col else None,
            }
            if "kill_event_sample" not in debug_payload:
                debug_payload["kill_event_sample"] = kill_event
            kills.append(kill_event)
            if kill_event["assistedflash"] and kill_event.get("assister_steamid64"):
                assistedflash_kill_count += 1
        if attacker_steam_raw_type_sample:
            debug_payload["attacker_steam_raw_type_sample"] = attacker_steam_raw_type_sample
        if tickrate_assumed_for_kills:
            debug_payload["tickrate_assumed"] = True

    flashes: list[dict[str, Any]] = []
    flash_events_count = 0
    missing_time_flashes = 0
    if flashes_df is not None and not flashes_df.empty:
        round_col = _pick_column(flashes_df, ["round", "round_num", "round_number"])
        tick_col = _pick_column(flashes_df, ["tick", "tick_num", "ticks"])
        thrower_col = _pick_column(
            flashes_df,
            ["attacker_steamid", "thrower_steamid", "flasher_steamid", "attackerSteamID"],
        )
        blinded_col = _pick_column(
            flashes_df,
            ["player_steamid", "victim_steamid", "blinded_steamid", "playerSteamID"],
        )
        thrower_name_col = _pick_column(flashes_df, ["attacker_name", "thrower_name", "flasher_name"])
        blinded_name_col = _pick_column(flashes_df, ["player_name", "victim_name", "blinded_name"])
        duration_col = _pick_column(flashes_df, ["flash_duration", "duration", "flashDuration"])
        thrower_side_col = _pick_column(flashes_df, ["attacker_side", "thrower_side", "attacker_team", "attackerTeam"])
        blinded_side_col = _pick_column(flashes_df, ["player_side", "blinded_side", "victim_side", "playerTeam"])

        for _, row in flashes_df.iterrows():
            round_number = _safe_int(row.get(round_col)) if round_col else None
            if round_number is not None:
                rounds_in_demo.add(round_number)
            t_round = _round_time_seconds(row, round_number, round_start_ticks, round_start_times, tick_rate)
            tick_value = _safe_int(row.get(tick_col)) if tick_col else None
            if t_round is None and tick_value is not None:
                start_tick = round_start_ticks.get(round_number) if round_number is not None else None
                if start_tick is None and round_number is not None:
                    start_tick = round_start_tick_map.get(round_number)
                if start_tick is not None:
                    t_round = max((tick_value - start_tick) / tick_rate, 0.0) if tick_rate else None
                else:
                    t_round = max(tick_value / 128.0, 0.0)
            if t_round is None:
                missing_time_flashes += 1
            thrower_side = _normalize_side(row.get(thrower_side_col)) if thrower_side_col else None
            blinded_side = _normalize_side(row.get(blinded_side_col)) if blinded_side_col else None
            thrower_id = normalize_steamid64(row.get(thrower_col)) if thrower_col else None
            blinded_id = normalize_steamid64(row.get(blinded_col)) if blinded_col else None
            thrower_name = (
                str(row.get(thrower_name_col)) if thrower_name_col and row.get(thrower_name_col) else None
            )
            blinded_name = (
                str(row.get(blinded_name_col)) if blinded_name_col and row.get(blinded_name_col) else None
            )
            flashes.append(
                {
                    "round": round_number,
                    "time": t_round,
                    "tick": tick_value,
                    "thrower": thrower_id,
                    "blinded": blinded_id,
                    "thrower_steamid64": thrower_id,
                    "blinded_steamid64": blinded_id,
                    "thrower_name": thrower_name,
                    "blinded_name": blinded_name,
                    "duration": _safe_float(row.get(duration_col)) if duration_col else None,
                    "thrower_side": thrower_side,
                    "blinded_side": blinded_side,
                    "is_teamflash": bool(thrower_side and blinded_side and thrower_side == blinded_side),
                }
            )
        flash_events_count = len(flashes)
    elif assistedflash_kill_count:
        flash_events_count = assistedflash_kill_count

    utility_damage: list[dict[str, Any]] = []
    missing_time_utility = 0
    if damages_df is not None and not damages_df.empty:
        raw_damages_df = damages_df.copy()
        round_col = _pick_column(damages_df, ["round", "round_num", "round_number"])
        tick_col = _pick_column(damages_df, ["tick", "tick_num", "ticks"])
        attacker_col = _pick_column(
            damages_df,
            ["attacker_steamid", "attackerSteamID", "attacker_steam_id"],
        )
        victim_col = _pick_column(damages_df, ["victim_steamid", "victimSteamID"])
        attacker_name_col = _pick_column(damages_df, ["attacker_name"])
        victim_name_col = _pick_column(damages_df, ["victim_name"])
        dmg_col = _pick_column(damages_df, ["hp_damage", "health_damage", "dmg_health", "damage"])
        weapon_col = _pick_column(damages_df, ["weapon", "weapon_name", "weapon_type", "weaponClass"])

        for idx, row in damages_df.iterrows():
            raw_row = raw_damages_df.loc[idx] if idx in raw_damages_df.index else row
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
            tick_value = _safe_int(row.get(tick_col)) if tick_col else None
            if t_round is None and tick_value is not None:
                start_tick = round_start_ticks.get(round_number) if round_number is not None else None
                if start_tick is None and round_number is not None:
                    start_tick = round_start_tick_map.get(round_number)
                if start_tick is not None:
                    t_round = max((tick_value - start_tick) / tick_rate, 0.0) if tick_rate else None
                else:
                    t_round = max(tick_value / 128.0, 0.0)
            if t_round is None:
                missing_time_utility += 1
            damage_amount = _safe_float(row.get(dmg_col)) if dmg_col else None
            if damage_amount is None or damage_amount <= 0:
                continue
            attacker_steam_raw = raw_row.get(attacker_col) if attacker_col else None
            victim_steam_raw = raw_row.get(victim_col) if victim_col else None
            attacker_steam_id = normalize_steamid64(attacker_steam_raw)
            victim_steam_id = normalize_steamid64(victim_steam_raw)
            attacker_steam_raw = raw_row.get(attacker_col) if attacker_col else None
            victim_steam_raw = raw_row.get(victim_col) if victim_col else None
            attacker_name = (
                str(row.get(attacker_name_col)) if attacker_name_col and row.get(attacker_name_col) else None
            )
            victim_name = str(row.get(victim_name_col)) if victim_name_col and row.get(victim_name_col) else None
            utility_damage.append(
                {
                    "round": round_number,
                    "time": t_round,
                    "tick": tick_value,
                    "attacker": attacker_steam_id,
                    "victim": victim_steam_id,
                    "attacker_steamid64": attacker_steam_id,
                    "victim_steamid64": victim_steam_id,
                    "attacker_steam_raw": attacker_steam_raw,
                    "victim_steam_raw": victim_steam_raw,
                    "attacker_name": attacker_name,
                    "victim_name": victim_name,
                    "damage": float(damage_amount),
                    "kind": kind,
                }
            )

    target_round_sides = _extract_player_round_sides(demo, target_steam_id) if target_steam_id else {}

    return ParsedDemoEvents(
        kills=kills,
        flashes=flashes,
        utility_damage=utility_damage,
        flash_events_count=flash_events_count,
        round_winners=round_winners,
        target_round_sides=target_round_sides,
        rounds_in_demo=rounds_in_demo,
        tick_rate=tick_rate,
        tick_rate_approx=tick_rate_approx,
        missing_time_kills=missing_time_kills,
        missing_time_flashes=missing_time_flashes,
        missing_time_utility=missing_time_utility,
        approx_time_kills=approx_time_kills,
        attacker_none_count=attacker_none_count,
        attacker_id_sample=attacker_id_sample,
        debug=debug_payload,
    )


def _is_trade_kill(
    kill: dict[str, Any],
    prior_kills: list[dict[str, Any]],
    target_steam_id: int,
    target_side: str | None,
    target_name: str | None = None,
) -> bool:
    if target_side is None:
        return False
    if kill.get("attacker") != target_steam_id:
        if not target_name or kill.get("attacker_name") != target_name:
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
    target_name: str | None = None,
) -> bool:
    if target_side is None:
        return False
    if kill.get("victim") != target_steam_id:
        if not target_name or kill.get("victim_name") != target_name:
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
    target_name: str | None = None,
) -> bool:
    if target_side is None:
        return False
    if kill.get("attacker") == target_steam_id:
        return False
    if target_name and kill.get("attacker_name") == target_name:
        return False
    if kill.get("attacker_side") != target_side:
        return False
    kill_time = kill.get("time")
    if kill_time is None:
        return False
    for flash in flashes:
        if flash.get("thrower") != target_steam_id:
            if not target_name or flash.get("thrower_name") != target_name:
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
    target_name: str | None = None,
) -> str | None:
    if round_number is not None and round_number in target_round_sides:
        return target_round_sides[round_number]
    for kill in round_kills:
        if kill.get("attacker") == target_steam_id:
            return kill.get("attacker_side")
        if kill.get("victim") == target_steam_id:
            return kill.get("victim_side")
        if target_name:
            if kill.get("attacker_name") == target_name:
                return kill.get("attacker_side")
            if kill.get("victim_name") == target_name:
                return kill.get("victim_side")
    return None


def aggregate_player_features(
    parsed_demos: list[ParsedDemoEvents],
    target_steam_id: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    target_id = normalize_steamid64(target_steam_id)
    events: list[dict[str, Any]] = []
    rounds_seen: set[tuple[int, int]] = set()
    tick_rate = 64.0
    player_kills = 0
    player_deaths = 0
    player_assists = 0
    player_utility_damage_total = 0.0
    target_attacker_kills = 0
    target_victim_deaths = 0
    target_assists = 0
    attacker_id_examples: list[str] = []
    victim_id_examples: list[str] = []
    attacker_ids_seen: set[str] = set()
    victim_ids_seen: set[str] = set()

    if target_id is None:
        return [], {"rounds": None, "tick_rate": tick_rate}, {
            "player_kills": 0,
            "player_deaths": 0,
            "player_assists": 0,
            "player_util_damage_total": 0.0,
            "utility_damage_per_round": None,
            "player_contacts": 0,
        }

    name_candidates: Counter[str] = Counter()
    steam_match_counts = {"attacker": 0, "victim": 0, "assister": 0}
    for parsed in parsed_demos:
        for kill in parsed.kills:
            if kill.get("attacker_steamid64") == target_id:
                steam_match_counts["attacker"] += 1
                if kill.get("attacker_name"):
                    name_candidates[str(kill.get("attacker_name"))] += 1
            if kill.get("victim_steamid64") == target_id:
                steam_match_counts["victim"] += 1
                if kill.get("victim_name"):
                    name_candidates[str(kill.get("victim_name"))] += 1
            if kill.get("assister_steamid64") == target_id:
                steam_match_counts["assister"] += 1
                if kill.get("assister_name"):
                    name_candidates[str(kill.get("assister_name"))] += 1

    target_name = None
    if name_candidates:
        target_name = name_candidates.most_common(1)[0][0]

    def _match_name(value: Any) -> bool:
        return bool(target_name and value and str(value) == target_name)

    def is_target_attacker(event: dict[str, Any]) -> bool:
        return event.get("attacker_steamid64") == target_id or _match_name(event.get("attacker_name"))

    def is_target_victim(event: dict[str, Any]) -> bool:
        return event.get("victim_steamid64") == target_id or _match_name(event.get("victim_name"))

    def is_target_assister(event: dict[str, Any]) -> bool:
        return event.get("assister_steamid64") == target_id or _match_name(event.get("assister_name"))

    def is_target_thrower(event: dict[str, Any]) -> bool:
        return event.get("thrower") == target_id or _match_name(event.get("thrower_name"))

    for demo_index, parsed in enumerate(parsed_demos, start=1):
        tick_rate = parsed.tick_rate or tick_rate
        if parsed.target_round_sides:
            rounds_seen.update({(demo_index, r) for r in parsed.target_round_sides.keys()})

        by_round: dict[int | None, list[dict[str, Any]]] = {}
        for kill in parsed.kills:
            by_round.setdefault(kill.get("round"), []).append(kill)
            attacker_id_value = kill.get("attacker_steam_id")
            victim_id_value = kill.get("victim_steam_id")
            if attacker_id_value and attacker_id_value not in attacker_ids_seen:
                attacker_ids_seen.add(attacker_id_value)
                if len(attacker_id_examples) < 10:
                    attacker_id_examples.append(attacker_id_value)
            if victim_id_value and victim_id_value not in victim_ids_seen:
                victim_ids_seen.add(victim_id_value)
                if len(victim_id_examples) < 10:
                    victim_id_examples.append(victim_id_value)

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
            target_side = _target_side_for_round(
                round_number,
                parsed.target_round_sides,
                round_kills_sorted,
                target_id,
                target_name,
            )

            round_events: list[dict[str, Any]] = []

            if round_number is not None and not parsed.target_round_sides:
                rounds_seen.add(round_key)

            first_kill = round_kills_sorted[0] if round_kills_sorted else None

            for idx, kill in enumerate(round_kills_sorted):
                kill_time = kill.get("time")
                prior = round_kills_sorted[:idx]
                later = round_kills_sorted[idx + 1 :]
                is_first_duel = first_kill is not None and kill is first_kill

                if is_target_attacker(kill):
                    player_kills += 1
                    target_attacker_kills += 1
                    round_events.append(
                        {
                            "type": "kill",
                            "round": round_number,
                            "time": kill_time,
                            "tick": kill.get("tick"),
                            "round_start_tick": kill.get("round_start_tick"),
                            "time_approx": kill.get("time_approx"),
                            "is_trade_kill": _is_trade_kill(kill, prior, target_id, target_side, target_name),
                            "is_first_duel": is_first_duel,
                            "is_first_duel_win": is_first_duel,
                        }
                    )

                if is_target_victim(kill):
                    player_deaths += 1
                    target_victim_deaths += 1
                    round_events.append(
                        {
                            "type": "death",
                            "round": round_number,
                            "time": kill_time,
                            "tick": kill.get("tick"),
                            "round_start_tick": kill.get("round_start_tick"),
                            "time_approx": kill.get("time_approx"),
                            "was_traded": _is_traded_death(kill, later, target_id, target_side, target_name),
                            "is_first_duel": is_first_duel,
                        }
                    )

                if is_target_assister(kill):
                    player_assists += 1
                    target_assists += 1
                    round_events.append(
                        {
                            "type": "assist",
                            "round": round_number,
                            "time": kill_time,
                            "tick": kill.get("tick"),
                            "round_start_tick": kill.get("round_start_tick"),
                            "time_approx": kill.get("time_approx"),
                            "exclude_from_timing": True,
                        }
                    )

                has_flash_assist = False
                if kill.get("assistedflash") and is_target_assister(kill):
                    has_flash_assist = True
                elif _has_flash_assist(
                    kill,
                    flashes_by_round.get(round_number, []),
                    target_id,
                    target_side,
                    target_name,
                ):
                    has_flash_assist = True
                if has_flash_assist:
                    round_events.append(
                        {
                            "type": "flash_assist",
                            "round": round_number,
                            "time": kill_time,
                            "is_flash_assist": True,
                            "exclude_from_timing": True,
                        }
                    )

            flashes_for_round = flashes_by_round.get(round_number, [])
            for flash in flashes_for_round:
                if not is_target_thrower(flash):
                    continue
                round_events.append(
                    {
                        "type": "flash",
                        "round": round_number,
                        "time": flash.get("time"),
                        "is_friendly_flash": flash.get("is_teamflash"),
                        "exclude_from_timing": True,
                    }
                )

            for dmg in utility_by_round.get(round_number, []):
                if not (dmg.get("attacker_steamid64") == target_id or _match_name(dmg.get("attacker_name"))):
                    continue
                round_events.append(
                    {
                        "type": "damage",
                        "round": round_number,
                        "time": dmg.get("time"),
                        "utility_damage": dmg.get("damage"),
                        "exclude_from_timing": True,
                    }
                )
                player_utility_damage_total += float(dmg.get("damage") or 0)

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
                                "exclude_from_timing": True,
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
                                    "exclude_from_timing": True,
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
    utility_damage_per_round = (
        player_utility_damage_total / rounds_total if rounds_total else None
    )

    meta = {
        "rounds": rounds_total if rounds_total else None,
        "tick_rate": tick_rate,
    }

    debug = {
        "player_kills": player_kills,
        "player_deaths": player_deaths,
        "player_assists": player_assists,
        "player_util_damage_total": player_utility_damage_total,
        "utility_damage_per_round": utility_damage_per_round,
        "player_contacts": player_kills + player_deaths,
        "target_attacker_kills": target_attacker_kills,
        "target_victim_deaths": target_victim_deaths,
        "target_assists": target_assists,
        "attacker_id_examples": attacker_id_examples,
        "victim_id_examples": victim_id_examples,
        "target_name": target_name,
        "name_candidates_count": len(name_candidates),
        "steam_match_counts": steam_match_counts,
    }

    return events, meta, debug


def _slice_label(bounds: tuple[int, int]) -> str:
    return f"{int(bounds[0])}-{int(bounds[1])}"


def _slice_for_time(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    for bounds in HEATMAP_TIME_SLICES:
        start, end = bounds
        if start <= seconds < end:
            return _slice_label(bounds)
    last = HEATMAP_TIME_SLICES[-1]
    return _slice_label(last)


def _rounds_from_events(events: list[dict[str, Any]]) -> int:
    rounds = {event.get("round") for event in events if event.get("round") is not None}
    return len(rounds)


def compute_awareness_before_death(events: list[dict[str, Any]]) -> dict[str, Any]:
    deaths = [event for event in events if event.get("type") == "death" and event.get("time") is not None]
    if not deaths:
        return {
            "aware_deaths": 0,
            "total_deaths": 0,
            "awareness_before_death_rate": None,
            "by_slice": {},
            "lookback_sec": DEATH_AWARENESS_LOOKBACK_SEC,
        }

    contact_events = [
        event
        for event in events
        if event.get("type") in {"kill", "assist", "damage", "flash_assist"}
        and event.get("time") is not None
    ]

    aware_count = 0
    by_slice: dict[str, dict[str, int]] = {}
    for death in deaths:
        death_time = float(death.get("time"))
        death_round = death.get("round")
        slice_label = _slice_for_time(death_time)
        bucket = by_slice.setdefault(slice_label, {"aware": 0, "total": 0})
        bucket["total"] += 1

        aware = any(
            event.get("round") == death_round
            and 0 <= death_time - float(event.get("time")) <= DEATH_AWARENESS_LOOKBACK_SEC
            for event in contact_events
        )
        if aware:
            aware_count += 1
            bucket["aware"] += 1

    total_deaths = len(deaths)
    return {
        "aware_deaths": aware_count,
        "total_deaths": total_deaths,
        "awareness_before_death_rate": (aware_count / total_deaths) * 100 if total_deaths else None,
        "by_slice": by_slice,
        "lookback_sec": DEATH_AWARENESS_LOOKBACK_SEC,
    }


def _quadrant_from_coords(x: float | None, y: float | None) -> str:
    if x is None or y is None:
        return "unknown"
    if x >= 0 and y >= 0:
        return "NE"
    if x < 0 and y >= 0:
        return "NW"
    if x >= 0 and y < 0:
        return "SE"
    return "SW"


def compute_multikill_metrics(events: list[dict[str, Any]]) -> dict[str, Any]:
    kills = [event for event in events if event.get("type") == "kill" and event.get("time") is not None]
    if not kills:
        return {
            "multikill_round_rate": None,
            "multikill_event_rate": None,
            "multikill_events": 0,
            "rounds_with_multikill": 0,
            "by_timing": {"early": 0, "late": 0},
            "by_zone": {},
            "window_sec": MULTIKILL_WINDOW_SEC,
            "early_threshold_sec": MULTIKILL_EARLY_THRESHOLD_SEC,
        }

    rounds_with_multikill = set()
    multikill_events = 0
    timing_breakdown = {"early": 0, "late": 0}
    zone_breakdown: dict[str, int] = {}

    kills_by_round: dict[int | None, list[dict[str, Any]]] = {}
    for kill in kills:
        kills_by_round.setdefault(kill.get("round"), []).append(kill)

    for round_number, round_kills in kills_by_round.items():
        if round_number is None:
            continue
        sorted_kills = sorted(round_kills, key=lambda k: k.get("time") or 0)
        streak: list[dict[str, Any]] = []
        for kill in sorted_kills:
            if not streak:
                streak = [kill]
                continue
            delta = float(kill.get("time") or 0) - float(streak[-1].get("time") or 0)
            if delta <= MULTIKILL_WINDOW_SEC:
                streak.append(kill)
            else:
                if len(streak) >= 2:
                    multikill_events += 1
                    rounds_with_multikill.add(round_number)
                    start_time = float(streak[0].get("time") or 0)
                    key = "early" if start_time <= MULTIKILL_EARLY_THRESHOLD_SEC else "late"
                    timing_breakdown[key] += 1
                    zone = streak[0].get("attacker_place") or _quadrant_from_coords(
                        streak[0].get("attacker_x"),
                        streak[0].get("attacker_y"),
                    )
                    zone_breakdown[zone] = zone_breakdown.get(zone, 0) + 1
                streak = [kill]
        if len(streak) >= 2:
            multikill_events += 1
            rounds_with_multikill.add(round_number)
            start_time = float(streak[0].get("time") or 0)
            key = "early" if start_time <= MULTIKILL_EARLY_THRESHOLD_SEC else "late"
            timing_breakdown[key] += 1
            zone = streak[0].get("attacker_place") or _quadrant_from_coords(
                streak[0].get("attacker_x"),
                streak[0].get("attacker_y"),
            )
            zone_breakdown[zone] = zone_breakdown.get(zone, 0) + 1

    rounds_total = _rounds_from_events(events)
    total_kills = len(kills)
    return {
        "multikill_round_rate": (multikill_events / rounds_total) * 100 if rounds_total else None,
        "multikill_event_rate": (multikill_events / total_kills) * 100 if total_kills else None,
        "multikill_events": multikill_events,
        "rounds_with_multikill": len(rounds_with_multikill),
        "by_timing": timing_breakdown,
        "by_zone": zone_breakdown,
        "window_sec": MULTIKILL_WINDOW_SEC,
        "early_threshold_sec": MULTIKILL_EARLY_THRESHOLD_SEC,
    }


def get_or_build_demo_features(
    profile,
    period: str,
    map_name: str,
    analytics_version: str,
    *,
    force_rebuild: bool = False,
    progress_callback: Callable[[int], None] | None = None,
    progress_start: int = 10,
    progress_end: int = 40,
) -> dict[str, Any]:
    steam_id = _profile_steamid64(profile)
    demo_files = discover_demo_files(profile, period, map_name)
    demos_count = len(demo_files)
    demo_set_hash = compute_demo_set_hash(demo_files) if demo_files else ""

    cache_key = demo_features_key(profile.id, period, map_name, demo_set_hash, analytics_version)
    if not force_rebuild:
        try:
            cached = cache.get(cache_key)
        except Exception:
            cached = None
        if cached:
            return cached

    debug = {
        "demos_count": demos_count,
        "rounds_count": 0,
        "kills_events_count": 0,
        "flash_events_count": 0,
        "util_damage_events_count": 0,
        "missing_time_kills": 0,
        "missing_time_flashes": 0,
        "missing_time_utility": 0,
        "approx_time_kills": 0,
        "attacker_none_count": 0,
        "attacker_id_sample": {"attacker": None, "victim": None},
        "player_kills": 0,
        "player_deaths": 0,
        "player_assists": 0,
        "player_util_damage_total": 0.0,
        "demo_set_hash": demo_set_hash,
        "min_rounds_required": MIN_ROUNDS_REQUIRED,
        "minimal_contacts": MIN_CONTACTS_REQUIRED,
        "raw_kill_columns": None,
        "raw_kill_keys": None,
        "raw_kill_row_sample": None,
        "kill_event_sample": None,
        "time_fields_present": None,
        "tickrate": None,
        "tickrate_assumed": None,
        "round_start_tick_sample": None,
        "attacker_steam_raw_type_sample": None,
        "target_name": None,
        "name_candidates_count": None,
        "steam_match_counts": None,
    }

    if demos_count == 0 or not steam_id:
        meta = {
            "rounds": None,
            "period": period,
            "profile_id": profile.id,
            "min_rounds_required": MIN_ROUNDS_REQUIRED,
            "minimal_contacts": MIN_CONTACTS_REQUIRED,
            "player_kills": 0,
            "player_deaths": 0,
            "flash_events_count": 0,
            "tickrate_assumed": debug.get("tickrate_assumed"),
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
        payload = safe_json(payload)
        try:
            cache.set(cache_key, payload, DEMO_FEATURES_TTL_SECONDS)
        except Exception:
            pass
        return payload

    parsed_demos: list[ParsedDemoEvents] = []
    for index, demo_path in enumerate(demo_files, start=1):
        parsed = parse_demo_events(demo_path, target_steam_id=steam_id)
        parsed_demos.append(parsed)
        debug["kills_events_count"] += len(parsed.kills)
        debug["flash_events_count"] += parsed.flash_events_count
        debug["util_damage_events_count"] += len(parsed.utility_damage)
        debug["rounds_count"] += len(parsed.rounds_in_demo)
        debug["missing_time_kills"] += parsed.missing_time_kills
        debug["missing_time_flashes"] += parsed.missing_time_flashes
        debug["missing_time_utility"] += parsed.missing_time_utility
        debug["approx_time_kills"] += parsed.approx_time_kills
        debug["attacker_none_count"] += parsed.attacker_none_count
        if parsed.attacker_id_sample.get("attacker") and not debug["attacker_id_sample"].get("attacker"):
            debug["attacker_id_sample"]["attacker"] = parsed.attacker_id_sample["attacker"]
        if parsed.attacker_id_sample.get("victim") and not debug["attacker_id_sample"].get("victim"):
            debug["attacker_id_sample"]["victim"] = parsed.attacker_id_sample["victim"]
        for key in (
            "raw_kill_columns",
            "raw_kill_keys",
            "raw_kill_row_sample",
            "kill_event_sample",
            "time_fields_present",
            "tickrate",
            "tickrate_assumed",
            "round_start_tick_sample",
            "attacker_steam_raw_type_sample",
        ):
            if parsed.debug.get(key) is not None and debug.get(key) in (None, [], {}):
                debug[key] = parsed.debug.get(key)
        if progress_callback:
            span = max(progress_end - progress_start, 1)
            progress = progress_start + int((index / max(demos_count, 1)) * span)
            progress_callback(progress)

    events, meta, player_debug = aggregate_player_features(parsed_demos, steam_id)
    debug.update(
        {
            "player_kills": player_debug.get("player_kills", 0),
            "player_deaths": player_debug.get("player_deaths", 0),
            "player_assists": player_debug.get("player_assists", 0),
            "player_util_damage_total": player_debug.get("player_util_damage_total", 0.0),
            "target_attacker_kills": player_debug.get("target_attacker_kills", 0),
            "target_victim_deaths": player_debug.get("target_victim_deaths", 0),
            "target_assists": player_debug.get("target_assists", 0),
            "attacker_id_examples": player_debug.get("attacker_id_examples", []),
            "victim_id_examples": player_debug.get("victim_id_examples", []),
            "target_name": player_debug.get("target_name"),
            "name_candidates_count": player_debug.get("name_candidates_count"),
            "steam_match_counts": player_debug.get("steam_match_counts"),
        }
    )
    rounds_total = meta.get("rounds") or 0

    awareness = compute_awareness_before_death(events)
    multikill = compute_multikill_metrics(events)

    kills = debug.get("player_kills", 0) or 0
    deaths = debug.get("player_deaths", 0) or 0
    assists = debug.get("player_assists", 0) or 0
    kda_ratio = (kills + assists) / deaths if deaths else None
    kda = {
        "kills": kills,
        "deaths": deaths,
        "assists": assists,
        "kda_ratio": kda_ratio,
        "assists_per_round": assists / rounds_total if rounds_total else None,
    }

    if rounds_total < MIN_ROUNDS_REQUIRED:
        meta = {
            "rounds": None,
            "period": period,
            "profile_id": profile.id,
            "min_rounds_required": MIN_ROUNDS_REQUIRED,
            "minimal_contacts": MIN_CONTACTS_REQUIRED,
            "player_kills": debug.get("player_kills", 0),
            "player_deaths": debug.get("player_deaths", 0),
            "flash_events_count": debug.get("flash_events_count", 0),
            "tickrate_assumed": debug.get("tickrate_assumed"),
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
            "tickrate_assumed": debug.get("tickrate_assumed"),
            "min_rounds_required": MIN_ROUNDS_REQUIRED,
            "minimal_contacts": MIN_CONTACTS_REQUIRED,
            "player_kills": debug.get("player_kills", 0),
            "player_deaths": debug.get("player_deaths", 0),
            "flash_events_count": debug.get("flash_events_count", 0),
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
        "awareness_before_death": awareness,
        "multikill": multikill,
        "kda": kda,
        "debug": debug,
        "rounds_total": rounds_total,
        "demos_count": demos_count,
        "demo_set_hash": demo_set_hash,
        "insufficient_rounds": insufficient_rounds,
    }

    payload = safe_json(payload)
    try:
        cache.set(cache_key, payload, DEMO_FEATURES_TTL_SECONDS)
    except Exception:
        pass
    if progress_callback:
        progress_callback(progress_end)
    return payload
