from __future__ import annotations

import hashlib
import json
import logging
import math
from collections import Counter
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from dataclasses import dataclass, field
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
from faceit_analytics.services.time_buckets import get_time_bucket_presets, time_bucket_for_seconds
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
ENTRY_SUPPORT_RADIUS = float(getattr(settings, "ENTRY_SUPPORT_RADIUS", 450.0))
ENTRY_SUPPORT_WINDOW_SECONDS = float(getattr(settings, "ENTRY_SUPPORT_WINDOW_SECONDS", 4.0))
ENTRY_HOLD_DELAY_SECONDS = float(getattr(settings, "ENTRY_HOLD_DELAY_SECONDS", 12.0))
ENTRY_PHASE_MAX_SECONDS = float(getattr(settings, "ENTRY_PHASE_MAX_SECONDS", 35.0))
PROXIMITY_RADIUS = float(getattr(settings, "PROXIMITY_RADIUS", 500.0))
PROXIMITY_WINDOW_SECONDS = float(getattr(settings, "PROXIMITY_WINDOW_SEC", 3.0))
PROXIMITY_SAMPLE_STEP_TICKS = int(getattr(settings, "PROXIMITY_SAMPLE_STEP_TICKS", 8))
PUSH_DISTANCE = float(getattr(settings, "PUSH_DISTANCE", 1600.0))
STEAMID64_MIN_VALUE = 7_000_000_000_000_000
SIDE_ROLE_SAMPLE_SECONDS = 30.0
SIDE_ROLE_MIN_SAMPLES = 6
SIDE_ROLE_APPROX_SAMPLES = 20
SITE_DISTANCE_RADIUS = 650.0
POSTPLANT_DISTANCE_RADIUS = 600.0
R_SITE = 650.0
R_PLANT = 600.0
R_FAR = 900.0
ENTRY_WINDOW_SEC = 30.0
MIN_SEC_IN_AREA = 4.0
ROUND_COL_CANDIDATES = ["round", "round_num", "round_number", "roundNum", "roundNumber", "roundnum"]
TICK_COL_CANDIDATES = ["tick", "ticks", "tick_num"]
START_TICK_CANDIDATES = [
    "start_tick",
    "round_start_tick",
    "freeze_end_tick",
    "startTick",
    "roundStartTick",
    "freezeEndTick",
    "freezeTimeEndTick",
    "freeze_time_end_tick",
    "freezeTimeEnd",
    "freezeEnd",
]
START_TIME_CANDIDATES = [
    "start_time",
    "round_start_time",
    "freeze_end_time",
    "startTime",
    "roundStartTime",
    "freezeEndTime",
    "freezeTimeEndTime",
]

KILLS_STEAMID_COLUMNS = ["attacker_steamid", "victim_steamid", "assister_steamid"]
UTIL_DAMAGE_STEAMID_COLUMNS = ["attacker_steamid", "victim_steamid"]

logger = logging.getLogger(__name__)


def first_not_none(*vals: Any) -> Any:
    for val in vals:
        if val is not None:
            return val
    return None


def safe_text(value: Any) -> str | None:
    return _safe_str(value)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except Exception:
        pass
    return False


def _safe_str(value: Any) -> str | None:
    if _is_missing(value):
        return None
    text = str(value).strip()
    return text if text else None


def _safe_bool(value: Any) -> bool:
    if _is_missing(value):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    try:
        return bool(value)
    except Exception:
        return False


def _is_valid_steamid64(value: Any) -> bool:
    steam_id = normalize_steamid64(value)
    if steam_id is None:
        return False
    return steam_id >= STEAMID64_MIN_VALUE


def _has_true_assist(kill: dict[str, Any]) -> bool:
    assister = kill.get("assister")
    attacker = kill.get("attacker")
    if not _is_valid_steamid64(assister):
        return False
    if not _is_valid_steamid64(attacker):
        return False
    return int(assister) != int(attacker)


def _align_round_number(rn: int | None, mapping: dict[int, Any]) -> int | None:
    if rn is None:
        return None
    if rn in mapping:
        return rn
    if isinstance(rn, int):
        if (rn - 1) in mapping:
            return rn - 1
        if (rn + 1) in mapping:
            return rn + 1
    return rn

def _read_parquet_with_steamid_strings(parquet_path: Path, steam_cols: Iterable[str]) -> pd.DataFrame:
    table = pq.read_table(parquet_path)
    for col in steam_cols:
        if col in table.column_names:
            index = table.schema.get_field_index(col)
            table = table.set_column(index, col, table[col].cast(pa.string()))
    return table.to_pandas(use_pyarrow_extension_array=True)


def _ensure_steamid_string_cols(df: pd.DataFrame, steam_cols: Iterable[str]) -> pd.DataFrame:
    for col in steam_cols:
        if col in df.columns:
            df[col] = df[col].astype("string")
    return df


def _load_demo_dataframe(value: Any, steam_cols: Iterable[str]) -> pd.DataFrame | None:
    """Load a demo table-like object into pandas without losing SteamID64 precision.

    AWPY/Polars often stores steamid columns as u64. Converting u64 to pandas can
    silently cast to float64 (losing precision), which breaks SteamID matching.
    We cast steamid columns to Int64 in Polars before conversion.
    """
    if value is None:
        return None
    if isinstance(value, pd.DataFrame):
        return _ensure_steamid_string_cols(value, steam_cols)
    if isinstance(value, (str, Path)):
        parquet_path = Path(value)
        if parquet_path.suffix == ".parquet" and parquet_path.exists():
            return _read_parquet_with_steamid_strings(parquet_path, steam_cols)

    # Polars DataFrame (awpy uses polars)
    try:
        import polars as pl  # type: ignore
        if isinstance(value, pl.DataFrame):
            cols = []
            for c in steam_cols:
                if c in value.columns:
                    cols.append(pl.col(c).cast(pl.Int64, strict=False))
            if cols:
                value = value.with_columns(cols)
            # Keep integer dtypes via Arrow extension arrays when possible
            try:
                df = value.to_pandas(use_pyarrow_extension_array=True)
            except TypeError:
                df = value.to_pandas()
            return _ensure_steamid_string_cols(df, steam_cols)
    except Exception:
        pass

    if hasattr(value, "to_pandas"):
        try:
            df = value.to_pandas(use_pyarrow_extension_array=True)
        except TypeError:
            df = value.to_pandas()
        except Exception:
            df = value.to_pandas()
        return _ensure_steamid_string_cols(df, steam_cols)

    return None


def _first_non_none_attr(obj: Any, names: list[str]) -> Any:
    """Return the first attribute value that is not None without triggering truthiness on DataFrames."""
    for name in names:
        try:
            value = getattr(obj, name, None)
        except Exception:
            value = None
        if value is not None:
            return value
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
    tick_positions_by_round: dict[int, list[dict[str, Any]]] = field(default_factory=dict)
    bomb_plants_by_round: dict[int, dict[str, Any]] = field(default_factory=dict)
    bomb_events_by_round: dict[int, list[dict[str, Any]]] = field(default_factory=dict)
    map_name: str | None = None
    missing_time_bomb: int = 0
    approx_time_bomb: int = 0


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


def normalize_bombsite(value: Any) -> str | None:
    if value is None:
        return None
    value_str = str(value).strip().lower()
    if not value_str:
        return None
    value_str = value_str.replace("bombsite", "").replace("_", "").replace(" ", "")
    if value_str.startswith("site"):
        value_str = value_str.replace("site", "")
    if value_str.endswith("a") or value_str == "a":
        return "A"
    if value_str.endswith("b") or value_str == "b":
        return "B"
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
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _cell(row: "pd.Series", col: str | None) -> Any | None:
    """Safely fetch a cell value from a pandas row (treat pandas.NA/NaN as None)."""
    if not col:
        return None
    try:
        value = row.get(col)
    except Exception:
        return None
    if _is_missing(value):
        return None
    return value


def _cell_str(row: "pd.Series", col: str | None) -> str | None:
    return _safe_str(_cell(row, col))


def normalize_steamid64(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, Decimal):
        try:
            return int(value)
        except (InvalidOperation, ValueError, TypeError):
            return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return None
        try:
            dec_value = Decimal(str(value))
        except (InvalidOperation, ValueError, TypeError):
            return None
        if dec_value.is_nan():
            return None
        try:
            return int(dec_value)
        except (InvalidOperation, ValueError, TypeError):
            return None
    if isinstance(value, str):
        try:
            dec_value = Decimal(value.strip())
        except (InvalidOperation, ValueError, TypeError):
            return None
        if dec_value.is_nan():
            return None
        try:
            return int(dec_value)
        except (InvalidOperation, ValueError, TypeError):
            return None
    if hasattr(value, "item"):
        return normalize_steamid64(value.item())
    try:
        return normalize_steamid64(str(value))
    except Exception:
        return None


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
        if value is not None and pd.notna(value):
            return float(value)
    header = first_not_none(getattr(demo, "header", None), {})
    for key in ("tickrate", "tick_rate", "tickRate"):
        if key in header and header[key] is not None and pd.notna(header[key]):
            return float(header[key])
    return 64.0


def _build_round_meta(
    rounds_df: pd.DataFrame,
    ticks_df: pd.DataFrame | None = None,
) -> tuple[dict[int, int], dict[int, float], dict[int, str | None], set[int]]:
    round_start_ticks: dict[int, int] = {}
    round_start_times: dict[int, float] = {}
    round_winners: dict[int, str | None] = {}
    rounds_in_demo: set[int] = set()

    if rounds_df is None or rounds_df.empty:
        rounds_df = None

    round_col = _pick_column(rounds_df, ROUND_COL_CANDIDATES) if rounds_df is not None else None
    start_tick_col = _pick_column(rounds_df, START_TICK_CANDIDATES) if rounds_df is not None else None
    start_time_col = _pick_column(rounds_df, START_TIME_CANDIDATES) if rounds_df is not None else None
    winner_col = _pick_column(rounds_df, ["winner", "winning_side", "round_winner"]) if rounds_df is not None else None

    if rounds_df is not None and not rounds_df.empty:
        for _, row in rounds_df.iterrows():
            round_number = _safe_int(_cell(row, round_col)) if round_col else None
            if round_number is None:
                continue
            rounds_in_demo.add(round_number)
            if start_tick_col:
                start_tick = _safe_int(_cell(row, start_tick_col))
                if start_tick is not None:
                    round_start_ticks[round_number] = start_tick
            if start_time_col:
                start_time = _safe_float(_cell(row, start_time_col))
                if start_time is not None:
                    round_start_times[round_number] = start_time
            if winner_col:
                round_winners[round_number] = _normalize_side(_cell(row, winner_col))

    if (start_tick_col is None or not round_start_ticks) and ticks_df is not None and not ticks_df.empty:
        tick_round_col = _pick_column(ticks_df, ROUND_COL_CANDIDATES)
        tick_col = _pick_column(ticks_df, TICK_COL_CANDIDATES)
        if tick_round_col and tick_col:
            tick_rounds = pd.to_numeric(ticks_df[tick_round_col], errors="coerce")
            tick_values = pd.to_numeric(ticks_df[tick_col], errors="coerce")
            grouped = pd.DataFrame({"round": tick_rounds, "tick": tick_values}).dropna()
            if not grouped.empty:
                start_ticks = grouped.groupby("round", as_index=False)["tick"].min()
                for _, row in start_ticks.iterrows():
                    round_number = _safe_int(row.get("round"))
                    start_tick = _safe_int(row.get("tick"))
                    if round_number is None or start_tick is None:
                        continue
                    round_start_ticks[round_number] = start_tick
                    rounds_in_demo.add(round_number)

    return round_start_ticks, round_start_times, round_winners, rounds_in_demo


def _round_time_seconds(
    row: pd.Series,
    round_number: int | None,
    round_start_ticks: dict[int, int],
    round_start_times: dict[int, float],
    tick_rate: float,
) -> float | None:
    for key in ("round_time", "time_from_round_start", "time_from_start", "roundTime"):
        value = _cell(row, key)
        if value is not None:
            return _safe_float(value)

    time_value = None
    for key in ("time", "seconds", "timestamp"):
        value = _cell(row, key)
        if value is not None:
            time_value = _safe_float(value)
            break

    mapping = round_start_times if round_start_times else round_start_ticks
    aligned_round = _align_round_number(round_number, mapping)

    if aligned_round is not None and time_value is not None:
        start_time = round_start_times.get(aligned_round)
        if start_time is not None:
            return time_value - start_time

    tick_value = None
    for key in ("tick", "ticks", "tick_num"):
        value = _cell(row, key)
        if value is not None:
            tick_value = _safe_int(value)
            break

    if aligned_round is not None and tick_value is not None:
        start_tick = round_start_ticks.get(aligned_round)
        if start_tick is not None and tick_rate:
            return (tick_value - start_tick) / tick_rate

    return None


def _extract_player_round_sides_from_ticks(ticks_df: pd.DataFrame, target_steam_id: str) -> dict[int, str]:
    if ticks_df is None or ticks_df.empty:
        return {}
    steamid_col = _pick_column(ticks_df, ["steamid", "steamID", "player_steamid", "playerSteamID"])
    round_col = _pick_column(ticks_df, ROUND_COL_CANDIDATES)
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


def _extract_player_round_sides(demo: Demo, target_steam_id: str) -> dict[int, str]:
    ticks_df = _load_demo_dataframe(getattr(demo, "ticks", None), ["steamid", "player_steamid", "playerSteamID"])
    return _extract_player_round_sides_from_ticks(ticks_df, target_steam_id)


def _downsample_ticks_df(ticks_df: pd.DataFrame, tick_col: str | None, step: int) -> pd.DataFrame:
    if not tick_col or step <= 1 or ticks_df.empty:
        return ticks_df
    tick_numeric = pd.to_numeric(ticks_df[tick_col], errors="coerce")
    return ticks_df[tick_numeric.fillna(0).astype(int) % step == 0]


def _extract_tick_positions(
    ticks_df: pd.DataFrame,
    target_round_sides: dict[int, str],
    target_steam_id: str | None,
    round_start_ticks: dict[int, int],
    round_start_times: dict[int, float],
    tick_rate: float,
) -> tuple[dict[int, list[dict[str, Any]]], int]:
    if ticks_df is None or ticks_df.empty:
        return {}, 0
    if not target_round_sides:
        return {}, 0
    steamid_col = _pick_column(ticks_df, ["steamid", "steamID", "player_steamid", "playerSteamID"])
    round_col = _pick_column(ticks_df, ROUND_COL_CANDIDATES)
    side_col = _pick_column(ticks_df, ["side", "player_side", "playerSide"])
    x_col = _pick_column(ticks_df, ["X", "x", "player_X", "player_x"])
    y_col = _pick_column(ticks_df, ["Y", "y", "player_Y", "player_y"])
    tick_col = _pick_column(ticks_df, TICK_COL_CANDIDATES)
    place_col = _pick_column(ticks_df, ["place", "place_name", "placeName", "area_name", "areaName"])
    health_col = _pick_column(ticks_df, ["health", "hp", "player_health", "playerHP"])
    if not steamid_col or not round_col or not side_col or not x_col or not y_col:
        return {}, 0
    ticks_df = _downsample_ticks_df(ticks_df, tick_col, PROXIMITY_SAMPLE_STEP_TICKS)
    positions_by_round: dict[int, list[dict[str, Any]]] = {}
    target_id = normalize_steamid64(target_steam_id) if target_steam_id else None
    missing_t_round = 0
    for _, row in ticks_df.iterrows():
        round_number = _safe_int(row.get(round_col))
        if round_number is None:
            continue
        round_side = target_round_sides.get(round_number)
        side_value = _normalize_side(row.get(side_col))
        if not round_side or not side_value or round_side != side_value:
            continue
        steam_id = normalize_steamid64(row.get(steamid_col))
        if steam_id is None:
            continue
        t_round = _round_time_seconds(row, round_number, round_start_ticks, round_start_times, tick_rate)
        if t_round is None:
            missing_t_round += 1
            continue
        x_val = _safe_float(row.get(x_col))
        y_val = _safe_float(row.get(y_col))
        if x_val is None or y_val is None:
            continue
        health_val = _safe_float(row.get(health_col)) if health_col else None
        if health_val is not None and health_val <= 0:
            continue
        positions_by_round.setdefault(round_number, []).append(
            {
                "time": float(t_round),
                "tick": _safe_int(row.get(tick_col)) if tick_col else None,
                "steamid": steam_id,
                "is_target": bool(target_id and steam_id == target_id),
                "side": side_value,
                "x": float(x_val),
                "y": float(y_val),
                "place": _cell_str(row, place_col),
                "health": float(health_val) if health_val is not None else None,
            }
        )
    return positions_by_round, missing_t_round


def _extract_bomb_events(
    demo: Demo,
    tick_rate: float,
    round_start_ticks: dict[int, int],
    round_start_times: dict[int, float],
    rounds_df: pd.DataFrame | None,
    map_name: str | None = None,
) -> tuple[dict[int, dict[str, Any]], dict[int, list[dict[str, Any]]], dict[str, int], int, int]:
    bomb_df = _load_demo_dataframe(
        _first_non_none_attr(demo, ["bomb", "bombs"]),
        [],
    )
    plants: dict[int, dict[str, Any]] = {}
    events_by_round: dict[int, list[dict[str, Any]]] = {}
    counts = {"plants": 0, "defuses": 0, "explodes": 0}
    missing_time_bomb = 0
    approx_time_bomb = 0

    if bomb_df is not None and not bomb_df.empty:
        event_col = _pick_column(bomb_df, ["event", "bomb_event", "type", "action"])
        round_col = _pick_column(bomb_df, ROUND_COL_CANDIDATES)
        tick_col = _pick_column(bomb_df, TICK_COL_CANDIDATES)
        time_col = _pick_column(bomb_df, ["time", "seconds", "round_time"])
        site_col = _pick_column(bomb_df, ["bombsite", "bomb_site", "site"])
        steamid_col = _pick_column(bomb_df, ["steamid", "steam_id", "player_steamid", "playerSteamID"])
        x_col = _pick_column(bomb_df, ["X", "x", "pos_x"])
        y_col = _pick_column(bomb_df, ["Y", "y", "pos_y"])
        z_col = _pick_column(bomb_df, ["Z", "z", "pos_z"])
        zone_config = _load_zone_config(map_name)
        for _, row in bomb_df.iterrows():
            event_raw = _cell(row, event_col) if event_col else None
            event_value = _safe_str(event_raw).lower() if event_raw is not None else ""
            if not event_value:
                continue
            if "plant" in event_value:
                event_key = "plant"
            elif "defuse" in event_value:
                event_key = "defuse"
            elif "explode" in event_value:
                event_key = "explode"
            elif "pickup" in event_value:
                event_key = "pickup"
            elif "drop" in event_value:
                event_key = "drop"
            else:
                continue

            round_number = _safe_int(row.get(round_col)) if round_col else None
            if round_number is None:
                continue
            tick_value = _safe_int(row.get(tick_col)) if tick_col else None
            t_round = _safe_float(row.get(time_col)) if time_col else None
            if t_round is None:
                t_round = _round_time_seconds(row, round_number, round_start_ticks, round_start_times, tick_rate)
            if t_round is None and tick_value is not None:
                start_tick = round_start_ticks.get(round_number)
                if start_tick is not None and tick_rate:
                    t_round = max((tick_value - start_tick) / tick_rate, 0.0)
                    approx_time_bomb += 1
            if t_round is None:
                missing_time_bomb += 1

            site_value = _cell_str(row, site_col)
            site_raw = normalize_bombsite(site_value)
            site_calc = _zone_from_coords(
                map_name,
                _safe_float(row.get(x_col)) if x_col else None,
                _safe_float(row.get(y_col)) if y_col else None,
                zone_config,
            )
            site_calc = site_calc if site_calc in {"A", "B"} else None
            site_value = site_calc or site_raw
            event_payload = {
                "event": event_key,
                "tick": tick_value,
                "time": t_round,
                "x": _safe_float(row.get(x_col)) if x_col else None,
                "y": _safe_float(row.get(y_col)) if y_col else None,
                "z": _safe_float(row.get(z_col)) if z_col else None,
                "site": site_value,
                "site_raw": site_raw,
                "site_calc": site_calc,
                "steamid": normalize_steamid64(row.get(steamid_col)) if steamid_col else None,
            }
            events_by_round.setdefault(round_number, []).append(event_payload)

            if event_key == "plant":
                counts["plants"] += 1
                current = plants.get(round_number)
                if current is None or (event_payload.get("time") is not None and current.get("time") is None):
                    plants[round_number] = {
                        "time": event_payload.get("time"),
                        "tick": event_payload.get("tick"),
                        "x": event_payload.get("x"),
                        "y": event_payload.get("y"),
                        "z": event_payload.get("z"),
                        "site": event_payload.get("site"),
                        "site_raw": event_payload.get("site_raw"),
                        "site_calc": event_payload.get("site_calc"),
                        "approx": event_payload.get("time") is None or event_payload.get("site") is None,
                    }
            elif event_key == "defuse":
                counts["defuses"] += 1
            elif event_key == "explode":
                counts["explodes"] += 1

    return plants, events_by_round, counts, missing_time_bomb, approx_time_bomb


def parse_demo_events(dem_path: Path, target_steam_id: str | None = None) -> ParsedDemoEvents:
    demo = Demo(str(dem_path), verbose=False)
    demo.parse()

    tick_rate = _tick_rate_from_demo(demo)
    tick_rate_approx = True
    if (
        getattr(demo, "tickrate", None) is not None
        or getattr(demo, "tick_rate", None) is not None
        or getattr(demo, "tickRate", None) is not None
    ):
        tick_rate_approx = False
    header = first_not_none(getattr(demo, "header", None), {})
    if header.get("tickrate") or header.get("tick_rate") or header.get("tickRate"):
        tick_rate_approx = False

    rounds_df = _load_demo_dataframe(getattr(demo, "rounds", None), [])
    ticks_df = _load_demo_dataframe(getattr(demo, "ticks", None), ["steamid", "player_steamid", "playerSteamID"])
    round_start_ticks, round_start_times, round_winners, rounds_in_demo = _build_round_meta(
        rounds_df,
        ticks_df=ticks_df,
    )
    map_name = header.get("map_name") if isinstance(header, dict) else None

    kills_df = _load_demo_dataframe(getattr(demo, "kills", None), KILLS_STEAMID_COLUMNS)
    flashes_df = _load_demo_dataframe(getattr(demo, "flashes", None), [])
    util_damage_df = _load_demo_dataframe(
        _first_non_none_attr(demo, ["util_damage", "utility_damage"]),
        UTIL_DAMAGE_STEAMID_COLUMNS,
    )
    damages_df = util_damage_df if util_damage_df is not None and not util_damage_df.empty else _load_demo_dataframe(
        getattr(demo, "damages", None),
        UTIL_DAMAGE_STEAMID_COLUMNS,
    )

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
        debug_payload["steamid_column_dtypes"] = {
            col: str(kills_df[col].dtype)
            for col in KILLS_STEAMID_COLUMNS
            if col in kills_df.columns
        }
        raw_kill_row_sample: dict[str, Any] | None = None
        if raw_kill_columns:
            sample_row = kills_df.iloc[0].to_dict()
            raw_kill_row_sample = {key: sample_row.get(key) for key in list(sample_row.keys())[:20]}
            debug_payload["raw_kill_row_sample"] = raw_kill_row_sample
        time_fields = {"time", "seconds", "round_time", "tick", "game_time", "clock_time"}
        debug_payload["time_fields_present"] = [
            column for column in raw_kill_columns if column.lower() in time_fields
        ]
        round_col = _pick_column(kills_df, ROUND_COL_CANDIDATES)
        tick_col = _pick_column(kills_df, TICK_COL_CANDIDATES)
        attacker_col = _pick_column(
            kills_df,
            [
                "killer_steamid",
                "attacker_steamid",
                "attackerSteamID64",
                "killerSteamID64",
                "killerSteamID",
                "killerSteamId",
                "attackerSteamID",
                "attackerSteamId",
                "killer",
                "attacker",
            ],
        )
        victim_col = _pick_column(
            kills_df,
            [
                "victim_steamid",
                "victimSteamID64",
                "victimSteamID",
                "victimSteamId",
                "victim",
            ],
        )
        assister_col = _pick_column(
            kills_df,
            [
                "assister_steamid",
                "assistant_steamid",
                "assisterSteamID64",
                "assistantSteamID64",
                "assisterSteamID",
                "assisterSteamId",
                "assistantSteamID",
                "assistantSteamId",
                "assister",
                "assist",
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
        attacker_x_col = _pick_column(kills_df, ["attacker_X", "attacker_x", "attackerX"])
        attacker_y_col = _pick_column(kills_df, ["attacker_Y", "attacker_y", "attackerY"])
        victim_x_col = _pick_column(kills_df, ["victim_X", "victim_x", "victimX"])
        victim_y_col = _pick_column(kills_df, ["victim_Y", "victim_y", "victimY"])

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
            attacker_name = _cell_str(row, attacker_name_col)
            victim_name = _cell_str(row, victim_name_col)
            assister_name = _cell_str(row, assister_name_col)
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
                "assistedflash": _safe_bool(_cell(row, assistedflash_col)) if assistedflash_col else False,
                "attacker_place": _cell_str(row, attacker_place_col),
                "victim_place": _cell_str(row, victim_place_col),
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
        round_col = _pick_column(flashes_df, ROUND_COL_CANDIDATES)
        tick_col = _pick_column(flashes_df, TICK_COL_CANDIDATES)
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
                _cell_str(row, thrower_name_col)
            )
            blinded_name = (
                _cell_str(row, blinded_name_col)
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
                    "is_teamflash": _safe_bool(thrower_side and blinded_side and thrower_side == blinded_side),
                }
            )
        flash_events_count = len(flashes)
    elif assistedflash_kill_count:
        flash_events_count = assistedflash_kill_count

    utility_damage: list[dict[str, Any]] = []
    missing_time_utility = 0
    if damages_df is not None and not damages_df.empty:
        raw_damages_df = damages_df.copy()
        debug_payload["utility_steamid_column_dtypes"] = {
            col: str(damages_df[col].dtype)
            for col in UTIL_DAMAGE_STEAMID_COLUMNS
            if col in damages_df.columns
        }
        round_col = _pick_column(damages_df, ROUND_COL_CANDIDATES)
        tick_col = _pick_column(damages_df, TICK_COL_CANDIDATES)
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
            tmp_weapon = safe_text(_cell(row, weapon_col))
            weapon = tmp_weapon.lower() if tmp_weapon is not None else ""
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
            attacker_name = _cell_str(row, attacker_name_col)
            victim_name = _cell_str(row, victim_name_col)
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

    target_round_sides = _extract_player_round_sides_from_ticks(ticks_df, target_steam_id) if target_steam_id else {}
    tick_positions_by_round, ticks_missing_t_round = _extract_tick_positions(
        ticks_df,
        target_round_sides,
        target_steam_id,
        round_start_ticks,
        round_start_times,
        tick_rate,
    )
    debug_payload["ticks_missing_t_round"] = ticks_missing_t_round
    (
        bomb_plants_by_round,
        bomb_events_by_round,
        bomb_debug_counts,
        missing_time_bomb,
        approx_time_bomb,
    ) = _extract_bomb_events(
        demo,
        tick_rate,
        round_start_ticks,
        round_start_times,
        rounds_df,
        map_name,
    )
    debug_payload["bomb_event_counts"] = bomb_debug_counts
    logger.debug(
        "Parsed demo=%s tick_rate=%s approx=%s round_start_tick_sample=%s steamid_dtypes=%s",
        dem_path,
        tick_rate,
        tick_rate_approx,
        debug_payload.get("round_start_tick_sample"),
        debug_payload.get("steamid_column_dtypes"),
    )
    return ParsedDemoEvents(
        kills=kills,
        flashes=flashes,
        utility_damage=utility_damage,
        flash_events_count=flash_events_count,
        round_winners=round_winners,
        target_round_sides=target_round_sides,
        rounds_in_demo=rounds_in_demo,
        tick_positions_by_round=tick_positions_by_round,
        bomb_plants_by_round=bomb_plants_by_round,
        bomb_events_by_round=bomb_events_by_round,
        map_name=map_name,
        missing_time_bomb=missing_time_bomb,
        approx_time_bomb=approx_time_bomb,
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


def _entry_support_nearby(
    entry_time: float | None,
    entry_x: float | None,
    entry_y: float | None,
    teammate_positions: list[tuple[float, float, float]],
) -> bool:
    if entry_time is None or entry_x is None or entry_y is None:
        return False
    for teammate_time, teammate_x, teammate_y in teammate_positions:
        if abs(entry_time - teammate_time) > ENTRY_SUPPORT_WINDOW_SECONDS:
            continue
        dx = entry_x - teammate_x
        dy = entry_y - teammate_y
        if (dx * dx + dy * dy) <= (ENTRY_SUPPORT_RADIUS * ENTRY_SUPPORT_RADIUS):
            return True
    return False


def _nearby_teammates_count(
    kill_tick: int | None,
    kill_x: float | None,
    kill_y: float | None,
    positions: list[dict[str, Any]],
    target_id: int | None,
    tick_rate: float,
) -> tuple[int | None, bool]:
    if kill_tick is None or kill_x is None or kill_y is None or not tick_rate:
        return None, True
    if not positions:
        return None, True
    window_ticks = int(tick_rate * PROXIMITY_WINDOW_SECONDS)
    window_start = int(kill_tick) - window_ticks
    nearby_ids: set[int] = set()
    window_has_rows = False
    for pos in positions:
        pos_tick = pos.get("tick")
        if pos_tick is None:
            continue
        if pos_tick < window_start or pos_tick > kill_tick:
            continue
        window_has_rows = True
        if target_id is not None and pos.get("steamid") == target_id:
            continue
        if pos.get("health") is not None and float(pos.get("health") or 0) <= 0:
            continue
        dx = float(kill_x) - float(pos.get("x") or 0)
        dy = float(kill_y) - float(pos.get("y") or 0)
        if (dx * dx + dy * dy) <= (PROXIMITY_RADIUS * PROXIMITY_RADIUS):
            steamid_value = pos.get("steamid")
            if steamid_value is not None:
                nearby_ids.add(int(steamid_value))
    if not window_has_rows:
        return None, True
    return len(nearby_ids), False


def _support_category(
    has_true_assist: bool,
    has_flash_assist: bool,
    nearby_count: int | None,
) -> str:
    if has_true_assist:
        return "assist"
    if has_flash_assist:
        return "flash"
    if nearby_count is None:
        return "unknown"
    if nearby_count >= 2:
        return "group"
    if nearby_count >= 1:
        return "partner"
    return "solo"


def _objective_site_from_kills(
    kills: list[dict[str, Any]],
    config: dict[str, Any],
) -> str | None:
    site_places = config.get("site_places") or {}
    for kill in sorted(kills, key=lambda k: k.get("time") or 0):
        place = kill.get("attacker_place")
        zone = _zone_from_place(place, config)
        if zone in {"A", "B"}:
            return zone
        if place:
            for site_key in ("A", "B"):
                site_data = site_places.get(site_key) or {}
                if _place_matches(place, site_data.get("core") or []) or _place_matches(
                    place, site_data.get("buffer") or []
                ):
                    return site_key
    return None


def _is_in_objective_area(
    place: str | None,
    objective_site: str | None,
    config: dict[str, Any],
    map_name: str | None = None,
    x: float | None = None,
    y: float | None = None,
) -> bool:
    if not place or objective_site not in {"A", "B"} or not config:
        return False
    site_places = (config.get("site_places") or {}).get(objective_site) or {}
    if _place_matches(place, site_places.get("core") or []) or _place_matches(place, site_places.get("buffer") or []):
        return True
    zone = _zone_from_place(place, config)
    if zone == objective_site:
        return True
    if map_name and x is not None and y is not None:
        zone_from_coords = _zone_from_coords(map_name, x, y, config)
        if zone_from_coords == objective_site:
            return True
    if zone == "ENTRY":
        return _place_matches(place, site_places.get("core") or []) or _place_matches(
            place, site_places.get("buffer") or []
        )
    return False


def _kill_phase(
    kill_side: str | None,
    kill_time: float | None,
    kill_tick: int | None,
    kill_x: float | None,
    kill_y: float | None,
    kill_place: str | None,
    map_name: str | None,
    bomb_info: dict[str, Any] | None,
    objective_site: str | None,
    anchor_time: float | None,
    config: dict[str, Any],
) -> tuple[str | None, bool, bool, bool]:
    if not kill_side:
        return None, True, False, False
    plant_tick = bomb_info.get("tick") if bomb_info else None
    plant_time = bomb_info.get("time") if bomb_info else None
    post_plant = False
    if plant_tick is not None and kill_tick is not None:
        post_plant = int(kill_tick) >= int(plant_tick)
    elif plant_time is not None and kill_time is not None:
        post_plant = kill_time >= float(plant_time)
    approx = plant_tick is None and plant_time is None
    no_plant = plant_tick is None and plant_time is None
    site_hint = objective_site or (bomb_info.get("site") if bomb_info else None)

    def _distance_to_center(center: tuple[float, float] | None) -> float | None:
        if center is None or kill_x is None or kill_y is None:
            return None
        return math.hypot(float(kill_x) - center[0], float(kill_y) - center[1])

    if post_plant:
        center = None
        if bomb_info:
            bomb_x = bomb_info.get("x")
            bomb_y = bomb_info.get("y")
            if bomb_x is not None and bomb_y is not None:
                center = (float(bomb_x), float(bomb_y))
        if center is None and site_hint:
            center = _site_center_world(map_name, site_hint, config)
            if center is not None:
                approx = True
        dist = _distance_to_center(center)
        if kill_side == "T":
            if dist is not None and dist <= POSTPLANT_DISTANCE_RADIUS:
                return "t_post_plant", approx, no_plant, False
            return "t_entry", approx, no_plant, False
        if kill_side == "CT":
            if dist is not None and dist <= POSTPLANT_DISTANCE_RADIUS:
                return "ct_retake", approx, no_plant, False
            return "ct_push", approx, no_plant, False
        return None, True, False, False

    site_center = _site_center_world(map_name, site_hint, config) if site_hint else None
    if site_center is None:
        approx = True
    dist = _distance_to_center(site_center)
    if kill_side == "T":
        if dist is not None and dist <= SITE_DISTANCE_RADIUS:
            if kill_time is not None and kill_time <= ENTRY_PHASE_MAX_SECONDS:
                return "t_execute", approx, no_plant, False
            if anchor_time is not None and kill_time is not None:
                if kill_time <= anchor_time + ENTRY_HOLD_DELAY_SECONDS:
                    return "t_execute", approx, no_plant, False
            return "t_hold", approx, no_plant, False
        return "t_entry", approx, no_plant, False
    if kill_side == "CT":
        if dist is None:
            return "ct_roam", True, no_plant, True
        return ("ct_hold" if dist <= SITE_DISTANCE_RADIUS else "ct_push"), approx, no_plant, False
    return None, True, False, False


def _init_entry_breakdown() -> dict[str, Any]:
    buckets = {label: 0 for label in get_time_bucket_presets().keys()}
    bucket_pcts = {label: None for label in get_time_bucket_presets().keys()}
    return {
        "entry_attempts": 0,
        "assisted_entry_count": 0,
        "solo_entry_count": 0,
        "unknown_entry_count": 0,
        "entry_with_assist": 0,
        "entry_with_flash": 0,
        "entry_with_partner": 0,
        "entry_with_group": 0,
        "unknown_support_count": 0,
        "assisted_entry_pct": None,
        "solo_entry_pct": None,
        "assisted_entry_pct_known": None,
        "solo_entry_pct_known": None,
        "assisted_entry_pct_all": None,
        "solo_entry_pct_all": None,
        "unknown_entry_pct": None,
        "entry_with_assist_pct": None,
        "entry_with_flash_pct": None,
        "entry_with_partner_pct": None,
        "entry_with_group_pct": None,
        "entry_with_assist_pct_known": None,
        "entry_with_flash_pct_known": None,
        "entry_with_partner_pct_known": None,
        "entry_with_group_pct_known": None,
        "assisted_by_bucket": dict(buckets),
        "solo_by_bucket": dict(buckets),
        "assisted_by_bucket_pct": dict(bucket_pcts),
        "solo_by_bucket_pct": dict(bucket_pcts),
        "avg_entry_time_s": None,
        "approx": False,
    }


def _entry_bucket_for_time(value: float | None) -> str:
    if value is None:
        return "unknown"
    thresholds = [15, 30, 45, 60, 75, 90]
    for threshold in thresholds:
        if value <= threshold:
            return f"0-{threshold}"
    return "0+"


def _finalize_entry_breakdown(entry_breakdown: dict[str, Any], entry_times: list[float]) -> dict[str, Any]:
    assisted = entry_breakdown["assisted_entry_count"]
    solo = entry_breakdown["solo_entry_count"]
    total = assisted + solo
    unknown = entry_breakdown.get("unknown_entry_count", 0)
    total_all = total + unknown
    if total:
        assisted_pct_known = (assisted / total) * 100
        solo_pct_known = (solo / total) * 100
        entry_breakdown["assisted_entry_pct_known"] = assisted_pct_known
        entry_breakdown["solo_entry_pct_known"] = solo_pct_known
        entry_breakdown["assisted_entry_pct"] = assisted_pct_known
        entry_breakdown["solo_entry_pct"] = solo_pct_known
        entry_breakdown["entry_with_assist_pct_known"] = (entry_breakdown["entry_with_assist"] / total) * 100
        entry_breakdown["entry_with_flash_pct_known"] = (entry_breakdown["entry_with_flash"] / total) * 100
        entry_breakdown["entry_with_partner_pct_known"] = (entry_breakdown["entry_with_partner"] / total) * 100
        entry_breakdown["entry_with_group_pct_known"] = (entry_breakdown["entry_with_group"] / total) * 100
        entry_breakdown["entry_with_assist_pct"] = entry_breakdown["entry_with_assist_pct_known"]
        entry_breakdown["entry_with_flash_pct"] = entry_breakdown["entry_with_flash_pct_known"]
        entry_breakdown["entry_with_partner_pct"] = entry_breakdown["entry_with_partner_pct_known"]
        entry_breakdown["entry_with_group_pct"] = entry_breakdown["entry_with_group_pct_known"]
        assisted_by_bucket_pct = {}
        solo_by_bucket_pct = {}
        for key in entry_breakdown["assisted_by_bucket"].keys():
            assisted_bucket = entry_breakdown["assisted_by_bucket"].get(key, 0)
            solo_bucket = entry_breakdown["solo_by_bucket"].get(key, 0)
            bucket_total = assisted_bucket + solo_bucket
            assisted_by_bucket_pct[key] = (assisted_bucket / bucket_total) * 100 if bucket_total else None
            solo_by_bucket_pct[key] = (solo_bucket / bucket_total) * 100 if bucket_total else None
        entry_breakdown["assisted_by_bucket_pct"] = assisted_by_bucket_pct
        entry_breakdown["solo_by_bucket_pct"] = solo_by_bucket_pct
    if total_all:
        entry_breakdown["assisted_entry_pct_all"] = (assisted / total_all) * 100
        entry_breakdown["solo_entry_pct_all"] = (solo / total_all) * 100
        entry_breakdown["unknown_entry_pct"] = (unknown / total_all) * 100
    if entry_times:
        entry_breakdown["avg_entry_time_s"] = sum(entry_times) / len(entry_times)
    entry_breakdown["approx"] = entry_breakdown["unknown_support_count"] > 0
    return entry_breakdown


def aggregate_player_features(
    parsed_demos: list[ParsedDemoEvents],
    target_steam_id: str,
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
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
    entry_breakdown = _init_entry_breakdown()
    entry_times: list[float] = []
    support_breakdown = {
        "total_kills": 0,
        "solo_kills": 0,
        "with_partner_kills": 0,
        "group_kills": 0,
        "assist_kills": 0,
        "flash_kills": 0,
        "categories": {"assist": 0, "flash": 0, "group": 0, "partner": 0, "solo": 0, "unknown": 0},
        "category_pct": {},
        "approx": False,
        "radius": PROXIMITY_RADIUS,
        "window_sec": PROXIMITY_WINDOW_SECONDS,
    }
    rounds_with_tick_positions: set[tuple[int, int]] = set()
    no_plant_rounds: set[tuple[int, int]] = set()
    kills_with_support_window = 0
    kills_with_support_data = 0
    hold_center_missing_count = 0
    attacker_id_examples: list[str] = []
    victim_id_examples: list[str] = []
    attacker_ids_seen: set[str] = set()
    victim_ids_seen: set[str] = set()

    if target_id is None:
        entry_breakdown = _init_entry_breakdown()
        side_roles = _compute_side_roles([])
        return (
            [],
            {"rounds": None, "tick_rate": tick_rate},
            {
                "player_kills": 0,
                "player_deaths": 0,
                "player_assists": 0,
                "player_util_damage_total": 0.0,
                "utility_damage_per_round": None,
                "player_contacts": 0,
            },
            entry_breakdown,
            support_breakdown,
            side_roles,
        )

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
        map_config = _load_zone_config(parsed.map_name)
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
            global_round_id = demo_index * 1000 + int(round_number or 0)
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
            round_positions = (
                parsed.tick_positions_by_round.get(round_number or 0, []) if parsed.tick_positions_by_round else []
            )
            if round_positions:
                rounds_with_tick_positions.add(round_key)
            bomb_info = parsed.bomb_plants_by_round.get(round_number or 0, {}) if parsed.bomb_plants_by_round else {}
            if not bomb_info or bomb_info.get("time") is None:
                no_plant_rounds.add(round_key)

            objective_site = None
            if bomb_info and bomb_info.get("site") in {"A", "B"}:
                objective_site = bomb_info.get("site")
            if objective_site is None:
                player_kills_in_round = [kill for kill in round_kills_sorted if is_target_attacker(kill)]
                objective_site = _objective_site_from_kills(player_kills_in_round, map_config)

            anchor_time = None
            if objective_site and round_kills_sorted:
                plant_time = bomb_info.get("time") if bomb_info else None
                for kill in round_kills_sorted:
                    if not is_target_attacker(kill):
                        continue
                    kill_time = kill.get("time")
                    if kill_time is None:
                        continue
                    if plant_time is not None and kill_time >= plant_time:
                        continue
                    if _is_in_objective_area(
                        kill.get("attacker_place"),
                        objective_site,
                        map_config,
                        parsed.map_name,
                        kill.get("attacker_x"),
                        kill.get("attacker_y"),
                    ):
                        anchor_time = kill_time if anchor_time is None else min(anchor_time, kill_time)

            for idx, kill in enumerate(round_kills_sorted):
                kill_time = kill.get("time")
                prior = round_kills_sorted[:idx]
                later = round_kills_sorted[idx + 1 :]
                is_first_duel = first_kill is not None and kill is first_kill

                if is_target_attacker(kill):
                    player_kills += 1
                    target_attacker_kills += 1
                    has_true_assist = _has_true_assist(kill)
                    has_flash_assist = bool(kill.get("assistedflash"))
                    kills_with_support_window += 1
                    nearby_count, support_approx = _nearby_teammates_count(
                        kill.get("tick"),
                        kill.get("attacker_x"),
                        kill.get("attacker_y"),
                        round_positions,
                        target_id,
                        tick_rate,
                    )
                    if not support_approx:
                        kills_with_support_data += 1
                    support_category = _support_category(has_true_assist, has_flash_assist, nearby_count)
                    phase, phase_approx, no_plant, hold_center_missing = _kill_phase(
                        kill.get("attacker_side") or target_side,
                        kill_time,
                        kill.get("tick"),
                        kill.get("attacker_x"),
                        kill.get("attacker_y"),
                        kill.get("attacker_place"),
                        parsed.map_name,
                        bomb_info,
                        objective_site,
                        anchor_time,
                        map_config,
                    )
                    ct_position_state = None
                    ct_position_approx = False
                    if phase in {"ct_hold", "ct_push"}:
                        ct_position_state, ct_position_approx, center_missing = _ct_position_state(
                            kill.get("attacker_place"),
                            kill.get("attacker_x"),
                            kill.get("attacker_y"),
                            parsed.map_name,
                            bomb_info,
                            objective_site,
                        )
                        hold_center_missing = hold_center_missing or center_missing
                    if no_plant:
                        no_plant_rounds.add(round_key)
                    if hold_center_missing:
                        hold_center_missing_count += 1
                    support_breakdown["total_kills"] += 1
                    if has_true_assist:
                        support_breakdown["assist_kills"] += 1
                    if has_flash_assist:
                        support_breakdown["flash_kills"] += 1
                    support_breakdown["categories"][support_category] = (
                        support_breakdown["categories"].get(support_category, 0) + 1
                    )
                    if support_category == "solo":
                        support_breakdown["solo_kills"] += 1
                    elif support_category == "partner":
                        support_breakdown["with_partner_kills"] += 1
                    elif support_category == "group":
                        support_breakdown["group_kills"] += 1
                    if support_category == "unknown" or nearby_count is None or support_approx:
                        support_breakdown["approx"] = True
                    round_events.append(
                        {
                            "type": "kill",
                            "round": global_round_id,
                            "round_num": round_number,
                            "demo_index": demo_index,
                            "time": kill_time,
                            "tick": kill.get("tick"),
                            "round_start_tick": kill.get("round_start_tick"),
                            "time_approx": kill.get("time_approx"),
                            "is_trade_kill": _is_trade_kill(kill, prior, target_id, target_side, target_name),
                            "is_first_duel": is_first_duel,
                            "is_first_duel_win": is_first_duel,
                            "attacker_place": kill.get("attacker_place"),
                            "attacker_x": kill.get("attacker_x"),
                            "attacker_y": kill.get("attacker_y"),
                            "side": kill.get("attacker_side") or target_side,
                            "assister": kill.get("assister_steamid64") or kill.get("assister"),
                            "assisted_by_teammate": has_true_assist or bool(kill.get("assistedflash")),
                            "has_true_assist": has_true_assist,
                            "has_flash_assist": has_flash_assist,
                            "nearby_teammates_count": nearby_count,
                            "support_category": support_category,
                            "support_approx": support_approx,
                            "is_solo_kill": support_category == "solo",
                            "phase": phase,
                            "phase_approx": phase_approx,
                            "ct_position_state": ct_position_state,
                            "ct_position_approx": ct_position_approx,
                            "hold_center_missing": hold_center_missing,
                        }
                    )

                if is_target_victim(kill):
                    player_deaths += 1
                    target_victim_deaths += 1
                    death_nearby_count, death_support_approx = _nearby_teammates_count(
                        kill.get("tick"),
                        kill.get("victim_x"),
                        kill.get("victim_y"),
                        round_positions,
                        target_id,
                        tick_rate,
                    )
                    death_support_category = _support_category(False, False, death_nearby_count)
                    round_events.append(
                        {
                            "type": "death",
                            "round": global_round_id,
                            "round_num": round_number,
                            "demo_index": demo_index,
                            "time": kill_time,
                            "tick": kill.get("tick"),
                            "round_start_tick": kill.get("round_start_tick"),
                            "time_approx": kill.get("time_approx"),
                            "was_traded": _is_traded_death(kill, later, target_id, target_side, target_name),
                            "is_first_duel": is_first_duel,
                            "victim_place": kill.get("victim_place"),
                            "victim_x": kill.get("victim_x"),
                            "victim_y": kill.get("victim_y"),
                            "side": kill.get("victim_side") or target_side,
                            "nearby_teammates_count": death_nearby_count,
                            "support_category": death_support_category,
                            "support_approx": death_support_approx,
                            "has_true_assist": False,
                            "has_flash_assist": False,
                        }
                    )

                if is_target_assister(kill):
                    player_assists += 1
                    target_assists += 1
                    round_events.append(
                        {
                            "type": "assist",
                            "round": global_round_id,
                            "round_num": round_number,
                            "demo_index": demo_index,
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
                            "round": global_round_id,
                            "round_num": round_number,
                            "demo_index": demo_index,
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
                        "round": global_round_id,
                        "round_num": round_number,
                        "demo_index": demo_index,
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
                        "round": global_round_id,
                        "round_num": round_number,
                        "demo_index": demo_index,
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
                                "round": global_round_id,
                                "round_num": round_number,
                                "demo_index": demo_index,
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
                                    "round": global_round_id,
                                    "round_num": round_number,
                                    "demo_index": demo_index,
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

            for event in round_events:
                event.setdefault("round", global_round_id)
                event.setdefault("round_num", round_number)
                event.setdefault("demo_index", demo_index)

            events.extend(round_events)

            entry_candidates = [event for event in round_events if event.get("type") in {"kill", "death"}]
            entry_event = None
            if entry_candidates:
                def _entry_sort_key(event: dict[str, Any]) -> tuple[int, float]:
                    tick_value = event.get("tick")
                    time_value = event.get("time")
                    tick_key = int(tick_value) if tick_value is not None else 10**9
                    time_key = float(time_value) if time_value is not None else float("inf")
                    return tick_key, time_key

                entry_event = min(entry_candidates, key=_entry_sort_key)

            # Entry support semantics (esports): supported = trade potential/spacing from teammates
            # (assist, flash assist, or partner/group proximity). Solo = isolated first contact with
            # no assist/flash and no nearby teammate. Unknown = missing time/coords/coverage.
            for entry in ([entry_event] if entry_event else []):
                entry_breakdown["entry_attempts"] += 1
                entry_time = entry.get("time")
                if entry_time is not None:
                    entry_times.append(float(entry_time))
                entry_bucket = _entry_bucket_for_time(entry_time)
                entry_x = entry.get("attacker_x") if entry.get("type") == "kill" else entry.get("victim_x")
                entry_y = entry.get("attacker_y") if entry.get("type") == "kill" else entry.get("victim_y")
                nearby_count, support_approx = _nearby_teammates_count(
                    entry.get("tick"),
                    entry_x,
                    entry_y,
                    round_positions,
                    target_id,
                    tick_rate,
                )
                has_true_assist = bool(entry.get("has_true_assist"))
                has_flash_assist = bool(entry.get("has_flash_assist"))
                assister_support_unknown = False
                effective_true_assist = False
                if has_true_assist:
                    assister_id = _safe_int(entry.get("assister"))
                    if assister_id is None or entry_time is None or entry_x is None or entry_y is None:
                        assister_support_unknown = True
                    else:
                        assister_positions = [
                            (pos.get("time"), pos.get("x"), pos.get("y"))
                            for pos in round_positions
                            if pos.get("steamid") == assister_id
                        ]
                        if assister_positions:
                            effective_true_assist = _entry_support_nearby(
                                float(entry_time),
                                float(entry_x),
                                float(entry_y),
                                [
                                    (float(t), float(x), float(y))
                                    for t, x, y in assister_positions
                                    if t is not None and x is not None and y is not None
                                ],
                            )
                        else:
                            assister_support_unknown = True
                if assister_support_unknown and has_true_assist and not has_flash_assist:
                    support_category = "unknown"
                    support_approx = True
                else:
                    support_category = _support_category(
                        effective_true_assist,
                        has_flash_assist,
                        nearby_count,
                    )
                assisted_flag = support_category in {"assist", "flash", "group", "partner"}
                if assisted_flag:
                    entry_breakdown["assisted_entry_count"] += 1
                    if entry_bucket in entry_breakdown["assisted_by_bucket"]:
                        entry_breakdown["assisted_by_bucket"][entry_bucket] += 1
                    if effective_true_assist:
                        entry_breakdown["entry_with_assist"] += 1
                    if has_flash_assist:
                        entry_breakdown["entry_with_flash"] += 1
                    if support_category == "group":
                        entry_breakdown["entry_with_group"] += 1
                    if support_category == "partner":
                        entry_breakdown["entry_with_partner"] += 1
                else:
                    unknown_support = (
                        entry_time is None
                        or entry_x is None
                        or entry_y is None
                        or support_category == "unknown"
                        or support_approx
                        or assister_support_unknown
                    )
                    if unknown_support:
                        entry_breakdown["unknown_support_count"] += 1
                        entry_breakdown["unknown_entry_count"] += 1
                    else:
                        entry_breakdown["solo_entry_count"] += 1
                        if entry_bucket in entry_breakdown["solo_by_bucket"]:
                            entry_breakdown["solo_by_bucket"][entry_bucket] += 1

    rounds_total = len(rounds_seen)
    utility_damage_per_round = (
        player_utility_damage_total / rounds_total if rounds_total else None
    )

    tick_coverage_pct = (len(rounds_with_tick_positions) / rounds_total) * 100 if rounds_total else None
    proximity_coverage_pct = (
        (kills_with_support_data / kills_with_support_window) * 100 if kills_with_support_window else None
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
        "tick_coverage_pct": tick_coverage_pct,
        "proximity_coverage_pct": proximity_coverage_pct,
        "hold_push_threshold": PUSH_DISTANCE,
        "R": PROXIMITY_RADIUS,
        "window_seconds": PROXIMITY_WINDOW_SECONDS,
        "no_plant_rounds_count": len(no_plant_rounds),
        "hold_center_missing_count": hold_center_missing_count,
    }

    entry_breakdown = _finalize_entry_breakdown(entry_breakdown, entry_times)
    side_roles = _compute_side_roles(parsed_demos)
    total_kills = support_breakdown["total_kills"]
    if total_kills:
        support_breakdown["category_pct"] = {
            key: (value / total_kills) * 100 for key, value in support_breakdown["categories"].items()
        }
        support_breakdown["solo_pct"] = (support_breakdown["solo_kills"] / total_kills) * 100
        support_breakdown["with_partner_pct"] = (support_breakdown["with_partner_kills"] / total_kills) * 100
        support_breakdown["group_pct"] = (support_breakdown["group_kills"] / total_kills) * 100
        support_breakdown["assist_pct"] = (support_breakdown["assist_kills"] / total_kills) * 100
        support_breakdown["flash_pct"] = (support_breakdown["flash_kills"] / total_kills) * 100
    return events, meta, debug, entry_breakdown, support_breakdown, side_roles


def _slice_label(bounds: tuple[int, int]) -> str:
    start, end = bounds
    if int(end) >= 999:
        return f"{int(start)}+"
    return f"{int(start)}-{int(end)}"


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


@lru_cache(maxsize=16)
def _load_zone_config(map_name: str | None) -> dict[str, Any]:
    if not map_name:
        return {}
    zones_path = Path(__file__).resolve().parent.parent / "maps" / "zones.json"
    if not zones_path.exists():
        return {}
    try:
        payload = json.loads(zones_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload.get(map_name, {})


def _normalize_place(place: str | None) -> str:
    if not place:
        return ""
    return str(place).strip().lower()


def _place_matches(place: str | None, tokens: Iterable[str]) -> bool:
    if not place:
        return False
    place_norm = _normalize_place(place)
    return any(token in place_norm for token in tokens)


def _zone_from_place(place: str | None, config: dict[str, Any]) -> str | None:
    if not place or not config:
        return None
    place_norm = _normalize_place(place)
    for zone, tokens in (config.get("place_map") or {}).items():
        for token in tokens:
            if token in place_norm:
                return zone
    return None


def _side_role_shares(counts: Counter[str]) -> dict[str, float]:
    total = sum(counts.values())
    shares = {key: 0.0 for key in ("A", "B", "MID", "OTHER")}
    if not total:
        return shares
    for key in shares.keys():
        shares[key] = (counts.get(key, 0) / total) if total else 0.0
    return shares


def _label_ct_role(shares: dict[str, float], samples: int, approx: bool) -> tuple[str, bool]:
    if samples < SIDE_ROLE_MIN_SAMPLES:
        return "Unknown", True
    approx = approx or samples < SIDE_ROLE_APPROX_SAMPLES
    share_a = shares.get("A", 0.0)
    share_b = shares.get("B", 0.0)
    share_mid = shares.get("MID", 0.0)
    share_other = shares.get("OTHER", 0.0)
    if share_a >= 0.55 and (share_a - share_b) >= 0.20:
        return "A anchor", approx
    if share_b >= 0.55 and (share_b - share_a) >= 0.20:
        return "B anchor", approx
    if share_mid >= 0.45:
        return "Mid rotator", approx
    if share_other >= 0.35:
        return "Roamer", approx
    return "Flexible rotator", approx


def _label_t_role(shares: dict[str, float], samples: int, approx: bool) -> tuple[str, bool]:
    if samples < SIDE_ROLE_MIN_SAMPLES:
        return "Unknown", True
    approx = approx or samples < SIDE_ROLE_APPROX_SAMPLES
    share_a = shares.get("A", 0.0)
    share_b = shares.get("B", 0.0)
    share_mid = shares.get("MID", 0.0)
    if share_mid >= 0.45:
        return "Mid controller", approx
    if share_a >= 0.55:
        return "A lane", approx
    if share_b >= 0.55:
        return "B lane", approx
    return "Split-flex", approx


def _compute_side_roles(parsed_demos: list[ParsedDemoEvents]) -> dict[str, Any]:
    counts = {"CT": Counter(), "T": Counter()}
    rounds_seen = {"CT": set(), "T": set()}
    samples = {"CT": 0, "T": 0}
    approx_flags = {"CT": False, "T": False}
    for demo_index, parsed in enumerate(parsed_demos, start=1):
        map_config = _load_zone_config(parsed.map_name)
        if not map_config:
            approx_flags["CT"] = True
            approx_flags["T"] = True
        for round_number, positions in (parsed.tick_positions_by_round or {}).items():
            for pos in positions:
                if not pos.get("is_target"):
                    continue
                t_round = pos.get("time")
                if t_round is None or t_round > SIDE_ROLE_SAMPLE_SECONDS:
                    continue
                side = pos.get("side")
                if side not in {"CT", "T"}:
                    continue
                rounds_seen[side].add((demo_index, round_number))
                zone = _zone_from_place(pos.get("place"), map_config)
                zone_bucket = zone if zone in {"A", "B", "MID"} else "OTHER"
                counts[side][zone_bucket] += 1
                samples[side] += 1

    ct_shares = _side_role_shares(counts["CT"])
    t_shares = _side_role_shares(counts["T"])
    ct_label, ct_approx = _label_ct_role(ct_shares, samples["CT"], approx_flags["CT"])
    t_label, t_approx = _label_t_role(t_shares, samples["T"], approx_flags["T"])
    ct_shares_pct = {key: value * 100 for key, value in ct_shares.items()}
    t_shares_pct = {key: value * 100 for key, value in t_shares.items()}
    return {
        "ct": {
            "label": ct_label,
            "shares": ct_shares_pct,
            "sample_seconds": SIDE_ROLE_SAMPLE_SECONDS,
            "samples": samples["CT"],
            "rounds": len(rounds_seen["CT"]),
            "approx": ct_approx,
        },
        "t": {
            "label": t_label,
            "shares": t_shares_pct,
            "sample_seconds": SIDE_ROLE_SAMPLE_SECONDS,
            "samples": samples["T"],
            "rounds": len(rounds_seen["T"]),
            "approx": t_approx,
        },
    }


@lru_cache(maxsize=16)
def _radar_meta(map_name: str) -> tuple[tuple[int, int], dict[str, Any]] | None:
    try:
        from faceit_analytics import analyzer

        radar, meta, _radar_name = analyzer.load_radar_and_meta(map_name)
        return radar.size, meta
    except Exception:
        return None


def _pixel_to_world(
    px: float,
    py: float,
    radar_size: tuple[int, int],
    map_meta: dict[str, Any],
) -> tuple[float, float]:
    try:
        from faceit_analytics import analyzer

        pos_x = float(map_meta["pos_x"])
        pos_y = float(map_meta["pos_y"])
        scale = float(map_meta["scale"])
        x_val = (float(px) - float(analyzer.OFFSET_X_PX)) * scale + pos_x
        y_val = pos_y - (float(py) - float(analyzer.OFFSET_Y_PX)) * scale
        return x_val, y_val
    except Exception:
        return float(px), float(py)


def _site_center_world(map_name: str | None, site_key: str | None, config: dict[str, Any]) -> tuple[float, float] | None:
    if not map_name or not site_key or not config:
        return None
    center_norm = (config.get("site_center_norm") or {}).get(site_key)
    if not center_norm or not isinstance(center_norm, (list, tuple)) or len(center_norm) != 2:
        return None
    meta = _radar_meta(map_name)
    if not meta:
        return None
    radar_size, radar_meta = meta
    px = float(center_norm[0]) * float(radar_size[0])
    py = float(center_norm[1]) * float(radar_size[1])
    return _pixel_to_world(px, py, radar_size, radar_meta)


def _ct_hold_state(
    kill_x: float | None,
    kill_y: float | None,
    map_name: str | None,
    bomb_info: dict[str, Any] | None,
    objective_site: str | None,
) -> tuple[str, bool, bool]:
    config = _load_zone_config(map_name)
    center = None
    approx = False
    if bomb_info:
        bomb_x = bomb_info.get("x")
        bomb_y = bomb_info.get("y")
        if bomb_x is not None and bomb_y is not None:
            center = (float(bomb_x), float(bomb_y))
        else:
            approx = True

    site_hint = _safe_str(objective_site) if objective_site else None
    if center is None and site_hint:
        center = _site_center_world(map_name, site_hint, config)
        if center is not None:
            approx = True

    if center is None or kill_x is None or kill_y is None:
        return "ct_roam", True, True

    dist = math.hypot(float(kill_x) - center[0], float(kill_y) - center[1])
    return ("ct_hold" if dist <= PUSH_DISTANCE else "ct_push"), approx, False


def _ct_position_state(
    kill_place: str | None,
    kill_x: float | None,
    kill_y: float | None,
    map_name: str | None,
    bomb_info: dict[str, Any] | None,
    objective_site: str | None,
) -> tuple[str, bool, bool]:
    state, approx, center_missing = _ct_hold_state(
        kill_x,
        kill_y,
        map_name,
        bomb_info,
        objective_site,
    )
    return state.replace("ct_", ""), approx, center_missing


def _zone_from_coords(map_name: str | None, x: float | None, y: float | None, config: dict[str, Any]) -> str | None:
    if not map_name or x is None or y is None or not config:
        return None
    bboxes = config.get("bbox") or {}
    if not bboxes:
        return None
    meta = _radar_meta(map_name)
    if not meta:
        return None
    radar_size, radar_meta = meta
    try:
        from faceit_analytics import analyzer

        w, h = radar_size
        pixel = analyzer._world_to_pixel(np.array([[x, y]], dtype=np.float32), radar_meta, (w, h))
        if pixel.size == 0:
            return None
        px, py = float(pixel[0][0]), float(pixel[0][1])
        nx = px / max(w, 1)
        ny = py / max(h, 1)
    except Exception:
        return None

    for zone, boxes in bboxes.items():
        for box in boxes:
            x1, y1, x2, y2 = box
            if x1 <= nx <= x2 and y1 <= ny <= y2:
                return zone
    return None


def _kill_zone(kill: dict[str, Any], map_name: str | None) -> str:
    config = _load_zone_config(map_name)
    zone = _zone_from_place(kill.get("attacker_place"), config)
    if zone:
        return zone
    zone = _zone_from_coords(map_name, kill.get("attacker_x"), kill.get("attacker_y"), config)
    if zone:
        return zone
    return _quadrant_from_coords(kill.get("attacker_x"), kill.get("attacker_y"))


def _site_centers(map_name: str | None, config: dict[str, Any]) -> dict[str, tuple[float, float]]:
    centers: dict[str, tuple[float, float]] = {}
    for site_key in ("A", "B"):
        center = _site_center_world(map_name, site_key, config)
        if center is not None:
            centers[site_key] = center
    return centers


def _distance_to_point(
    x: float | None,
    y: float | None,
    center: tuple[float, float] | None,
) -> float | None:
    if x is None or y is None or center is None:
        return None
    return math.hypot(float(x) - center[0], float(y) - center[1])


def _site_zone_from_coords(
    map_name: str | None,
    x: float | None,
    y: float | None,
    config: dict[str, Any],
    centers: dict[str, tuple[float, float]],
) -> str | None:
    zone = _zone_from_coords(map_name, x, y, config)
    if zone in {"A", "B"}:
        return zone
    if x is None or y is None:
        return None
    best_site = None
    best_dist = None
    for site_key, center in centers.items():
        dist = _distance_to_point(x, y, center)
        if dist is None:
            continue
        if dist <= R_SITE and (best_dist is None or dist < best_dist):
            best_site = site_key
            best_dist = dist
    return best_site


def _is_offsite_far(
    map_name: str | None,
    x: float | None,
    y: float | None,
    config: dict[str, Any],
    centers: dict[str, tuple[float, float]],
) -> bool:
    zone = _zone_from_coords(map_name, x, y, config)
    if zone in {"A", "B"}:
        return False
    if x is None or y is None:
        return False
    if centers:
        distances = [
            _distance_to_point(x, y, center)
            for center in centers.values()
            if _distance_to_point(x, y, center) is not None
        ]
        if distances:
            return all(dist >= R_FAR for dist in distances if dist is not None)
    return zone is not None and zone not in {"A", "B"}


def _is_in_plant_area(
    x: float | None,
    y: float | None,
    plant_xy: tuple[float, float] | None,
) -> bool:
    if x is None or y is None or plant_xy is None:
        return False
    return math.hypot(float(x) - plant_xy[0], float(y) - plant_xy[1]) <= R_PLANT


def _is_post_plant(
    tick: int | None,
    time: float | None,
    plant_tick: int | None,
    plant_time: float | None,
) -> bool:
    if plant_tick is not None and tick is not None:
        return int(tick) >= int(plant_tick)
    if plant_time is not None and time is not None:
        return float(time) >= float(plant_time)
    return False


def _infer_objective_site_from_ticks(
    positions: list[dict[str, Any]],
    round_kills: list[dict[str, Any]],
    map_name: str | None,
    config: dict[str, Any],
    centers: dict[str, tuple[float, float]],
    plant_tick: int | None,
    plant_time: float | None,
) -> str | None:
    site_counts: Counter[str] = Counter()
    for pos in positions:
        t_round = pos.get("time")
        if t_round is None or t_round > ENTRY_WINDOW_SEC:
            continue
        if _is_post_plant(pos.get("tick"), t_round, plant_tick, plant_time):
            continue
        site_zone = _site_zone_from_coords(map_name, pos.get("x"), pos.get("y"), config, centers)
        if site_zone in {"A", "B"}:
            site_counts[site_zone] += 1
    if site_counts:
        return site_counts.most_common(1)[0][0]
    for kill in sorted(round_kills, key=lambda k: k.get("time") or 0):
        site_zone = _site_zone_from_coords(map_name, kill.get("attacker_x"), kill.get("attacker_y"), config, centers)
        if site_zone in {"A", "B"}:
            return site_zone
    return None


def _assign_kill_phase(
    kill: dict[str, Any],
    side: str | None,
    map_name: str | None,
    config: dict[str, Any],
    centers: dict[str, tuple[float, float]],
    plant_info: dict[str, Any] | None,
) -> str:
    kill_time = kill.get("time")
    kill_tick = kill.get("tick")
    kill_x = kill.get("attacker_x")
    kill_y = kill.get("attacker_y")
    plant_tick = plant_info.get("tick") if plant_info else None
    plant_time = plant_info.get("time") if plant_info else None
    plant_xy = None
    if plant_info:
        plant_x = plant_info.get("x")
        plant_y = plant_info.get("y")
        if plant_x is not None and plant_y is not None:
            plant_xy = (float(plant_x), float(plant_y))
    post_plant = _is_post_plant(kill_tick, kill_time, plant_tick, plant_time)
    site_zone = _site_zone_from_coords(map_name, kill_x, kill_y, config, centers)
    in_site_area = site_zone in {"A", "B"}
    near_plant = _is_in_plant_area(kill_x, kill_y, plant_xy) if post_plant else False
    offsite_far = _is_offsite_far(map_name, kill_x, kill_y, config, centers)
    in_entry_window = kill_time is not None and kill_time <= ENTRY_WINDOW_SEC

    if side == "T":
        if not post_plant and in_site_area:
            return "t_execute"
        if post_plant and near_plant:
            return "t_postplant_hold"
        if in_entry_window and offsite_far:
            return "t_entry"
    if side == "CT":
        if not post_plant and in_site_area:
            return "ct_hold"
        if post_plant and near_plant:
            return "ct_retake"
        if not post_plant and offsite_far:
            return "ct_push"
    return "other"


def compute_kill_output_by_phase(
    parsed_demos: list[ParsedDemoEvents],
    target_steam_id: str,
    target_name: str | None,
) -> dict[str, Any]:
    target_id = normalize_steamid64(target_steam_id)
    phases = ["t_entry", "t_execute", "t_postplant_hold", "ct_hold", "ct_push", "ct_retake", "other"]
    hist_template = {f"k{i}": 0 for i in range(6)}
    output = {phase: {**hist_template, "total_rounds": 0} for phase in phases}
    if target_id is None:
        return {"phases": output}

    opportunity: dict[str, set[tuple[int, int]]] = {phase: set() for phase in phases}
    round_kill_counts: dict[tuple[int, int], Counter[str]] = {}
    kills_by_phase_total = Counter()
    total_kills = 0
    missing_kill_debug: list[dict[str, Any]] = []

    def is_target_attacker(kill: dict[str, Any]) -> bool:
        if kill.get("attacker_steamid64") == target_id:
            return True
        return bool(target_name and kill.get("attacker_name") == target_name)

    for demo_index, parsed in enumerate(parsed_demos, start=1):
        map_name = parsed.map_name
        config = _load_zone_config(map_name)
        centers = _site_centers(map_name, config)
        tick_rate = parsed.tick_rate or 64.0
        seconds_per_sample = (PROXIMITY_SAMPLE_STEP_TICKS / tick_rate) if tick_rate else 0.0
        kills_by_round: dict[int | None, list[dict[str, Any]]] = {}
        for kill in parsed.kills:
            if is_target_attacker(kill):
                kills_by_round.setdefault(kill.get("round"), []).append(kill)
                total_kills += 1

        rounds = set(kills_by_round.keys()) | set((parsed.tick_positions_by_round or {}).keys())
        for round_number in rounds:
            if round_number is None:
                continue
            round_key = (demo_index, round_number)
            round_kills = kills_by_round.get(round_number, [])
            target_side = _target_side_for_round(
                round_number,
                parsed.target_round_sides,
                round_kills,
                target_id,
                target_name,
            )
            plant_info = parsed.bomb_plants_by_round.get(round_number, {}) if parsed.bomb_plants_by_round else {}
            plant_tick = plant_info.get("tick") if plant_info else None
            plant_time = plant_info.get("time") if plant_info else None
            plant_xy = None
            if plant_info:
                plant_x = plant_info.get("x")
                plant_y = plant_info.get("y")
                if plant_x is not None and plant_y is not None:
                    plant_xy = (float(plant_x), float(plant_y))

            round_positions = (parsed.tick_positions_by_round or {}).get(round_number, [])
            target_positions = [pos for pos in round_positions if pos.get("is_target")]

            objective_site = plant_info.get("site") if plant_info else None
            if objective_site not in {"A", "B"}:
                objective_site = _infer_objective_site_from_ticks(
                    target_positions,
                    round_kills,
                    map_name,
                    config,
                    centers,
                    plant_tick,
                    plant_time,
                )

            sec_pre_site_a = 0.0
            sec_pre_site_b = 0.0
            sec_pre_site_any = 0.0
            sec_pre_offsite_far = 0.0
            sec_entry_offsite = 0.0
            sec_post_plant_area = 0.0

            if target_side:
                for pos in target_positions:
                    pos_time = pos.get("time")
                    if pos_time is None:
                        continue
                    pos_tick = pos.get("tick")
                    pos_x = pos.get("x")
                    pos_y = pos.get("y")
                    post_plant = _is_post_plant(pos_tick, pos_time, plant_tick, plant_time)
                    site_zone = _site_zone_from_coords(map_name, pos_x, pos_y, config, centers)
                    in_site = site_zone in {"A", "B"}
                    offsite_far = _is_offsite_far(map_name, pos_x, pos_y, config, centers)
                    if post_plant:
                        if _is_in_plant_area(pos_x, pos_y, plant_xy):
                            sec_post_plant_area += seconds_per_sample
                        continue
                    if site_zone == "A":
                        sec_pre_site_a += seconds_per_sample
                    if site_zone == "B":
                        sec_pre_site_b += seconds_per_sample
                    if in_site:
                        sec_pre_site_any += seconds_per_sample
                    if offsite_far:
                        sec_pre_offsite_far += seconds_per_sample
                        if pos_time <= ENTRY_WINDOW_SEC:
                            sec_entry_offsite += seconds_per_sample

                if target_side == "T":
                    if objective_site == "A":
                        sec_target_site = sec_pre_site_a
                    elif objective_site == "B":
                        sec_target_site = sec_pre_site_b
                    else:
                        sec_target_site = sec_pre_site_any
                    if sec_target_site >= MIN_SEC_IN_AREA:
                        opportunity["t_execute"].add(round_key)
                    if plant_info and sec_post_plant_area >= MIN_SEC_IN_AREA:
                        opportunity["t_postplant_hold"].add(round_key)
                    if sec_entry_offsite >= MIN_SEC_IN_AREA:
                        opportunity["t_entry"].add(round_key)
                elif target_side == "CT":
                    if sec_pre_site_any >= MIN_SEC_IN_AREA:
                        opportunity["ct_hold"].add(round_key)
                    if plant_info and sec_post_plant_area >= MIN_SEC_IN_AREA:
                        opportunity["ct_retake"].add(round_key)
                    if sec_pre_offsite_far >= MIN_SEC_IN_AREA:
                        opportunity["ct_push"].add(round_key)

            if round_kills:
                round_counts = round_kill_counts.setdefault(round_key, Counter())
                for kill in round_kills:
                    kill_side = target_side or _normalize_side(kill.get("attacker_side"))
                    phase = _assign_kill_phase(kill, kill_side, map_name, config, centers, plant_info)
                    round_counts[phase] += 1
                    kills_by_phase_total[phase] += 1
                    opportunity[phase].add(round_key)

        if None in kills_by_round:
            for kill in kills_by_round.get(None, []):
                kill_side = _normalize_side(kill.get("attacker_side"))
                phase = _assign_kill_phase(kill, kill_side, map_name, config, centers, {})
                kills_by_phase_total[phase] += 1

    for phase in phases:
        rounds = opportunity[phase]
        output[phase]["total_rounds"] = len(rounds)
        for round_key in rounds:
            count = round_kill_counts.get(round_key, {}).get(phase, 0)
            bucket = min(int(count), 5)
            output[phase][f"k{bucket}"] += 1

    accounted_kills = sum(kills_by_phase_total.values())
    missing_kills = total_kills - accounted_kills
    if missing_kills:
        for parsed in parsed_demos:
            for kill in parsed.kills:
                if not is_target_attacker(kill):
                    continue
                missing_kill_debug.append(
                    {
                        "round": kill.get("round"),
                        "time": kill.get("time"),
                        "attacker_side": kill.get("attacker_side"),
                        "attacker_x": kill.get("attacker_x"),
                        "attacker_y": kill.get("attacker_y"),
                        "victim_x": kill.get("victim_x"),
                        "victim_y": kill.get("victim_y"),
                    }
                )
        logger.debug(
            "Kill output by phase missing kills=%s target_id=%s debug_sample=%s",
            missing_kills,
            target_id,
            missing_kill_debug[:5],
        )

    reconciliation = {
        "total_kills": total_kills,
        "accounted_kills": accounted_kills,
        "missing_kills": missing_kills,
    }
    kills_by_phase_total_payload = {phase: int(kills_by_phase_total.get(phase, 0)) for phase in phases}
    return {
        "phases": output,
        "kills_by_phase_total": kills_by_phase_total_payload,
        "reconciliation": reconciliation,
    }


def compute_multikill_metrics(events: list[dict[str, Any]], map_name: str | None = None) -> dict[str, Any]:
    kills = [event for event in events if event.get("type") == "kill" and event.get("time") is not None]
    if not kills:
        empty_state = {
            "t_entry": {"k1": 0, "k2": 0, "k3": 0, "k4": 0, "k5": 0},
            "t_execute": {"k1": 0, "k2": 0, "k3": 0, "k4": 0, "k5": 0},
            "t_hold": {"k1": 0, "k2": 0, "k3": 0, "k4": 0, "k5": 0},
            "t_post_plant": {"k1": 0, "k2": 0, "k3": 0, "k4": 0, "k5": 0},
            "ct_hold": {"k1": 0, "k2": 0, "k3": 0, "k4": 0, "k5": 0},
            "ct_push": {"k1": 0, "k2": 0, "k3": 0, "k4": 0, "k5": 0},
            "ct_roam": {"k1": 0, "k2": 0, "k3": 0, "k4": 0, "k5": 0},
            "ct_retake": {"k1": 0, "k2": 0, "k3": 0, "k4": 0, "k5": 0},
        }
        return {
            "multikill_round_rate": None,
            "multikill_event_rate": None,
            "multikill_events": 0,
            "rounds_with_multikill": 0,
            "by_timing": {"early": 0, "late": 0},
            "by_state": empty_state,
            "by_zone": {},
            "ace_rounds": 0,
            "window_sec": MULTIKILL_WINDOW_SEC,
            "early_threshold_sec": MULTIKILL_EARLY_THRESHOLD_SEC,
            "entry_phase_max_sec": ENTRY_PHASE_MAX_SECONDS,
            "entry_hold_delay_sec": ENTRY_HOLD_DELAY_SECONDS,
        }

    rounds_with_multikill = set()
    multikill_events = 0
    timing_breakdown = {"early": 0, "late": 0}
    state_counts = {
        "t_entry": {"k1": 0, "k2": 0, "k3": 0, "k4": 0, "k5": 0},
        "t_execute": {"k1": 0, "k2": 0, "k3": 0, "k4": 0, "k5": 0},
        "t_hold": {"k1": 0, "k2": 0, "k3": 0, "k4": 0, "k5": 0},
        "t_post_plant": {"k1": 0, "k2": 0, "k3": 0, "k4": 0, "k5": 0},
        "ct_hold": {"k1": 0, "k2": 0, "k3": 0, "k4": 0, "k5": 0},
        "ct_push": {"k1": 0, "k2": 0, "k3": 0, "k4": 0, "k5": 0},
        "ct_roam": {"k1": 0, "k2": 0, "k3": 0, "k4": 0, "k5": 0},
        "ct_retake": {"k1": 0, "k2": 0, "k3": 0, "k4": 0, "k5": 0},
    }
    zone_breakdown: dict[str, int] = {}
    ace_rounds = 0

    kills_by_round: dict[int | None, list[dict[str, Any]]] = {}
    for kill in kills:
        kills_by_round.setdefault(kill.get("round"), []).append(kill)

    for round_number, round_kills in kills_by_round.items():
        if round_number is None:
            continue
        sorted_kills = sorted(round_kills, key=lambda k: k.get("time") or 0)
        total_kills_round = len(sorted_kills)
        if total_kills_round >= 5:
            ace_rounds += 1

        best_count = 0
        best_start_index = 0
        left = 0
        for right in range(total_kills_round):
            while left <= right:
                start_time = float(sorted_kills[left].get("time") or 0)
                end_time = float(sorted_kills[right].get("time") or 0)
                if end_time - start_time <= MULTIKILL_WINDOW_SEC:
                    break
                left += 1
            window_count = right - left + 1
            if window_count > best_count:
                best_count = window_count
                best_start_index = left

        if best_count >= 2:
            multikill_events += 1
            rounds_with_multikill.add(round_number)
            window_start_kill = sorted_kills[best_start_index]
            start_time = float(window_start_kill.get("time") or 0)
            key = "early" if start_time <= MULTIKILL_EARLY_THRESHOLD_SEC else "late"
            timing_breakdown[key] += 1
            zone = _kill_zone(window_start_kill, map_name)
            zone_breakdown[zone] = zone_breakdown.get(zone, 0) + 1
            phase = window_start_kill.get("phase")
            if phase in state_counts:
                streak_key = f"k{min(best_count, 5)}"
                state_counts[phase][streak_key] += 1

    rounds_total = _rounds_from_events(events)
    total_kills = len(kills)
    return {
        "multikill_round_rate": (multikill_events / rounds_total) * 100 if rounds_total else None,
        "multikill_event_rate": (multikill_events / total_kills) * 100 if total_kills else None,
        "multikill_events": multikill_events,
        "rounds_with_multikill": len(rounds_with_multikill),
        "by_timing": timing_breakdown,
        "by_state": state_counts,
        "by_zone": zone_breakdown,
        "ace_rounds": ace_rounds,
        "window_sec": MULTIKILL_WINDOW_SEC,
        "early_threshold_sec": MULTIKILL_EARLY_THRESHOLD_SEC,
        "entry_phase_max_sec": ENTRY_PHASE_MAX_SECONDS,
        "entry_hold_delay_sec": ENTRY_HOLD_DELAY_SECONDS,
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
        "missing_time_bomb": 0,
        "approx_time_kills": 0,
        "approx_time_bomb": 0,
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
        "bomb_event_counts": None,
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
        entry_breakdown = _init_entry_breakdown()
        entry_breakdown["approx"] = True
        support_breakdown = {
            "total_kills": 0,
            "solo_kills": 0,
            "with_partner_kills": 0,
            "group_kills": 0,
            "assist_kills": 0,
            "flash_kills": 0,
            "categories": {"assist": 0, "flash": 0, "group": 0, "partner": 0, "solo": 0, "unknown": 0},
            "category_pct": {},
            "approx": True,
            "radius": PROXIMITY_RADIUS,
            "window_sec": PROXIMITY_WINDOW_SECONDS,
        }
        side_roles = _compute_side_roles([])
        kill_output_by_phase = compute_kill_output_by_phase([], steam_id or "", None)
        payload = {
            "role_fingerprint": role_fingerprint,
            "utility_iq": utility_iq,
            "timing_slices": timing_slices,
            "entry_breakdown": entry_breakdown,
            "kill_support": support_breakdown,
            "side_roles": side_roles,
            "kill_output_by_phase": kill_output_by_phase,
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
        debug["missing_time_bomb"] += parsed.missing_time_bomb
        debug["approx_time_kills"] += parsed.approx_time_kills
        debug["approx_time_bomb"] += parsed.approx_time_bomb
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
            "bomb_event_counts",
        ):
            if parsed.debug.get(key) is not None and debug.get(key) in (None, [], {}):
                debug[key] = parsed.debug.get(key)
        if progress_callback:
            span = max(progress_end - progress_start, 1)
            progress = progress_start + int((index / max(demos_count, 1)) * span)
            progress_callback(progress)

    events, meta, player_debug, entry_breakdown, support_breakdown, side_roles = aggregate_player_features(
        parsed_demos,
        steam_id,
    )
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
            "tick_coverage_pct": player_debug.get("tick_coverage_pct"),
            "proximity_coverage_pct": player_debug.get("proximity_coverage_pct"),
            "hold_push_threshold": player_debug.get("hold_push_threshold"),
            "R": player_debug.get("R"),
            "window_seconds": player_debug.get("window_seconds"),
            "no_plant_rounds_count": player_debug.get("no_plant_rounds_count"),
            "hold_center_missing_count": player_debug.get("hold_center_missing_count"),
        }
    )
    rounds_total = meta.get("rounds") or 0

    awareness = compute_awareness_before_death(events)
    multikill = compute_multikill_metrics(events, map_name)
    kill_output_by_phase = compute_kill_output_by_phase(
        parsed_demos,
        steam_id,
        player_debug.get("target_name"),
    )

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
        if entry_breakdown:
            entry_breakdown["approx"] = True
        if support_breakdown:
            support_breakdown["approx"] = True
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
        "kill_output_by_phase": kill_output_by_phase,
        "entry_breakdown": entry_breakdown,
        "kill_support": support_breakdown,
        "side_roles": side_roles,
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
