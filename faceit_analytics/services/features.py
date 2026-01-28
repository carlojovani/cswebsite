from __future__ import annotations

from dataclasses import dataclass
from typing import Any


TRADE_WINDOW_SECONDS = 5


@dataclass(frozen=True)
class FeatureValue:
    value: float | None
    approx: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"value": self.value, "approx": self.approx}


def _safe_divide(numerator: float | int | None, denominator: float | int | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return float(numerator) / float(denominator)


def _get_event_type(event: dict[str, Any]) -> str:
    return str(event.get("type") or event.get("event") or event.get("name") or "").lower()


def _get_event_time_seconds(event: dict[str, Any], meta: dict[str, Any]) -> float | None:
    for key in ("round_time", "time_from_round_start", "time", "timestamp"):
        if key in event and event[key] is not None:
            return float(event[key])

    if "tick" in event and event["tick"] is not None:
        tick_rate = meta.get("tick_rate") or meta.get("ticks_per_second")
        if tick_rate:
            return float(event["tick"]) / float(tick_rate)
    return None


def _round_count(events: list[dict[str, Any]], meta: dict[str, Any]) -> int | None:
    for key in ("rounds", "round_count", "total_rounds"):
        if meta.get(key):
            return int(meta[key])
    round_values = [e.get("round") for e in events if e.get("round") is not None]
    if round_values:
        unique = {int(v) for v in round_values}
        return len(unique)
    return None


def _flagged(event: dict[str, Any], *keys: str) -> bool:
    for key in keys:
        if event.get(key) is True:
            return True
    return False


def _is_first_duel_attempt(event: dict[str, Any]) -> bool:
    event_type = _get_event_type(event)
    return _flagged(event, "is_first_duel", "first_duel_attempt") or event_type in {
        "first_duel_attempt",
        "first_duel",
    }


def _is_first_duel_win(event: dict[str, Any]) -> bool:
    event_type = _get_event_type(event)
    return _flagged(event, "first_duel_win", "is_first_duel_win") or event_type == "first_duel_win"


def _is_contact_event(event: dict[str, Any]) -> bool:
    event_type = _get_event_type(event)
    if _flagged(event, "is_contact", "is_first_contact"):
        return True
    return event_type in {"contact", "kill", "death", "assist", "flash_assist", "damage"}


def _is_trade_kill(event: dict[str, Any]) -> bool:
    if _flagged(event, "is_trade_kill", "trade_kill"):
        return True
    if event.get("trade_time_delta") is not None:
        return float(event["trade_time_delta"]) <= TRADE_WINDOW_SECONDS
    if event.get("time_since_teammate_death") is not None:
        return float(event["time_since_teammate_death"]) <= TRADE_WINDOW_SECONDS
    return _get_event_type(event) == "trade_kill"


def _is_traded_death(event: dict[str, Any]) -> bool:
    if _flagged(event, "was_traded", "traded_death"):
        return True
    if event.get("time_to_trade") is not None:
        return float(event["time_to_trade"]) <= TRADE_WINDOW_SECONDS
    return _get_event_type(event) == "traded_death"


def _clutch_opportunity(event: dict[str, Any]) -> bool:
    return _flagged(event, "clutch_opportunity", "is_clutch_opportunity") or _get_event_type(event) == "clutch_opportunity"


def _clutch_win(event: dict[str, Any]) -> bool:
    return _flagged(event, "clutch_win", "is_clutch_win") or _get_event_type(event) == "clutch_win"


def compute_role_fingerprint(
    events: list[dict[str, Any]],
    positions: list[dict[str, Any]] | None,
    meta: dict[str, Any],
) -> dict[str, Any]:
    rounds = _round_count(events, meta)
    approx = False

    first_duel_attempts = sum(1 for event in events if _is_first_duel_attempt(event))
    first_duel_wins = sum(1 for event in events if _is_first_duel_win(event))
    if rounds:
        first_duel_attempts_per_round = FeatureValue(first_duel_attempts / rounds)
    else:
        first_duel_attempts_per_round = FeatureValue(None, approx=True)
        approx = True

    first_duel_success_rate = _safe_divide(first_duel_wins, first_duel_attempts)
    if first_duel_attempts == 0:
        first_duel_success = FeatureValue(None, approx=True)
        approx = True
    else:
        first_duel_success = FeatureValue(first_duel_success_rate)

    first_contact_events = sum(1 for event in events if _flagged(event, "is_first_contact"))
    if rounds and first_contact_events:
        first_contact_rate = FeatureValue(first_contact_events / rounds)
    elif rounds:
        first_contact_rate = FeatureValue(first_duel_attempts_per_round.value, approx=True)
        approx = True
    else:
        first_contact_rate = FeatureValue(None, approx=True)
        approx = True

    trade_kills = sum(1 for event in events if _is_trade_kill(event))
    total_kills = sum(1 for event in events if _get_event_type(event) == "kill")
    trade_kill_rate = _safe_divide(trade_kills, total_kills)
    if total_kills == 0:
        trade_kill = FeatureValue(None, approx=True)
        approx = True
    else:
        trade_kill = FeatureValue(trade_kill_rate)

    traded_deaths = sum(1 for event in events if _is_traded_death(event))
    total_deaths = sum(1 for event in events if _get_event_type(event) == "death")
    traded_death_rate = _safe_divide(traded_deaths, total_deaths)
    if total_deaths == 0:
        traded_death = FeatureValue(None, approx=True)
        approx = True
    else:
        traded_death = FeatureValue(traded_death_rate)

    clutch_ops = sum(1 for event in events if _clutch_opportunity(event))
    clutch_wins = sum(1 for event in events if _clutch_win(event))
    clutch_opportunities_rate = _safe_divide(clutch_ops, rounds)
    clutch_win_rate = _safe_divide(clutch_wins, clutch_ops)
    if rounds is None or clutch_ops == 0:
        clutch_ops_value = FeatureValue(None, approx=True)
        clutch_win_value = FeatureValue(None, approx=True)
        approx = True
    else:
        clutch_ops_value = FeatureValue(clutch_opportunities_rate)
        clutch_win_value = FeatureValue(clutch_win_rate)

    flash_assists = sum(
        1 for event in events if _flagged(event, "is_flash_assist") or _get_event_type(event) == "flash_assist"
    )
    if rounds:
        flash_assists_per_round = FeatureValue(flash_assists / rounds)
    else:
        flash_assists_per_round = FeatureValue(None, approx=True)
        approx = True

    utility_damage = sum(
        float(event.get("utility_damage") or event.get("nade_damage") or 0) for event in events
    )
    if rounds:
        utility_damage_per_round = FeatureValue(utility_damage / rounds)
    else:
        utility_damage_per_round = FeatureValue(None, approx=True)
        approx = True

    friendly_flash_count = sum(
        1 for event in events if _flagged(event, "is_friendly_flash", "friendly_flash")
    )
    total_flash_events = sum(
        1
        for event in events
        if _get_event_type(event) in {"flash", "flash_assist"}
        or _flagged(event, "is_flash", "is_flash_assist")
    )
    friendly_flash_rate = _safe_divide(friendly_flash_count, total_flash_events)
    if total_flash_events == 0:
        friendly_flash = FeatureValue(None, approx=True)
        approx = True
    else:
        friendly_flash = FeatureValue(friendly_flash_rate)

    metrics = {
        "first_duel_attempts_per_round": first_duel_attempts_per_round.to_dict(),
        "first_duel_success_rate": first_duel_success.to_dict(),
        "first_contact_rate": first_contact_rate.to_dict(),
        "trade_kill_rate": trade_kill.to_dict(),
        "traded_death_rate": traded_death.to_dict(),
        "clutch_opportunities_rate": clutch_ops_value.to_dict(),
        "clutch_win_rate": clutch_win_value.to_dict(),
        "flash_assists_per_round": flash_assists_per_round.to_dict(),
        "utility_damage_per_round": utility_damage_per_round.to_dict(),
        "friendly_flash_rate": friendly_flash.to_dict(),
    }

    tags = _build_role_tags(metrics, meta)
    summary = _build_role_summary(metrics, tags)

    return {
        "metrics": metrics,
        "tags": tags,
        "summary": summary,
        "approx": approx or any(m["approx"] for m in metrics.values()),
    }


def compute_utility_iq(events: list[dict[str, Any]], meta: dict[str, Any]) -> dict[str, Any]:
    rounds = _round_count(events, meta)
    approx = False

    flash_assists = sum(
        1 for event in events if _flagged(event, "is_flash_assist") or _get_event_type(event) == "flash_assist"
    )
    if rounds:
        flash_assists_per_round = FeatureValue(flash_assists / rounds)
    else:
        flash_assists_per_round = FeatureValue(None, approx=True)
        approx = True

    utility_damage = sum(
        float(event.get("utility_damage") or event.get("nade_damage") or 0) for event in events
    )
    if rounds:
        utility_damage_per_round = FeatureValue(utility_damage / rounds)
    else:
        utility_damage_per_round = FeatureValue(None, approx=True)
        approx = True

    friendly_flash_count = sum(
        1 for event in events if _flagged(event, "is_friendly_flash", "friendly_flash")
    )
    total_flash_events = sum(
        1
        for event in events
        if _get_event_type(event) in {"flash", "flash_assist"}
        or _flagged(event, "is_flash", "is_flash_assist")
    )
    friendly_flash_rate = _safe_divide(friendly_flash_count, total_flash_events)
    if total_flash_events == 0:
        friendly_flash = FeatureValue(None, approx=True)
        approx = True
    else:
        friendly_flash = FeatureValue(friendly_flash_rate)

    score = _utility_iq_score(
        flash_assists_per_round.value,
        utility_damage_per_round.value,
        friendly_flash.value,
    )
    if score is None:
        approx = True

    return {
        "score": score,
        "breakdown": {
            "flash_assists_per_round": flash_assists_per_round.to_dict(),
            "utility_damage_per_round": utility_damage_per_round.to_dict(),
            "friendly_flash_rate": friendly_flash.to_dict(),
        },
        "approx": approx,
    }


def compute_timing_slices(
    events: list[dict[str, Any]] | None,
    meta: dict[str, Any],
) -> dict[str, Any]:
    if not events:
        buckets = [
            {"label": "0-20", "value": None, "approx": True},
            {"label": "20-40", "value": None, "approx": True},
            {"label": "40-60", "value": None, "approx": True},
            {"label": "60+", "value": None, "approx": True},
        ]
        return {"buckets": buckets, "approx": True}

    counts = {
        "0-20": 0,
        "20-40": 0,
        "40-60": 0,
        "60+": 0,
    }
    total = 0
    for event in events:
        if not _is_contact_event(event):
            continue
        event_time = _get_event_time_seconds(event, meta)
        if event_time is None:
            continue
        total += 1
        if event_time < 20:
            counts["0-20"] += 1
        elif event_time < 40:
            counts["20-40"] += 1
        elif event_time < 60:
            counts["40-60"] += 1
        else:
            counts["60+"] += 1

    if total == 0:
        buckets = [
            {"label": "0-20", "value": None, "approx": True},
            {"label": "20-40", "value": None, "approx": True},
            {"label": "40-60", "value": None, "approx": True},
            {"label": "60+", "value": None, "approx": True},
        ]
        return {"buckets": buckets, "approx": True}

    buckets = [
        {"label": "0-20", "value": (counts["0-20"] / total) * 100, "approx": False},
        {"label": "20-40", "value": (counts["20-40"] / total) * 100, "approx": False},
        {"label": "40-60", "value": (counts["40-60"] / total) * 100, "approx": False},
        {"label": "60+", "value": (counts["60+"] / total) * 100, "approx": False},
    ]

    return {"buckets": buckets, "approx": False}


def _utility_iq_score(
    flash_assists_per_round: float | None,
    utility_damage_per_round: float | None,
    friendly_flash_rate: float | None,
) -> int | None:
    if flash_assists_per_round is None and utility_damage_per_round is None:
        return None

    flash_score = min((flash_assists_per_round or 0) / 0.2, 1.0) * 40
    util_damage_score = min((utility_damage_per_round or 0) / 30, 1.0) * 40
    friendly_flash_penalty = min((friendly_flash_rate or 0) / 0.2, 1.0) * 20

    score = max(0, min(100, round(flash_score + util_damage_score - friendly_flash_penalty)))
    return score


def _build_role_tags(metrics: dict[str, dict[str, Any]], meta: dict[str, Any]) -> list[str]:
    def _value(name: str) -> float | None:
        metric = metrics.get(name) or {}
        return metric.get("value")

    tags: list[str] = []
    first_duel_attempts = _value("first_duel_attempts_per_round")
    early_contacts = None
    timing_meta = meta.get("timing_slices")
    if timing_meta and isinstance(timing_meta, dict):
        for bucket in timing_meta.get("buckets", []):
            if bucket.get("label") == "0-20":
                early_contacts = bucket.get("value")
                break
    traded_death_rate = _value("traded_death_rate")
    flash_assists = _value("flash_assists_per_round")
    utility_damage = _value("utility_damage_per_round")
    clutch_opportunities = _value("clutch_opportunities_rate")
    clutch_win = _value("clutch_win_rate")

    if first_duel_attempts is not None and first_duel_attempts >= 0.14:
        if early_contacts is None or early_contacts >= 40:
            tags.append("Aggressive Entry")

    if first_duel_attempts is not None and first_duel_attempts >= 0.14:
        if traded_death_rate is not None and traded_death_rate < 0.3:
            tags.append("Low Trade Synergy")

    if (flash_assists is not None and flash_assists >= 0.12) or (
        utility_damage is not None and utility_damage >= 18
    ):
        tags.append("Support Utility")

    if clutch_opportunities is not None and clutch_opportunities >= 0.08:
        if clutch_win is not None and clutch_win >= 0.4:
            tags.append("Clutch Factor")

    if meta.get("side") == "CT":
        if early_contacts is not None and early_contacts < 25:
            tags.append("Passive Anchor")

    return tags[:5]


def _build_role_summary(metrics: dict[str, dict[str, Any]], tags: list[str]) -> str:
    if not tags:
        return "Not enough data to build a stable role fingerprint."

    primary = tags[0]
    second = tags[1] if len(tags) > 1 else None
    summary = f"{primary} tendencies"
    if second:
        summary += f" with {second.lower()} patterns"
    summary += "."

    first_duel_success = metrics.get("first_duel_success_rate", {}).get("value")
    if first_duel_success is not None:
        summary += f" First-duel success rate is {first_duel_success:.0%}."

    return summary
