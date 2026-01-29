from __future__ import annotations

from typing import Any


def _value(metric: dict[str, Any] | None) -> float | None:
    if not metric:
        return None
    value = metric.get("value")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_ratio(value: float | None, fallback: float = 0.0) -> float:
    if value is None:
        return fallback
    return float(value)


def infer_archetype(metrics: dict[str, Any], timing_slices: dict[str, Any], distances_optional: Any) -> dict[str, Any]:
    role_metrics = (metrics or {}).get("role_fingerprint", {}).get("metrics", {})
    kda = (metrics or {}).get("kda", {}) or {}

    first_duel_attempts = _value(role_metrics.get("first_duel_attempts_per_round"))
    first_contact_rate = _value(role_metrics.get("first_contact_rate"))
    trade_kill_rate = _value(role_metrics.get("trade_kill_rate"))
    traded_death_rate = _value(role_metrics.get("traded_death_rate"))
    flash_assists = _value(role_metrics.get("flash_assists_per_round"))
    assists_per_round = kda.get("assists_per_round")

    early_share = None
    late_share = None
    buckets = (timing_slices or {}).get("buckets") or []
    if buckets:
        early = buckets[0]
        late = buckets[-1]
        try:
            early_share = float(early.get("value")) / 100.0 if early.get("value") is not None else None
            late_share = float(late.get("value")) / 100.0 if late.get("value") is not None else None
        except (TypeError, ValueError):
            early_share = None
            late_share = None

    entry_score = (
        _safe_ratio(first_duel_attempts) * 1.6
        + _safe_ratio(first_contact_rate) * 1.3
        + _safe_ratio(early_share) * 1.1
    )
    support_score = (
        _safe_ratio(flash_assists) * 1.4
        + _safe_ratio(trade_kill_rate) * 1.1
        + _safe_ratio(assists_per_round) * 0.9
    )
    lurk_score = (
        _safe_ratio(late_share) * 1.4
        + (1.0 - _safe_ratio(traded_death_rate, 0.0)) * 0.8
        + (1.0 - _safe_ratio(first_duel_attempts, 0.0)) * 0.6
    )

    scores = {
        "Entry": entry_score,
        "Support": support_score,
        "Lurk": lurk_score,
    }
    label = max(scores, key=scores.get) if scores else "Unknown"

    reasons: list[str] = []
    if label == "Entry":
        if first_duel_attempts is not None:
            reasons.append("High first duel attempts per round.")
        if first_contact_rate is not None:
            reasons.append("Frequent early contacts.")
        if early_share is not None:
            reasons.append("Most contacts happen early in the round.")
    elif label == "Support":
        if flash_assists is not None:
            reasons.append("Consistent flash assists per round.")
        if trade_kill_rate is not None:
            reasons.append("Good trade kill involvement.")
        if assists_per_round is not None:
            reasons.append("High assist volume per round.")
    else:
        if late_share is not None:
            reasons.append("More contacts happen late in the round.")
        if traded_death_rate is not None:
            reasons.append("Lower traded death rate suggests spacing.")
        reasons.append("Less frequent early duels.")

    return {
        "label": label,
        "scores": scores,
        "reasons": reasons[:3],
        "approx": any(value is None for value in (first_duel_attempts, trade_kill_rate, flash_assists)),
    }
