from __future__ import annotations

from typing import Any


def adapt_feature_inputs(profile, period: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """
    Adapter to normalize data sources into the feature pipeline input format.
    Currently returns placeholders until match-level events/positions are wired.
    """
    events: list[dict[str, Any]] = []
    positions: list[dict[str, Any]] = []
    meta = {
        "map_name": None,
        "side": None,
        "period": period,
        "rounds": None,
        "profile_id": getattr(profile, "id", None),
    }
    return events, positions, meta
