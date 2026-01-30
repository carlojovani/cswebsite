"""tools/debug_profile_analytics.py

Простой диагностический скрипт, чтобы понять:
- совпадает ли steam_id профиля с тем, что реально встречается в демках
- сколько киллов у игрока в последней демке, в каких раундах, сколько эйсов
- есть ли bomb events / plants / rounds
- есть ли *_pxt keys в heatmap cache
- какие значения записаны в AnalyticsAggregate.metrics_json по ключевым метрикам

Запуск (PowerShell):
  python manage.py shell -c "from tools.debug_profile_analytics import main; main()"

ENV:
  PROFILE_ID=2
  MAP_NAME=de_mirage
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from users.models import PlayerProfile
from faceit_analytics.models import AnalyticsAggregate
from faceit_analytics.services.paths import get_demos_dir
from faceit_analytics.services.demo_events import parse_demo_events


BAD_RE = re.compile(r"(Р[А-Яа-я]|С[А-Яа-я]|вЂ|Ð)")


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    if not v:
        return default
    try:
        return int(v)
    except Exception:
        return default


def _get_nested(d: Any, path: str) -> Any:
    cur = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def _json_preview(x: Any, n: int = 260) -> str:
    try:
        s = json.dumps(x, ensure_ascii=False)
    except Exception:
        s = str(x)
    return s[:n] + ("…" if len(s) > n else "")


def _iter_demo_paths(profile: PlayerProfile, map_name: str) -> List[Path]:
    d = Path(get_demos_dir(profile, map_name))
    return sorted(d.glob("*.dem"))


def _steamid_set_from_parsed(ev) -> Tuple[set[str], set[str]]:
    ids_kills = set()
    for k in (ev.kills or []):
        a = k.get("attacker_steamid64")
        v = k.get("victim_steamid64")
        if a is not None:
            ids_kills.add(str(a))
        if v is not None:
            ids_kills.add(str(v))
    # ticks steamid встречается в debug (если parse_demo_events его туда кладёт) не всегда.
    # Поэтому в этой проверке только kills.
    return ids_kills, set()


def _heatmap_cache_keys(profile: PlayerProfile, map_name: str) -> Tuple[int, int, List[str]]:
    base = Path("media") / "heatmaps_cache" / str(profile.steam_id) / map_name
    if not base.exists():
        return 0, 0, []
    npzs = sorted(base.glob("*.npz"))
    with_pxt = 0
    keys_union = set()
    for p in npzs:
        try:
            z = np.load(p, allow_pickle=True)
            keys = list(z.keys())
            keys_union.update(keys)
            if any(k.endswith("_pxt") for k in keys):
                with_pxt += 1
        except Exception:
            continue
    return len(npzs), with_pxt, sorted(keys_union)


def main() -> None:
    profile_id = _env_int("PROFILE_ID", 2)
    map_name = os.environ.get("MAP_NAME", "de_mirage")

    p = PlayerProfile.objects.filter(id=profile_id).first()
    if not p:
        raise SystemExit(f"PlayerProfile id={profile_id} not found")

    steam = str(p.steam_id)
    print("=== PROFILE ===")
    print("profile_id:", p.id)
    print("steam_id:", steam)

    agg = AnalyticsAggregate.objects.filter(profile=p).order_by("-id").first()
    print("\n=== DB: AnalyticsAggregate ===")
    if not agg:
        print("NO AnalyticsAggregate row for this profile")
        mj = {}
    else:
        mj = agg.metrics_json or {}
        print("agg_id:", agg.id)
        print("analytics_version:", getattr(agg, "analytics_version", None))
        print("period:", getattr(agg, "period", None))
        top = sorted(list(mj.keys()))
        print("metrics_json top keys:", top[:60])
        # Быстрый поиск битого текста в JSON
        s = str(mj)
        print("contains_mojibake_markers:", bool(BAD_RE.search(s)))

    paths = [
        "playstyle.entry_breakdown.solo_entry_pct",
        "playstyle.entry_breakdown.assisted_entry_pct",
        "playstyle.entry_breakdown.solo_by_bucket_pct",
        "playstyle.entry_breakdown.assisted_by_bucket_pct",
        "multikill.aces",
        "multikill.phases",
        "kill_support",
        "timing_slices",
    ]
    print("\n=== DB: suspicious fields (if present) ===")
    for path in paths:
        v = _get_nested(mj, path)
        if v is not None:
            print(path, "=", _json_preview(v))

    demo_paths = _iter_demo_paths(p, map_name)
    print("\n=== DEMOS ===")
    print("demos_dir:", str(Path(get_demos_dir(p, map_name))))
    print("dem_count:", len(demo_paths))
    if not demo_paths:
        raise SystemExit("No demos found")

    demo = demo_paths[0 -1]
    print("using_demo:", demo.name)

    ev = parse_demo_events(demo, target_steam_id=steam)
    kills = ev.kills or []
    print("\n=== PARSE: kills summary ===")
    print("total_kill_events_in_demo:", len(kills))

    target_kills = [k for k in kills if str(k.get("attacker_steamid64")) == steam]
    print("target_kills:", len(target_kills))
    print("non_target_kills:", len(kills) - len(target_kills))

    # киллы по раундам
    by_round = Counter()
    for k in target_kills:
        r = k.get("round")
        if r is None:
            continue
        by_round[int(r)] += 1

    if by_round:
        ace_rounds = [r for r, c in sorted(by_round.items()) if c >= 5]
        print("rounds_with_target_kills:", len(by_round))
        print("ace_rounds(k>=5):", ace_rounds)
        print("ace_count:", len(ace_rounds))
        print("round -> target_kills (first 25):")
        for r, c in sorted(by_round.items())[:25]:
            print(f"{r:>2} -> {c}")
    else:
        print("NO target kills found by attacker_steamid64 match. (Possible ID mismatch or wrong demo.)")

    sample_ids = sorted({str(k.get("attacker_steamid64")) for k in kills[:200] if k.get("attacker_steamid64") is not None})[:15]
    print("\n=== PARSE: attacker_steamid64 samples ===")
    print("samples:", sample_ids)

    print("\n=== PARSE: bomb/rounds quick ===")
    # если parse_demo_events прокидывает bomb_plants_by_round/round_winners/debug
    for name in ["round_winners", "target_round_sides", "rounds_in_demo", "tick_rate", "tick_rate_approx", "missing_time_kills"]:
        v = getattr(ev, name, None)
        if v is not None:
            if isinstance(v, (dict, set, list)):
                print(name + ":", _json_preview(list(v) if isinstance(v, set) else v))
            else:
                print(name + ":", v)

    print("\n=== HEATMAP CACHE ===")
    npz_count, with_pxt, keys_union = _heatmap_cache_keys(p, map_name)
    print("npz_count:", npz_count)
    print("npz_with_pxt:", with_pxt)
    print("keys_union:", keys_union)

    print("\n=== QUICK DIAGNOSIS ===")
    if len(kills) > 0 and len(target_kills) == 0:
        print("* PROBLEM: profile.steam_id not present in kill events of this demo.")
        print("  -> либо steam_id в профиле неверный, либо демка не этого игрока.")
    if npz_count > 0 and with_pxt == 0:
        print("* PROBLEM: heatmap cache lacks *_pxt arrays => time slices (0-15 etc) cannot work.")
        print("  -> нужно пересобрать heatmap cache с time dimension.")

    print("done")


if __name__ == "__main__":
    main()
