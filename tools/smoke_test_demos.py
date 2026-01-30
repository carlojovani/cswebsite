# tools/smoke_test_demos.py
import os
import json
from pathlib import Path
from collections import Counter

from django.conf import settings
from users.models import PlayerProfile
from faceit_analytics.services.paths import get_demos_dir

# ВАЖНО: импортируй именно то, что реально существует в проекте
from faceit_analytics.services.demo_events import parse_demo_events


def _safe_int(x):
    try:
        return int(x)
    except Exception:
        return None


def _short_exc(e: Exception) -> str:
    return f"{type(e).__name__}: {e}"


def _print_kv(title: str, data: dict):
    print(title)
    for k, v in data.items():
        print(f"  {k}: {v}")


def _scan_npz_cache(steamid: str, map_name: str):
    cache_dir = Path("media") / "heatmaps_cache" / steamid / map_name
    out = {
        "cache_dir": str(cache_dir),
        "npz_count": 0,
        "npz_with_any_pxt": 0,
        "pxt_keys_union": [],
        "px_keys_union": [],
    }
    if not cache_dir.exists():
        return out

    npzs = sorted(cache_dir.glob("*.npz"))
    out["npz_count"] = len(npzs)
    if not npzs:
        return out

    # без numpy импорта, чтобы было проще; но он почти всегда есть — используем
    import numpy as np

    px_union = set()
    pxt_union = set()
    with_pxt = 0
    for p in npzs[:50]:
        try:
            z = np.load(p, allow_pickle=True)
            keys = list(z.keys())
            px_union |= {k for k in keys if k.endswith("_px")}
            pxt_union |= {k for k in keys if k.endswith("_pxt")}
            if any(k.endswith("_pxt") for k in keys):
                with_pxt += 1
        except Exception:
            continue

    out["npz_with_any_pxt"] = with_pxt
    out["px_keys_union"] = sorted(px_union)
    out["pxt_keys_union"] = sorted(pxt_union)
    return out


def main():
    # можно переопределить через env:
    # $env:PROFILE_ID="2"
    # $env:MAP_NAME="de_mirage"
    profile_id = os.environ.get("PROFILE_ID")
    map_name = os.environ.get("MAP_NAME", "de_mirage")

    if profile_id:
        profile = PlayerProfile.objects.filter(id=_safe_int(profile_id)).first()
    else:
        profile = PlayerProfile.objects.order_by("-id").first()

    if not profile:
        raise SystemExit("PlayerProfile not found (set PROFILE_ID env or create profile).")

    steam = str(profile.steam_id)

    print("=== SMOKE TEST: DEMO PARSING ===")
    _print_kv("PROFILE:", {"id": profile.id, "steam_id": steam, "map": map_name})

    demos_dir = Path(get_demos_dir(profile, map_name))
    demo_paths = sorted(demos_dir.glob("*.dem"))
    _print_kv("DEMOS:", {"dir": str(demos_dir), "count": len(demo_paths)})

    if not demo_paths:
        raise SystemExit("No .dem files found in demos dir.")

    failures = []
    parsed_ok = 0

    # статистика по steamid, чтобы понять — есть ли профиль вообще в демках
    steam_seen_in_ticks = Counter()
    steam_seen_in_kills = Counter()

    # берём до 20 демо, чтобы не ждать вечность
    for dem_path in demo_paths[-20:]:
        print(f"\n--- {dem_path.name} ---")
        try:
            ev = parse_demo_events(dem_path, target_steam_id=steam)
            parsed_ok += 1

            kills = ev.kills or []
            # normalize: attacker_steamid64 может быть int/np.uint64/str — приводим к str
            for k in kills:
                a = k.get("attacker_steamid64")
                if a is not None:
                    steam_seen_in_kills[str(a)] += 1

            # debug info: target_round_sides/round_winners полезны, но не обязательны
            print(f"tick_rate={getattr(ev,'tick_rate',None)} approx={getattr(ev,'tick_rate_approx',None)}")
            print(f"kills={len(kills)} flashes={len(getattr(ev,'flashes',[]) or [])} util_dmg={len(getattr(ev,'utility_damage',[]) or [])}")
            print(f"rounds_in_demo={len(getattr(ev,'rounds_in_demo',[]) or [])} missing_time_kills={getattr(ev,'missing_time_kills',None)}")

            # ключевой чек: есть ли target steam_id среди киллов
            target_kills = sum(1 for k in kills if str(k.get("attacker_steamid64")) == steam)
            print(f"target_kills_in_demo={target_kills}")

            # если parse_demo_events хранит debug payload — покажем типы steamid сырья
            dbg = getattr(ev, "debug", None) or {}
            if "attacker_steam_raw_type_sample" in dbg:
                print("attacker_steam_raw_type_sample:", dbg.get("attacker_steam_raw_type_sample"))

        except Exception as e:
            failures.append((dem_path.name, _short_exc(e)))
            print("ERROR:", _short_exc(e))

    print("\n=== SUMMARY ===")
    print("parsed_ok:", parsed_ok, "/", min(len(demo_paths), 20))
    if failures:
        print("\nFAILURES:")
        for name, err in failures:
            print(f"  {name}: {err}")

    # steam id presence analysis
    print("\n=== STEAMID PRESENCE (kills) ===")
    top_ids = steam_seen_in_kills.most_common(12)
    print("top attacker_steamid64:", top_ids)
    print("profile steam_id present in kills:", steam in steam_seen_in_kills)

    if steam not in steam_seen_in_kills:
        print("\n!!! PROBLEM !!!")
        print("Профильный steam_id НЕ встречается среди attacker_steamid64 в киллах.")
        print("Это почти гарантированно означает:")
        print("- steam_id в профиле неверный, ИЛИ")
        print("- демки в папке принадлежат другому аккаунту, ИЛИ")
        print("- steamid теряет точность (float/scientific notation) при загрузке данных.")
        print("До исправления этого метрики будут кривые.")

    # heatmap cache check
    print("\n=== HEATMAP CACHE CHECK ===")
    cache_info = _scan_npz_cache(steam, map_name)
    print(json.dumps(cache_info, ensure_ascii=False, indent=2))

    if cache_info["npz_count"] > 0 and cache_info["npz_with_any_pxt"] == 0:
        print("\n!!! PROBLEM !!!")
        print("Heatmap cache .npz НЕ содержит *_pxt ключей.")
        print("Значит time slice (0-15/0-30/...) работать не будет, и будет лог: 'missing time slice data'.")

    print("\nDONE")


# Чтобы можно было вызывать как модуль
if __name__ == "__main__":
    main()
