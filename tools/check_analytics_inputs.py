# tools/check_analytics_inputs.py
from __future__ import annotations

import os
import json
from pathlib import Path
from collections import Counter, defaultdict

from django.conf import settings
from users.models import PlayerProfile
from faceit_analytics.services.paths import get_demos_dir

# awpy
from awpy import Demo


def _env_int(name: str, default: int | None = None) -> int | None:
    v = os.environ.get(name)
    if not v:
        return default
    try:
        return int(v)
    except Exception:
        return default


def _pick_existing_col(cols: list[str], candidates: list[str]) -> str | None:
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def _to_pandas(df_like):
    # awpy uses polars; some attrs might already be polars DF
    if df_like is None:
        return None
    try:
        return df_like.to_pandas()
    except Exception:
        return None


def _safe_list_unique(values, limit=12):
    out = []
    seen = set()
    for v in values:
        if v is None:
            continue
        s = str(int(v)) if str(v).isdigit() else str(v)
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
        if len(out) >= limit:
            break
    return out


def analyze_demo(demo_path: Path, target_steam: str):
    dem = Demo(str(demo_path), verbose=False)
    dem.parse()

    rounds_pd = _to_pandas(getattr(dem, "rounds", None))
    kills_pd = _to_pandas(getattr(dem, "kills", None))
    ticks_pd = _to_pandas(getattr(dem, "ticks", None))
    bomb_raw = getattr(dem, "bomb", None)
    if bomb_raw is None or (hasattr(bomb_raw, "is_empty") and bomb_raw.is_empty()):
        bomb_raw = getattr(dem, "bombs", None)
    bomb_pd = _to_pandas(bomb_raw)

    # tickrate
    tickrate = None
    for a in ("tickrate", "tick_rate", "tickRate"):
        if getattr(dem, a, None):
            tickrate = getattr(dem, a)
            break
    if tickrate is None:
        header = getattr(dem, "header", None) or {}
        tickrate = header.get("tickrate") or header.get("tick_rate") or header.get("tickRate")

    # rounds count
    rounds_count = 0
    if rounds_pd is not None and not rounds_pd.empty:
        col = _pick_existing_col(list(rounds_pd.columns), ["round_num", "round", "round_number"])
        if col:
            rounds_count = int(rounds_pd[col].nunique())
        else:
            rounds_count = len(rounds_pd)

    # ticks: does target appear?
    target_in_ticks = False
    ticks_target_rows = 0
    ticks_ids_sample = []
    if ticks_pd is not None and not ticks_pd.empty:
        steam_col = _pick_existing_col(list(ticks_pd.columns), ["steamid", "steam_id", "player_steamid", "playerSteamID"])
        if steam_col:
            # compare as int-string
            s = ticks_pd[steam_col].dropna()
            ticks_ids_sample = _safe_list_unique(s.values, limit=12)
            target_in_ticks = any(str(int(x)) == target_steam for x in s.values if str(x).replace(".", "").isdigit())
            if target_in_ticks:
                ticks_target_rows = int((s.astype("Int64").astype(str) == target_steam).sum())

    # kills: attacker/victim counts
    target_kills = 0
    target_deaths = 0
    kills_ids_sample = []
    victim_ids_sample = []
    if kills_pd is not None and not kills_pd.empty:
        cols = list(kills_pd.columns)
        a_col = _pick_existing_col(
            cols,
            [
                "attacker_steamid", "attackerSteamID", "attackerSteamId",
                "killer_steamid", "killerSteamID", "killerSteamId",
                "attacker", "killer",
            ],
        )
        v_col = _pick_existing_col(cols, ["victim_steamid", "victimSteamID", "victimSteamId", "victim"])
        r_col = _pick_existing_col(cols, ["round_num", "round", "round_number"])
        t_col = _pick_existing_col(cols, ["tick", "tick_num", "ticks"])

        if a_col:
            a = kills_pd[a_col].dropna()
            kills_ids_sample = _safe_list_unique(a.values, limit=12)
            # count kills by exact steam match
            try:
                target_kills = int((a.astype("Int64").astype(str) == target_steam).sum())
            except Exception:
                target_kills = sum(str(x).split(".")[0] == target_steam for x in a.values)

        if v_col:
            v = kills_pd[v_col].dropna()
            victim_ids_sample = _safe_list_unique(v.values, limit=12)
            try:
                target_deaths = int((v.astype("Int64").astype(str) == target_steam).sum())
            except Exception:
                target_deaths = sum(str(x).split(".")[0] == target_steam for x in v.values)

        # extra: rounds with >=5 kills (aces) for target
        ace_rounds = []
        if r_col and a_col:
            tmp = kills_pd[[r_col, a_col]].dropna()
            # normalize
            try:
                tmp[a_col] = tmp[a_col].astype("Int64").astype(str)
            except Exception:
                tmp[a_col] = tmp[a_col].astype(str).str.split(".").str[0]
            tmp = tmp[tmp[a_col] == target_steam]
            if not tmp.empty:
                rr = tmp.groupby(r_col).size().to_dict()
                ace_rounds = [int(r) for r, c in rr.items() if int(c) >= 5]

    # bomb events
    bomb_events = 0
    plants_by_round = defaultdict(int)
    if bomb_pd is not None and not bomb_pd.empty:
        bomb_events = len(bomb_pd)
        cols = list(bomb_pd.columns)
        ev_col = _pick_existing_col(cols, ["event", "type"])
        r_col = _pick_existing_col(cols, ["round_num", "round", "round_number"])
        if ev_col and r_col:
            for _, row in bomb_pd[[ev_col, r_col]].dropna().iterrows():
                ev = str(row[ev_col]).lower()
                rn = int(row[r_col])
                # awpy bomb events often: plant, planted, bomb_planted (varies)
                if "plant" in ev:
                    plants_by_round[rn] += 1

    return {
        "demo": demo_path.name,
        "tickrate": tickrate,
        "rounds_count": rounds_count,
        "target_in_ticks": target_in_ticks,
        "ticks_target_rows": ticks_target_rows,
        "ticks_ids_sample": ticks_ids_sample,
        "target_kills": target_kills,
        "target_deaths": target_deaths,
        "kills_attacker_ids_sample": kills_ids_sample,
        "kills_victim_ids_sample": victim_ids_sample,
        "bomb_events": bomb_events,
        "plants_by_round": dict(sorted(plants_by_round.items(), key=lambda x: x[0]))[:10],
    }


def check_heatmap_cache(steamid: str, map_name: str):
    base = Path("media") / "heatmaps_cache" / steamid / map_name
    out = {
        "cache_dir": str(base),
        "npz_count": 0,
        "npz_with_pxt": 0,
        "example_keys_union": [],
    }
    if not base.exists():
        return out

    union = set()
    npzs = sorted(base.glob("*.npz"))
    out["npz_count"] = len(npzs)
    for p in npzs:
        try:
            import numpy as np
            z = np.load(p, allow_pickle=True)
            keys = list(z.keys())
            for k in keys:
                union.add(k)
            if any(k.endswith("_pxt") for k in keys):
                out["npz_with_pxt"] += 1
        except Exception:
            pass

    out["example_keys_union"] = sorted(list(union))[:40]
    return out


def main():
    profile_id = _env_int("PROFILE_ID")
    map_name = os.environ.get("MAP_NAME") or "de_mirage"

    if profile_id:
        p = PlayerProfile.objects.filter(id=profile_id).first()
    else:
        p = PlayerProfile.objects.order_by("-id").first()

    if not p:
        raise SystemExit("PlayerProfile not found (set PROFILE_ID env var)")

    target_steam = str(p.steam_id)
    print("=== INPUT CHECK ===")
    print("profile_id:", p.id)
    print("steam_id:", target_steam)
    print("map:", map_name)

    demos_dir = Path(get_demos_dir(p, map_name))
    demo_paths = sorted(demos_dir.glob("*.dem"))
    print("\n=== DEMOS DIR ===")
    print("path:", demos_dir)
    print("dem_count:", len(demo_paths))
    if not demo_paths:
        raise SystemExit("No demos found for this map in demos_dir")

    print("\n=== PER-DEMO ===")
    any_kills_found = False
    any_ticks_found = False

    for dp in demo_paths:
        try:
            r = analyze_demo(dp, target_steam)
        except Exception as e:
            print(f"\n--- {dp.name} ---")
            print("ERROR while parsing demo:", repr(e))
            continue

        print(f"\n--- {r['demo']} ---")
        print("tickrate:", r["tickrate"], "| rounds:", r["rounds_count"])
        print("target_in_ticks:", r["target_in_ticks"], "| ticks_target_rows:", r["ticks_target_rows"])
        print("target_kills:", r["target_kills"], "| target_deaths:", r["target_deaths"])
        print("kills attacker ids sample:", r["kills_attacker_ids_sample"])
        print("kills victim ids sample  :", r["kills_victim_ids_sample"])
        print("ticks ids sample         :", r["ticks_ids_sample"])
        print("bomb_events:", r["bomb_events"], "| plants_by_round(sample):", json.dumps(r["plants_by_round"], ensure_ascii=False))

        any_kills_found = any_kills_found or (r["target_kills"] > 0 or r["target_deaths"] > 0)
        any_ticks_found = any_ticks_found or r["target_in_ticks"]

    print("\n=== HEATMAP CACHE CHECK ===")
    hm = check_heatmap_cache(target_steam, map_name)
    print("cache_dir:", hm["cache_dir"])
    print("npz_count:", hm["npz_count"])
    print("npz_with_pxt:", hm["npz_with_pxt"])
    print("keys_union(sample):", hm["example_keys_union"])

    print("\n=== DIAGNOSIS ===")
    if not any_ticks_found:
        print("* ПРОБЛЕМА: steam_id профиля вообще не встречается в ticks демо.")
        print("  -> либо steam_id в профиле неверный, либо папка демо не соответствует этому профилю.")
    elif not any_kills_found:
        print("* ВНИМАНИЕ: steam_id встречается в ticks, но НЕТ ни киллов, ни смертей в kills по этому steam_id.")
        print("  -> либо в kills используются другие колонки/формат (нужна нормализация), либо эти демо не этого игрока,")
        print("     либо awpy отдаёт steamid в kills иначе (float/обрезка/другая колонка).")
    else:
        print("* ОК: steam_id есть в ticks и в kills. Можно уже сравнивать расчёты метрик с raw-данными.")

    if hm["npz_count"] > 0 and hm["npz_with_pxt"] == 0:
        print("* ПРОБЛЕМА HEATMAP: кэш *.npz без *_pxt. Слайсы 0-15 / 0-30 не смогут работать корректно.")
        print("  -> нужно пересобрать heatmaps_cache с генерацией pxt (x,y,t) массивов.")


if __name__ == "__main__":
    main()
