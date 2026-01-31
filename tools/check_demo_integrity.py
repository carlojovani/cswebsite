# tools/check_demo_integrity.py
from __future__ import annotations

import os
import json


# --- polars-safe df picker (no bool(polars.DataFrame)) ---
def _pick_df(*cands):
    for x in cands:
        if x is None:
            continue
        try:
            if hasattr(x, "is_empty") and x.is_empty():
                continue
        except Exception:
            pass
        return x
    return None

from pathlib import Path

from awpy import Demo
from users.models import PlayerProfile
from faceit_analytics.services.paths import get_demos_dir


PROFILE_ID = int(os.getenv("PROFILE_ID", "2"))
MAP_NAME = os.getenv("MAP_NAME", "de_mirage")
DEMO_NAME = os.getenv("DEMO_NAME", "")  # если пусто — берём последнюю .dem


def _as_polars(df):
    # awpy возвращает polars.DataFrame, но на всякий случай
    try:
        import polars as pl  # noqa
        return df
    except Exception:
        return df


def _print_json(title, obj, max_len=1000):
    s = json.dumps(obj, ensure_ascii=False, indent=2)
    if len(s) > max_len:
        s = s[:max_len] + "\n... (truncated)"
    print(title)
    print(s)


def main():
    print("=== DEMO INTEGRITY CHECK ===")
    print("PROFILE_ID:", PROFILE_ID, "MAP_NAME:", MAP_NAME, "DEMO_NAME:", DEMO_NAME or "(latest)")

    profile = PlayerProfile.objects.filter(id=PROFILE_ID).first()
    if not profile:
        raise SystemExit(f"PlayerProfile id={PROFILE_ID} not found")

    steam_id = str(profile.steam_id)
    demos_dir = Path(get_demos_dir(profile, MAP_NAME))
    dem_files = sorted(demos_dir.glob("*.dem"))

    print("\n=== PROFILE ===")
    print("profile_id:", profile.id)
    print("steam_id:", steam_id)
    print("demos_dir:", demos_dir)
    print("dem_count:", len(dem_files))
    if not dem_files:
        raise SystemExit("No demos in demos_dir")

    if DEMO_NAME:
        demo_path = demos_dir / DEMO_NAME
        if not demo_path.exists():
            raise SystemExit(f"DEMO_NAME not found: {demo_path}")
    else:
        demo_path = dem_files[-1]

    print("\n=== USING DEMO ===")
    print("demo:", demo_path.name)

    dem = Demo(str(demo_path), verbose=False)
    dem.parse()

    # --------- 1) CHECK TICKS / PLAYERS ----------
    ticks = getattr(dem, "ticks", None)
    if ticks is None:
        print("\n[FAIL] demo.ticks is None (невозможно строить movement/presence)")
    else:
        # ticks columns typical: health, place, side, X, Y, Z, tick, steamid, name, round_num
        print("\n=== TICKS (players presence) ===")
        try:
            cols = list(ticks.columns)
            print("ticks rows:", ticks.height, "cols(sample):", cols[:15])
        except Exception as e:
            print("ticks exists but cannot read columns:", e)
            cols = []

        required = ["steamid", "round_num", "X", "Y", "tick"]
        missing = [c for c in required if c not in cols]
        if missing:
            print("[FAIL] ticks missing columns:", missing)
        else:
            import polars as pl

            t = ticks.select(["steamid", "round_num", "X", "Y", "tick"])

            # 1) steam_id присутствует в ticks?
            has_target = t.filter(pl.col("steamid").cast(pl.Utf8) == steam_id).height > 0
            print("target steam_id present in ticks:", has_target)

            # 2) уникальные игроки по раундам (должно быть ~10)
            per_round = (
                t.group_by("round_num")
                .agg([
                    pl.col("steamid").n_unique().alias("uniq_players"),
                    pl.len().alias("rows"),
                    pl.col("X").null_count().alias("null_X"),
                    pl.col("Y").null_count().alias("null_Y"),
                ])
                .sort("round_num")
            )

            # найдём раунды, где явно меньше игроков
            bad_rounds = per_round.filter(pl.col("uniq_players") < 9).select(["round_num", "uniq_players", "rows"]).to_dicts()
            print("rounds with uniq_players < 9:", len(bad_rounds))
            if bad_rounds:
                print("sample bad rounds:", bad_rounds[:15])

            # доля null X/Y (в целом)
            total = t.height
            null_x = int(t.select(pl.col("X").null_count()).item())
            null_y = int(t.select(pl.col("Y").null_count()).item())
            print(f"null X: {null_x}/{total} ({(null_x/max(total,1))*100:.2f}%)")
            print(f"null Y: {null_y}/{total} ({(null_y/max(total,1))*100:.2f}%)")

            # экстремальные координаты (индикатор “кривой карты/трансформации/парсинга”)
            # (порог специально широкий, чтобы ловить реально мусор)
            extreme = t.filter(
                (pl.col("X").abs() > 100000) | (pl.col("Y").abs() > 100000)
            ).height
            print("extreme |X| or |Y| > 100000 rows:", extreme)

    # --------- 2) CHECK BOMB / PLANT ----------
    bomb = _pick_df(getattr(dem, 'bomb', None), getattr(dem, 'bombs', None))
    rounds = getattr(dem, "rounds", None)

    print("\n=== BOMB / PLANT CHECK ===")
    if bomb is None:
        print("[WARN] demo.bomb is None (нет явных bomb events от awpy)")
    else:
        try:
            bcols = list(bomb.columns)
            print("bomb rows:", bomb.height, "cols:", bcols)
        except Exception as e:
            print("bomb exists but cannot read columns:", e)
            bcols = []

        # стандарт: tick,event,X,Y,Z,steamid,name,bombsite,round_num
        import polars as pl

        if "event" not in bcols:
            print("[FAIL] bomb missing column: event")
        else:
            # какие события есть
            ev_counts = bomb.group_by("event").agg(pl.len().alias("n")).sort("n", descending=True).to_dicts()
            print("bomb event types (top):", ev_counts[:12])

            # plant events
            plant_df = bomb.filter(pl.col("event").str.contains("plant", literal=False))
            print("plant-like rows:", plant_df.height)

            if plant_df.height > 0:
                # bombsite заполнен?
                if "bombsite" in bcols:
                    site_counts = plant_df.group_by("bombsite").agg(pl.len().alias("n")).sort("n", descending=True).to_dicts()
                    print("plant bombsite counts:", site_counts)
                    null_site = plant_df.select(pl.col("bombsite").null_count()).item()
                    print("plant null bombsite:", int(null_site))
                else:
                    print("[WARN] bomb has no bombsite column")

                # координаты
                if "X" in bcols and "Y" in bcols:
                    null_x = int(plant_df.select(pl.col("X").null_count()).item())
                    null_y = int(plant_df.select(pl.col("Y").null_count()).item())
                    print("plant null X:", null_x, "null Y:", null_y)

                    extreme_plants = plant_df.filter(
                        (pl.col("X").abs() > 100000) | (pl.col("Y").abs() > 100000)
                    ).height
                    print("plant extreme coords >100000:", extreme_plants)

                    # пример 10 строк
                    sample = plant_df.select([c for c in ["round_num", "tick", "event", "X", "Y", "steamid", "name", "bombsite"] if c in bcols]).head(10).to_dicts()
                    _print_json("plant sample:", sample, max_len=2000)
                else:
                    print("[FAIL] bomb missing X/Y columns, cannot validate plant coords")

    # --------- 3) CHECK ROUNDS META VS BOMB (если доступно) ----------
    print("\n=== ROUNDS META CHECK ===")
    if rounds is None:
        print("[WARN] demo.rounds is None")
    else:
        try:
            rcols = list(rounds.columns)
            print("rounds rows:", rounds.height, "cols:", rcols)
        except Exception as e:
            print("rounds exists but cannot read columns:", e)
            rcols = []

        # часто есть: round_num, start, freeze_end, end, winner, bomb_plant, bomb_site
        import polars as pl

        if "round_num" in rcols:
            # базовая целостность: уникальные раунды
            uniq_rounds = rounds.select(pl.col("round_num").n_unique()).item()
            print("unique round_num in rounds:", int(uniq_rounds))

            # bomb_plant vs bomb events
            if "bomb_plant" in rcols:
                planted_rounds = rounds.filter(pl.col("bomb_plant").is_not_null()).select(["round_num", "bomb_plant"] + (["bomb_site"] if "bomb_site" in rcols else [])).to_dicts()
                print("rounds with bomb_plant != null:", len(planted_rounds))
                if planted_rounds:
                    _print_json("rounds bomb_plant sample:", planted_rounds[:10], max_len=2000)

                if bomb is not None and "round_num" in list(bomb.columns):
                    # сравним: раунды где rounds.bomb_plant есть, но в bomb нет plant events
                    if "event" in list(bomb.columns):
                        import polars as pl2
                        bomb_rounds_with_plant = set(
                            bomb.filter(pl2.col("event").str.contains("plant")).select(pl2.col("round_num")).to_series().to_list()
                        )
                        rounds_with_plant = set([x["round_num"] for x in planted_rounds])
                        missing_in_bomb = sorted(list(rounds_with_plant - bomb_rounds_with_plant))
                        extra_in_bomb = sorted(list(bomb_rounds_with_plant - rounds_with_plant))
                        print("rounds have bomb_plant in rounds but NO plant event in bomb:", missing_in_bomb[:20], ("..." if len(missing_in_bomb) > 20 else ""))
                        print("rounds have plant event in bomb but bomb_plant is null in rounds:", extra_in_bomb[:20], ("..." if len(extra_in_bomb) > 20 else ""))
            else:
                print("[WARN] rounds has no bomb_plant column; cannot cross-check")

    print("\n=== QUICK CONCLUSION ===")
    print("* If you see rounds with uniq_players < 9 -> ticks parsing incomplete or demo is partial.")
    print("* If plant events exist but bombsite is null or coords are extreme -> plant mapping will look wrong.")
    print("* If rounds.bomb_plant exists but bomb has no plant events (or vice versa) -> awpy parse inconsistency.")
    print("DONE")


if __name__ == '__main__':
    main()
