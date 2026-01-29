from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from faceit_analytics.services.heatmaps import _collect_points_from_cache, get_time_slice_labels
from faceit_analytics.services.paths import get_demos_dir
from users.models import PlayerProfile


class Command(BaseCommand):
    help = "Проверка аналитики: кэш тепловых карт, временные срезы и диагностика."

    def add_arguments(self, parser):
        parser.add_argument("--profile_id", type=int, required=True, help="ID профиля игрока.")
        parser.add_argument("--map", dest="map_name", default="de_mirage", help="Название карты.")
        parser.add_argument("--period", default="last_20", help="Период аналитики.")

    def handle(self, *args, **options):
        profile_id = options["profile_id"]
        map_name = options["map_name"]
        period = options["period"]
        profile = PlayerProfile.objects.filter(id=profile_id).first()
        if not profile:
            raise CommandError("Профиль не найден.")
        steamid64 = (
            getattr(profile, "steamid64", None)
            or getattr(profile, "steam_id64", None)
            or getattr(profile, "steam_id", None)
        )
        if not steamid64:
            raise CommandError("В профиле отсутствует SteamID64.")

        media_root = Path(getattr(settings, "MEDIA_ROOT", "media"))
        cache_root = media_root / "heatmaps_cache" / str(steamid64) / map_name
        if not cache_root.exists():
            raise CommandError(f"Кэш тепловых карт не найден: {cache_root}")

        required_keys = {
            "presence_all_pxt",
            "presence_ct_pxt",
            "presence_t_pxt",
            "kills_pxt",
            "deaths_pxt",
        }
        missing_keys: dict[str, list[str]] = {}
        for cache_path in cache_root.glob("*.npz"):
            with np.load(cache_path) as cached:
                missing = [key for key in required_keys if key not in cached.files]
                if missing:
                    missing_keys[cache_path.name] = missing

        status_lines = []
        ok = True
        if missing_keys:
            ok = False
            status_lines.append("FAIL: В кэше отсутствуют ключи времени:")
            for name, keys in missing_keys.items():
                status_lines.append(f"  - {name}: {', '.join(keys)}")
        else:
            status_lines.append("OK: В кэше присутствуют ключи *_pxt.")

        time_slices = [label for label in get_time_slice_labels() if label != "all"]
        if len(time_slices) >= 2:
            demos_dir = get_demos_dir(profile, map_name)
            slice_a, slice_b = time_slices[0], time_slices[1]
            points_a, _size_a, meta_a = _collect_points_from_cache(
                demos_dir,
                str(steamid64),
                map_name,
                period,
                "all",
                "kills",
                slice_a,
            )
            points_b, _size_b, meta_b = _collect_points_from_cache(
                demos_dir,
                str(steamid64),
                map_name,
                period,
                "all",
                "kills",
                slice_b,
            )
            if meta_a.get("missing_time_data_reason") or meta_b.get("missing_time_data_reason"):
                ok = False
                status_lines.append("FAIL: Временные срезы не применяются.")
                status_lines.append(
                    f"  - {slice_a}: {meta_a.get('missing_time_data_reason') or 'нет данных'}"
                )
                status_lines.append(
                    f"  - {slice_b}: {meta_b.get('missing_time_data_reason') or 'нет данных'}"
                )
            else:
                hash_a = hashlib.sha1(str(points_a).encode("utf-8")).hexdigest()
                hash_b = hashlib.sha1(str(points_b).encode("utf-8")).hexdigest()
                if hash_a == hash_b:
                    ok = False
                    status_lines.append(
                        f"FAIL: Срезы {slice_a} и {slice_b} дают одинаковые результаты."
                    )
                else:
                    status_lines.append(
                        f"OK: Срезы {slice_a} и {slice_b} отличаются (точек: {len(points_a)} vs {len(points_b)})."
                    )
        else:
            status_lines.append("OK: Недостаточно временных срезов для проверки.")

        status_lines.append("OK: Проверка завершена." if ok else "FAIL: Проверка завершена с ошибками.")
        for line in status_lines:
            self.stdout.write(line)
