"""tools/fix_all_rebuild.py

Одна команда для полного «reset + пересчёт» профиля + фиксы для SteamID mismatch и heatmap cache.

Запуск (PowerShell):
  python manage.py shell -c "from tools.fix_all_rebuild import main; main()"

Переменные окружения (опционально):
  PROFILE_ID=2
  MAP_NAME=de_mirage
  PERIOD=last_20
  ANALYTICS_VERSION=v2
  WRITE_STEAMID=1           # если найден корректный steamid64 в демках, обновит PlayerProfile.steam_id

Что делает:
  1) Берёт профиль, находит демки, проверяет, есть ли steam_id профиля в событиях.
  2) Если steam_id нет, пытается найти правильный по совпадению имени (username / nickname) в demo ticks/kills.
  3) Чистит AnalyticsAggregate для профиля и сбрасывает django cache.
  4) Удаляет heatmaps_cache для профиля (чтобы пересобрался с актуальными ключами).
  5) Создаёт ProcessingJob (даже если модель поменялась) и вызывает run_full_pipeline(...).

Важно:
  - Этот скрипт НЕ требует redis-cli (cache.clear() достаточно для local).
  - Если job модель строго требует значения в полях (choices), скрипт постарается заполнить разумными дефолтами.
"""

from __future__ import annotations

import os
import re
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

from django.core.cache import cache
from django.db import transaction
from django.utils import timezone

from users.models import PlayerProfile
from faceit_analytics.models import AnalyticsAggregate
from faceit_analytics.services.paths import get_demos_dir
from faceit_analytics.services.demo_events import parse_demo_events
from faceit_analytics.services.pipeline import run_full_pipeline


# ---------------------------- helpers ----------------------------

def _env(name: str, default: str) -> str:
    v = os.environ.get(name)
    return v if v not in (None, "") else default


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    if not v:
        return default
    try:
        return int(v)
    except Exception:
        return default


def _norm_name(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _safe_str(x: Any) -> str:
    try:
        return str(x)
    except Exception:
        return ""


@dataclass
class SteamResolveResult:
    steamid64: str
    reason: str
    confidence: float


def _collect_demo_names_and_ids(demo_path: Path) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Возвращает:
    - name->count (как часто встречается)
    - steamid->name (одно из имён)

    Используем awpy Demo напрямую, чтобы не зависеть от вашей нормализации в parse_demo_events.
    """
    from awpy import Demo

    d = Demo(str(demo_path), verbose=False)
    d.parse()

    name_counts: Dict[str, int] = {}
    id_to_name: Dict[int, str] = {}

    # ticks — самый надёжный источник имён
    try:
        ticks = getattr(d, "ticks", None)
        if ticks is not None:
            df = ticks.select(["steamid", "name"]).drop_nulls().unique().to_pandas()
            for _, row in df.iterrows():
                sid = int(row["steamid"]) if row.get("steamid") is not None else None
                nm = _safe_str(row.get("name"))
                if sid is None or not nm:
                    continue
                id_to_name.setdefault(sid, nm)
                nn = _norm_name(nm)
                name_counts[nn] = name_counts.get(nn, 0) + 1
    except Exception:
        pass

    # kills — запасной источник
    try:
        kills = getattr(d, "kills", None)
        if kills is not None:
            # attacker_steamid + attacker_name
            cols = kills.columns
            want = []
            for c in ["attacker_steamid", "attacker_name", "victim_steamid", "victim_name"]:
                if c in cols:
                    want.append(c)
            if want:
                dfk = kills.select(want).to_pandas()
                for _, row in dfk.iterrows():
                    for sid_key, name_key in [("attacker_steamid", "attacker_name"), ("victim_steamid", "victim_name")]:
                        if sid_key in row and name_key in row:
                            sid = row.get(sid_key)
                            nm = _safe_str(row.get(name_key))
                            if sid is None or not nm:
                                continue
                            try:
                                sid_i = int(sid)
                            except Exception:
                                continue
                            id_to_name.setdefault(sid_i, nm)
                            nn = _norm_name(nm)
                            name_counts[nn] = name_counts.get(nn, 0) + 1
    except Exception:
        pass

    return name_counts, id_to_name


def _resolve_steamid_for_profile(profile: PlayerProfile, demo_path: Path) -> SteamResolveResult:
    """Если steam_id профиля отсутствует в демке — пытаемся подобрать корректный steamid64 по имени."""

    target = str(profile.steam_id)
    name_counts, id_to_name = _collect_demo_names_and_ids(demo_path)
    ids_in_demo = set(id_to_name.keys())

    # 1) exact steam id exists
    try:
        if int(target) in ids_in_demo:
            return SteamResolveResult(steamid64=target, reason="steam_id_exists_in_demo", confidence=1.0)
    except Exception:
        pass

    # 2) try name match
    candidate_names = []
    # username
    if getattr(profile, "user", None) is not None:
        candidate_names.append(_norm_name(getattr(profile.user, "username", "")))
    # faceit nickname / custom fields (если есть)
    for attr in ["nickname", "faceit_nickname", "name", "display_name"]:
        if hasattr(profile, attr):
            candidate_names.append(_norm_name(_safe_str(getattr(profile, attr))))

    candidate_names = [n for n in candidate_names if n]

    # exact name
    best: Optional[SteamResolveResult] = None
    for sid, nm in id_to_name.items():
        nn = _norm_name(nm)
        if not nn:
            continue
        if any(nn == cn for cn in candidate_names):
            # чем чаще имя встречается (ticks), тем лучше
            freq = float(name_counts.get(nn, 1))
            conf = min(1.0, 0.7 + min(freq / 50.0, 0.3))
            best = SteamResolveResult(steamid64=str(sid), reason=f"name_exact:{nn}", confidence=conf)
            break

    if best:
        return best

    # 3) fuzzy contains
    def score(n_demo: str, n_prof: str) -> float:
        if not n_demo or not n_prof:
            return 0.0
        if n_demo == n_prof:
            return 1.0
        if n_demo in n_prof or n_prof in n_demo:
            return 0.85
        # simple token overlap
        a = set(n_demo.split())
        b = set(n_prof.split())
        inter = len(a & b)
        union = len(a | b) or 1
        return inter / union

    best_sid = None
    best_score = 0.0
    best_name = ""
    for sid, nm in id_to_name.items():
        nn = _norm_name(nm)
        for cn in candidate_names:
            sc = score(nn, cn)
            if sc > best_score:
                best_score = sc
                best_sid = sid
                best_name = nn

    if best_sid is not None and best_score >= 0.6:
        conf = min(0.9, 0.5 + best_score * 0.4)
        return SteamResolveResult(steamid64=str(best_sid), reason=f"name_fuzzy:{best_name}", confidence=conf)

    # 4) give up — just keep original
    return SteamResolveResult(steamid64=target, reason="no_match_keep_profile_steamid", confidence=0.0)


def _remove_heatmap_cache(profile: PlayerProfile, map_name: str) -> None:
    base = Path("media") / "heatmaps_cache" / str(profile.steam_id) / map_name
    if base.exists():
        shutil.rmtree(base, ignore_errors=True)
        print(f"heatmaps_cache removed: {base}")
    else:
        print(f"heatmaps_cache not found: {base}")


def _delete_aggregates(profile: PlayerProfile, period: str, analytics_version: str) -> int:
    qs = AnalyticsAggregate.objects.filter(profile=profile, period=period, analytics_version=analytics_version)
    n = qs.count()
    qs.delete()
    return n


def _create_processing_job(profile: PlayerProfile) -> Any:
    """Создаём ProcessingJob максимально «универсально», не зная точные поля модели."""
    # модель может быть в faceit_analytics.models
    from faceit_analytics import models as fa_models

    Job = getattr(fa_models, "ProcessingJob", None)
    if Job is None:
        raise RuntimeError("ProcessingJob model not found in faceit_analytics.models")

    field_map: Dict[str, Any] = {}
    for f in Job._meta.fields:
        # auto fields
        if f.auto_created and f.primary_key:
            continue

        name = f.name

        # FK на profile
        if name == "profile":
            field_map[name] = profile
            continue

        # defaults
        if f.has_default():
            continue

        # nullable/blank → пропускаем
        if getattr(f, "null", False) or getattr(f, "blank", False):
            continue

        internal = f.get_internal_type()

        if internal in ("CharField", "TextField"):
            # если choices есть — берём первое значение
            choices = list(getattr(f, "choices", []) or [])
            if choices:
                field_map[name] = choices[0][0]
            else:
                field_map[name] = "manual"
        elif internal in ("IntegerField", "BigIntegerField", "PositiveIntegerField", "SmallIntegerField"):
            field_map[name] = 0
        elif internal in ("FloatField", "DecimalField"):
            field_map[name] = 0.0
        elif internal == "BooleanField":
            field_map[name] = False
        elif internal in ("DateTimeField", "DateField"):
            field_map[name] = timezone.now()
        elif internal == "UUIDField":
            field_map[name] = uuid.uuid4()
        elif internal == "JSONField":
            field_map[name] = {}
        else:
            # fallback
            field_map[name] = getattr(f, "default", None) if f.has_default() else None

    job = Job.objects.create(**field_map)
    return job


# ---------------------------- main ----------------------------

def main() -> None:
    profile_id = _env_int("PROFILE_ID", 2)
    map_name = _env("MAP_NAME", "de_mirage")
    period = _env("PERIOD", "last_20")
    analytics_version = _env("ANALYTICS_VERSION", "v2")
    write_steamid = _env("WRITE_STEAMID", "").strip() in ("1", "true", "yes", "y")

    print("=== FIX ALL REBUILD ===")
    print("profile_id:", profile_id)

    profile = PlayerProfile.objects.filter(id=profile_id).select_related("user").first()
    if not profile:
        raise SystemExit(f"PlayerProfile id={profile_id} not found")

    print("old steam_id:", profile.steam_id)
    print("map:", map_name, "period:", period, "version:", analytics_version)

    demos_dir = Path(get_demos_dir(profile, map_name))
    demos = sorted(demos_dir.glob("*.dem"))
    if not demos:
        raise SystemExit(f"No demos found in {demos_dir}")

    demo = demos[-1]
    print("using demo:", demo.name)

    # resolve steam id mismatch
    res = _resolve_steamid_for_profile(profile, demo)
    print("resolved steam_id:", res.steamid64, "reason:", res.reason, "confidence:", f"{res.confidence:.2f}")

    if res.steamid64 != str(profile.steam_id):
        if write_steamid and res.confidence >= 0.6:
            print("UPDATING PlayerProfile.steam_id ->", res.steamid64)
            profile.steam_id = int(res.steamid64)
            profile.save(update_fields=["steam_id"])
        else:
            print("NOTE: steam_id differs, but WRITE_STEAMID not enabled or confidence low. Analytics may remain wrong.")

    # sanity check: do we now see target kills?
    ev = parse_demo_events(demo, target_steam_id=str(profile.steam_id))
    kills = ev.kills or []
    target_kills = [k for k in kills if str(k.get("attacker_steamid64")) == str(profile.steam_id)]
    print("parsed kills:", len(kills), "target_kills:", len(target_kills))

    with transaction.atomic():
        deleted = _delete_aggregates(profile, period=period, analytics_version=analytics_version)
        print("deleted AnalyticsAggregate rows:", deleted)

    cache.clear()
    print("django cache cleared")

    _remove_heatmap_cache(profile, map_name)

    # create job and run pipeline
    print("=== RUN PIPELINE (forced) ===")
    job = _create_processing_job(profile)
    print("created ProcessingJob id:", getattr(job, "id", None))

    # run_full_pipeline signature differs between versions.
    # We call it with the most common params, and fall back if needed.
    kwargs = dict(
        profile_id=profile.id,
        job_id=getattr(job, "id", None),
        period=period,
        map_name=map_name,
        analytics_version=analytics_version,
        force_demo_features=True,
        force_heatmaps=True,
        force_aggregates=True,
    )

    try:
        run_full_pipeline(**kwargs)
        print("PIPELINE: done")
    except TypeError as e:
        # if signature differs, retry with minimal set
        print("PIPELINE TypeError:", e)
        minimal = dict(profile_id=profile.id, job_id=getattr(job, "id", None))
        run_full_pipeline(**minimal)
        print("PIPELINE: done (minimal args)")


if __name__ == "__main__":
    main()
