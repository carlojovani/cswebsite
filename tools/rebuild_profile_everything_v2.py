# -*- coding: utf-8 -*-
"""
Rebuild everything for one PlayerProfile:
- sanity-check that profile steam_id exists in demo ticks/kills (after demo_events precision fix)
- clear AnalyticsAggregate rows for the profile (optional)
- wipe heatmaps_cache for this steam_id/map
- create ProcessingJob (best-effort via introspection) and run run_full_pipeline(job_id=...)

Usage (PowerShell):
  $env:PROFILE_ID="2"
  $env:MAP_NAME="de_mirage"
  $env:PERIOD="last_20"
  $env:FORCE="1"           # optional
  python manage.py shell -c "exec(open('tools/rebuild_profile_everything.py','r',encoding='utf-8').read())"

Notes:
- This script prints what it does. It avoids celery; runs pipeline inline.
"""
import os, shutil, json
from pathlib import Path as _Path

from django.utils import timezone
from django.db import transaction

from users.models import PlayerProfile
from faceit_analytics.services.paths import get_demos_dir

PROFILE_ID = int(os.getenv("PROFILE_ID", "2"))
MAP_NAME = os.getenv("MAP_NAME", "de_mirage")
PERIOD = os.getenv("PERIOD", "last_20")
FORCE = os.getenv("FORCE", "1") == "1"

def _print_kv(title, d):
    print(title)
    for k,v in d.items():
        print(f"  {k}: {v}")

def _safe_model_create(Model, **kwargs):
    """
    Create a model instance even if we don't know all required fields.
    We fill missing required (null=False, default=None) fields with best-effort values.
    """
    meta = Model._meta
    data = dict(kwargs)
    for f in meta.fields:
        if f.primary_key:
            continue
        if f.name in data:
            continue
        # skip auto fields
        if getattr(f, "auto_now", False) or getattr(f, "auto_now_add", False):
            continue
        if f.has_default() or f.null:
            continue
        # required without default
        if f.many_to_one and f.related_model and f.name.endswith("_id"):
            # FK id required but not provided
            continue
        if f.get_internal_type() in ("CharField", "TextField"):
            data[f.name] = ""
        elif f.get_internal_type() in ("IntegerField", "BigIntegerField", "PositiveIntegerField", "SmallIntegerField"):
            data[f.name] = 0
        elif f.get_internal_type() in ("BooleanField",):
            data[f.name] = False
        elif f.get_internal_type() in ("DateTimeField",):
            data[f.name] = timezone.now()
        elif f.get_internal_type() in ("JSONField",):
            data[f.name] = {}
        else:
            # last resort
            data[f.name] = None
    return Model.objects.create(**data)

def _unique_ids_in_demo(demo_path):
    # Import here so script works even if awpy isn't installed in non-demo environments
    from awpy import Demo
    demo = Demo(str(demo_path), verbose=False)
    demo.parse()
    ids = set()
    for table_name in ("ticks", "kills"):
        t = getattr(demo, table_name, None)
        if t is None:
            continue
        try:
            df = t.to_pandas(use_pyarrow_extension_array=True)
        except TypeError:
            df = t.to_pandas()
        for col in df.columns:
            if "steamid" in col.lower():
                # preserve as string
                for x in df[col].dropna().unique().tolist()[:5000]:
                    try:
                        ids.add(str(int(x)))
                    except Exception:
                        pass
    return ids

def _wipe_heatmaps_cache(steam_id, map_name):
    cache_dir = _Path("media") / "heatmaps_cache" / str(steam_id) / map_name
    if cache_dir.exists():
        shutil.rmtree(cache_dir, ignore_errors=True)
        print(f"deleted heatmaps_cache: {cache_dir}")
    else:
        print(f"heatmaps_cache not found: {cache_dir}")

def _clear_analytics(profile):
    from faceit_analytics.models import AnalyticsAggregate
    deleted, _ = AnalyticsAggregate.objects.filter(profile=profile).delete()
    print("deleted AnalyticsAggregate rows:", deleted)

def _run_pipeline(profile):
    from faceit_analytics.models import ProcessingJob
    from faceit_analytics.services.pipeline import run_full_pipeline

    # create job robustly
    kwargs = {}
    if "profile" in {f.name for f in ProcessingJob._meta.fields}:
        kwargs["profile"] = profile
    elif "profile_id" in {f.name for f in ProcessingJob._meta.fields}:
        kwargs["profile_id"] = profile.id

    # useful fields if they exist
    for name, val in [
        ("map_name", MAP_NAME),
        ("map", MAP_NAME),
        ("period", PERIOD),
        ("status", "queued"),
    ]:
        if name in {f.name for f in ProcessingJob._meta.fields} and name not in kwargs:
            kwargs[name] = val

    job = _safe_model_create(ProcessingJob, **kwargs)
    print("created ProcessingJob id:", job.id)

    # run
    run_full_pipeline(job_id=job.id, force_demo_features=FORCE, force_heatmaps=FORCE)
    print("pipeline finished")

def main():
    print("=== REBUILD EVERYTHING ===")
    profile = PlayerProfile.objects.filter(id=PROFILE_ID).first()
    if not profile:
        raise SystemExit(f"PlayerProfile id={PROFILE_ID} not found")
    steam = str(profile.steam_id)
    _print_kv("profile", {"id": profile.id, "steam_id": steam, "map": MAP_NAME, "period": PERIOD, "force": FORCE})

    demos_dir = _Path(get_demos_dir(profile, MAP_NAME))
    demos = sorted(demos_dir.glob("*.dem"))
    if not demos:
        raise SystemExit(f"No demos in {demos_dir}")
    demo = demos[-1]
    print("using demo:", demo.name)

    ids = _unique_ids_in_demo(demo)
    found = steam in ids
    print("steam_id present in raw demo ids:", found)
    if not found:
        # show closest matches
        near = sorted(ids, key=lambda x: abs(int(x) - int(steam)))[:10]
        print("closest steamid candidates in demo:", near)
        print("WARNING: steam_id mismatch. Fix profile steam_id or demo folder.")

    # do rebuild regardless
    with transaction.atomic():
        _clear_analytics(profile)

    _wipe_heatmaps_cache(steam, MAP_NAME)

    # clear django cache
    try:
        from django.core.cache import cache
        cache.clear()
        print("django cache cleared")
    except Exception as e:
        print("cache clear failed:", e)

    _run_pipeline(profile)

if __name__ == "__main__":
    main()

main()
