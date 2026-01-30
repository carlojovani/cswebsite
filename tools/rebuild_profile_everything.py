"""
Rebuild analytics + heatmaps for one PlayerProfile, with robust pipeline calling.

Usage (PowerShell):
  python manage.py shell -c "exec(open('tools/rebuild_profile_everything.py','r',encoding='utf-8').read())"

Optional env vars:
  PROFILE_ID=2
  MAP_NAME=de_mirage
  PERIOD=last_20
  FORCE=1        (default 1)
  CLEAR_CACHE=1  (default 1)  -> clears django cache + removes media/heatmaps_cache/<steam>/<map>
  DELETE_AGG=1   (default 1)  -> deletes AnalyticsAggregate rows for this profile/map/period/version (best-effort)

This script:
  - loads the profile
  - prints steam id and checks if it appears in demo kill events (sanity)
  - clears caches
  - creates ProcessingJob (if model exists)
  - calls run_full_pipeline with signature introspection (so it won't break if signature differs)
  - prints resulting AnalyticsAggregate + heatmap cache keys summary
"""
from __future__ import annotations

import os
import shutil
import inspect
from pathlib import Path

from django.core.cache import cache

from users.models import PlayerProfile
from faceit_analytics.services.paths import get_demos_dir

def _env(name: str, default: str) -> str:
    v = os.environ.get(name)
    return v if v is not None and v != "" else default

def _env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    return v.strip().lower() in ("1","true","yes","y","on")

def _try_import(path: str):
    try:
        mod_path, attr = path.rsplit(".", 1)
        mod = __import__(mod_path, fromlist=[attr])
        return getattr(mod, attr)
    except Exception:
        return None

def _call_with_signature(fn, **kwargs):
    sig = inspect.signature(fn)
    accepted = {}
    for name, param in sig.parameters.items():
        if name in kwargs:
            accepted[name] = kwargs[name]
    return fn(**accepted)

def main():
    profile_id = int(_env("PROFILE_ID", "2"))
    map_name = _env("MAP_NAME", "de_mirage")
    period = _env("PERIOD", "last_20")
    analytics_version = _env("ANALYTICS_VERSION", "v2")

    force = _env_bool("FORCE", True)
    clear_cache = _env_bool("CLEAR_CACHE", True)
    delete_agg = _env_bool("DELETE_AGG", True)

    print("=== REBUILD PROFILE EVERYTHING ===")
    print("PROFILE_ID:", profile_id, "MAP:", map_name, "PERIOD:", period, "VERSION:", analytics_version)

    profile = PlayerProfile.objects.filter(id=profile_id).first()
    if not profile:
        raise SystemExit(f"PlayerProfile id={profile_id} not found")
    steam = str(profile.steam_id)
    print("steam_id:", steam)

    # Optional: quick sanity check on latest demo -> whether steam id appears in kills
    demo_dir = Path(get_demos_dir(profile, map_name))
    demos = sorted(demo_dir.glob("*.dem"))
    print("demos_dir:", demo_dir)
    print("demo_count:", len(demos))
    if demos:
        demo = demos[-1]
        print("using_demo:", demo.name)
        parse_demo_events = _try_import("faceit_analytics.services.demo_events.parse_demo_events")
        if parse_demo_events:
            ev = parse_demo_events(demo, target_steam_id=steam)
            kills = ev.kills or []
            ids = {str(k.get("attacker_steamid64")) for k in kills if k.get("attacker_steamid64") is not None}
            ids |= {str(k.get("victim_steamid64")) for k in kills if k.get("victim_steamid64") is not None}
            print("unique steamids in kills:", len(ids))
            print("steam_id present in kills:", steam in ids)
        else:
            print("WARN: parse_demo_events not importable")
    else:
        print("WARN: no demos found, pipeline may still run using other maps/periods")

    # Delete aggregate rows (best-effort)
    if delete_agg:
        AnalyticsAggregate = _try_import("faceit_analytics.models.AnalyticsAggregate")
        if AnalyticsAggregate:
            qs = AnalyticsAggregate.objects.filter(profile=profile, period=period, analytics_version=analytics_version)
            deleted = qs.delete()[0]
            print("deleted AnalyticsAggregate rows:", deleted)
        else:
            print("WARN: AnalyticsAggregate model not found, skip delete")

    # Clear django cache + heatmaps_cache
    if clear_cache:
        try:
            cache.clear()
            print("django cache cleared")
        except Exception as e:
            print("WARN: cache.clear failed:", e)

        hm_dir = Path("media") / "heatmaps_cache" / steam / map_name
        if hm_dir.exists():
            shutil.rmtree(hm_dir, ignore_errors=True)
            print("removed heatmaps_cache:", hm_dir)
        else:
            print("heatmaps_cache not found:", hm_dir)

    # Create a ProcessingJob if model exists
    ProcessingJob = _try_import("faceit_analytics.models.ProcessingJob") or _try_import("users.models.ProcessingJob")
    job = None
    if ProcessingJob:
        try:
            job = ProcessingJob.objects.create(profile=profile, status="running")
            print("created ProcessingJob id:", job.id)
        except Exception as e:
            print("WARN: cannot create ProcessingJob:", e)

    run_full_pipeline = _try_import("faceit_analytics.services.pipeline.run_full_pipeline")
    if not run_full_pipeline:
        raise SystemExit("run_full_pipeline not found at faceit_analytics.services.pipeline.run_full_pipeline")

    kwargs = {
        "job_id": getattr(job, "id", None),
        "processing_job_id": getattr(job, "id", None),
        "profile_id": profile.id,
        "profile": profile,
        "period": period,
        "map_name": map_name,
        "analytics_version": analytics_version,
        "force_rebuild": force,
        "force_demo_features": force,
        "force_heatmaps": force,
        "force": force,
    }
    # remove None values so we don't pass them accidentally
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    print("=== RUN PIPELINE ===")
    try:
        _call_with_signature(run_full_pipeline, **kwargs)
        print("pipeline finished")
    except Exception as e:
        print("PIPELINE ERROR:", repr(e))
        raise

    # Print last aggregate
    AnalyticsAggregate = _try_import("faceit_analytics.models.AnalyticsAggregate")
    if AnalyticsAggregate:
        agg = AnalyticsAggregate.objects.filter(profile=profile).order_by("-id").first()
        if agg:
            print("=== RESULT AGG ===")
            print("agg_id:", agg.id, "period:", getattr(agg, "period", None), "version:", getattr(agg, "analytics_version", None))
            mj = agg.metrics_json or {}
            print("metrics_json keys:", sorted(list(mj.keys()))[:40])
        else:
            print("NO AnalyticsAggregate created")
    else:
        print("WARN: AnalyticsAggregate model not found")

    # Heatmap cache keys check
    hm_dir = Path("media") / "heatmaps_cache" / steam / map_name
    if hm_dir.exists():
        npzs = sorted(hm_dir.glob("*.npz"))
        print("=== HEATMAP CACHE ===")
        print("npz_count:", len(npzs))
        if npzs:
            import numpy as np
            z = np.load(npzs[0], allow_pickle=True)
            print("sample npz keys:", list(z.keys()))
    else:
        print("heatmaps_cache folder still missing:", hm_dir)

    print("done")

if __name__ == "__main__":
    main()
else:
    # allow `exec(open(...).read())`
    main()
