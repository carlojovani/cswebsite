import json
import time
from pathlib import Path

from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.core.cache import cache
from django.core.files.storage import default_storage
from django.http import JsonResponse
from django.shortcuts import get_object_or_404
from django.views.decorators.http import require_GET, require_POST
from requests import HTTPError

from .cache_keys import DEFAULT_TTL_SECONDS, HeatmapKeyParts, heatmap_image_url_key, heatmap_meta_key
from .constants import ANALYTICS_VERSION
from .analyzer import build_heatmaps
from .demo_fetch import get_demo_dem_path
from .faceit_client import FaceitClient
from .models import AnalyticsAggregate, HeatmapAggregate, ProcessingJob
from .services.heatmaps import (
    HeatmapTimeDataMissing,
    _collect_points_from_cache,
    build_time_slice_from_bounds,
    build_time_slice_from_bucket,
    ensure_heatmap_image,
    get_or_build_heatmap,
    normalize_map_name,
    normalize_metric,
    normalize_period,
    normalize_side,
    normalize_time_slice,
    normalize_version,
    parse_time_slice,
)
from .services.time_buckets import normalize_time_bucket
from .services.paths import get_demos_dir
from .tasks import task_full_pipeline
from .utils import to_jsonable
from users.models import PlayerProfile


def _with_cache_buster(url: str | None, version: int | None) -> str | None:
    if not url:
        return url
    version = int(version if version is not None else time.time())
    separator = "&" if "?" in url else "?"
    return f"{url}{separator}v={version}"


@require_GET
def faceit_heatmaps(request):
    """
    GET /api/faceit/heatmaps?nickname=NAME
    Возвращает JSON + ссылки на PNG в MEDIA.
    """
    nickname = request.GET.get("nickname", "").strip()
    if not nickname:
        return JsonResponse({"error": "Никнейм обязателен."}, status=400)

    client = FaceitClient(api_key=getattr(settings, "FACEIT_API_KEY", None))

    player = client.search_player(nickname, game="cs2")
    if not player.get("player_id"):
        player = client.search_player(nickname, game="csgo")
    if not player.get("player_id"):
        player = client.search_player(nickname, game=None)
    player_id = player.get("player_id")
    games = player.get("games", {})
    steamid64 = (games.get("cs2") or games.get("csgo") or {}).get("game_player_id")

    if not player_id or not steamid64:
        return JsonResponse(
            {"error": "Could not resolve player_id or steamid64 for this nickname"},
            status=404,
        )

    hist = client.player_history(player_id, game="cs2", limit=1)
    items = hist.get("items") or []
    if not items:
        return JsonResponse({"error": "No matches found in history"}, status=404)

    match_id = items[0].get("match_id")
    if not match_id:
        return JsonResponse({"error": "No match_id in history item"}, status=500)

    match = client.match_details(match_id)
    raw_demo_urls = match.get("demo_url") or match.get("demo_urls") or []
    demo_urls = []
    if isinstance(raw_demo_urls, str):
        demo_urls = [raw_demo_urls]
    elif isinstance(raw_demo_urls, dict):
        for key in ("url", "demo_url", "resource_url", "download_url"):
            value = raw_demo_urls.get(key)
            if value:
                demo_urls.append(value)
    elif isinstance(raw_demo_urls, list):
        for item in raw_demo_urls:
            if isinstance(item, str):
                demo_urls.append(item)
            elif isinstance(item, dict):
                for key in ("url", "demo_url", "resource_url", "download_url"):
                    value = item.get(key)
                    if value:
                        demo_urls.append(value)
    if not demo_urls:
        return JsonResponse(
            {"error": "No demo_url in match details (maybe demo not available yet)"},
            status=404,
        )

    download_url = None
    last_error = None
    for resource_url in demo_urls:
        try:
            download_url = client.get_download_url(resource_url)
            break
        except HTTPError as exc:
            last_error = exc
            continue
    if not download_url:
        details = str(last_error) if last_error else "No downloadable demo URL"
        return JsonResponse(
            {
                "error": "Failed to obtain downloadable demo URL",
                "details": details,
            },
            status=404,
        )

    media_root = Path(getattr(settings, "MEDIA_ROOT", "media"))
    out_dir = media_root / "faceit_heatmaps" / nickname / match_id
    out_dir.mkdir(parents=True, exist_ok=True)

    kills_png = out_dir / "kills_heatmap_512.png"
    presence_png = out_dir / "presence_heatmap_512.png"
    presence_ct_png = out_dir / "presence_heatmap_ct_512.png"
    presence_t_png = out_dir / "presence_heatmap_t_512.png"

    if kills_png.exists() and presence_png.exists():
        summary = {
            "nickname": nickname,
            "player_id": player_id,
            "steamid64": steamid64,
            "match_id": match_id,
            "cached": True,
            "images": {
                "kills": (
                    f"{settings.MEDIA_URL}faceit_heatmaps/{nickname}/{match_id}/"
                    "kills_heatmap_512.png"
                ),
                "presence": (
                    f"{settings.MEDIA_URL}faceit_heatmaps/{nickname}/{match_id}/"
                    "presence_heatmap_512.png"
                ),
                "presence_ct": (
                    f"{settings.MEDIA_URL}faceit_heatmaps/{nickname}/{match_id}/"
                    "presence_heatmap_ct_512.png"
                )
                if presence_ct_png.exists()
                else None,
                "presence_t": (
                    f"{settings.MEDIA_URL}faceit_heatmaps/{nickname}/{match_id}/"
                    "presence_heatmap_t_512.png"
                )
                if presence_t_png.exists()
                else None,
            },
        }
        return JsonResponse(summary)

    work_dir = media_root / "faceit_cache" / nickname / match_id
    dem_path = get_demo_dem_path(download_url, work_dir)

    stats = build_heatmaps(dem_path, out_dir, steamid64)

    resp = {
        "nickname": nickname,
        "player_id": player_id,
        "steamid64": steamid64,
        "match_id": match_id,
        "cached": False,
        "map": stats["map"],
        "kills": stats["counts"]["kills"],
        "images": {
            "kills": (
                f"{settings.MEDIA_URL}faceit_heatmaps/{nickname}/{match_id}/"
                f"{stats['files']['kills']}"
            ),
            "presence": (
                f"{settings.MEDIA_URL}faceit_heatmaps/{nickname}/{match_id}/"
                f"{stats['files']['presence']}"
            ),
            "presence_ct": (
                f"{settings.MEDIA_URL}faceit_heatmaps/{nickname}/{match_id}/"
                f"{stats['files']['presence_ct']}"
            ),
            "presence_t": (
                f"{settings.MEDIA_URL}faceit_heatmaps/{nickname}/{match_id}/"
                f"{stats['files']['presence_t']}"
            ),
        },
    }
    return JsonResponse(resp)


@login_required
@require_POST
def start_analytics_processing(request, profile_id: int):
    profile = get_object_or_404(PlayerProfile, id=profile_id)
    if not request.user.is_superuser and request.user != profile.user:
        return JsonResponse({"error": "forbidden"}, status=403)

    map_name = (request.POST.get("map") or "").strip()
    if not map_name:
        try:
            payload = json.loads(request.body.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            payload = {}
        map_name = str(payload.get("map") or "").strip()
    map_name = map_name or "de_mirage"

    job = ProcessingJob.objects.create(
        profile=profile,
        job_type=ProcessingJob.JOB_FULL_PIPELINE,
        status=ProcessingJob.STATUS_PENDING,
        progress=0,
        requested_by=request.user,
    )

    try:
        task = task_full_pipeline.delay(
            profile.id,
            job.id,
            period="last_20",
            map_name=map_name,
            resolution=64,
        )
        ProcessingJob.objects.filter(id=job.id).update(celery_task_id=task.id)
    except Exception:
        ProcessingJob.objects.filter(id=job.id).update(
            status=ProcessingJob.STATUS_FAILED,
            error="Worker/Redis недоступен",
        )
        return JsonResponse({"error": "Worker/Redis недоступен"}, status=503)

    return JsonResponse({"job_id": job.id, "status": job.status, "progress": job.progress})


@login_required
@require_GET
def analytics_job_status(request, job_id: int):
    job = get_object_or_404(ProcessingJob, id=job_id)
    if not request.user.is_superuser and request.user != job.profile.user:
        return JsonResponse({"error": "forbidden"}, status=403)

    status_map = {
        ProcessingJob.STATUS_STARTED: ProcessingJob.STATUS_RUNNING,
        ProcessingJob.STATUS_PROCESSING: ProcessingJob.STATUS_RUNNING,
        ProcessingJob.STATUS_DONE: ProcessingJob.STATUS_SUCCESS,
    }
    payload = {
        "status": status_map.get(job.status, job.status),
        "progress": job.progress,
    }
    if job.error:
        payload["error"] = job.error
    if job.started_at:
        payload["started_at"] = job.started_at.isoformat()
    if job.finished_at:
        payload["finished_at"] = job.finished_at.isoformat()
    return JsonResponse(payload)


def _get_request_profile(request) -> PlayerProfile:
    return get_object_or_404(PlayerProfile, user=request.user)


def _heatmap_response(request, profile: PlayerProfile) -> JsonResponse:
    period = normalize_period(request.GET.get("period", "last_20"))
    map_name = normalize_map_name(request.GET.get("map", "de_mirage"))
    side = normalize_side(request.GET.get("side", AnalyticsAggregate.SIDE_ALL))
    kind = normalize_metric(request.GET.get("kind") or request.GET.get("metric") or HeatmapAggregate.METRIC_KILLS)
    version = normalize_version(request.GET.get("v", ANALYTICS_VERSION))
    time_bucket = request.GET.get("time_bucket") or request.GET.get("bucket")
    time_from = request.GET.get("time_from")
    time_to = request.GET.get("time_to")
    slice_override = build_time_slice_from_bounds(time_from, time_to)
    if slice_override:
        time_slice = normalize_time_slice(slice_override)
        bucket_value = "custom"
    elif time_bucket:
        bucket_value = normalize_time_bucket(time_bucket)
        if bucket_value != "all":
            time_slice = normalize_time_slice(build_time_slice_from_bucket(bucket_value))
        else:
            time_slice = normalize_time_slice(request.GET.get("slice") or request.GET.get("t"))
    else:
        time_slice = normalize_time_slice(request.GET.get("slice") or request.GET.get("t"))
        bucket_value = "all"
    bucket_value = bucket_value or "all"
    try:
        resolution = int(request.GET.get("res", 64))
    except (TypeError, ValueError):
        resolution = 64
    if resolution not in {64, 128, 256}:
        resolution = 64
    render_options: dict[str, float] = {}
    for key, target in (
        ("blur", "blur"),
        ("gamma", "gamma"),
        ("alpha", "alpha"),
        ("clip", "clip_pct"),
    ):
        raw = request.GET.get(key)
        if raw is None or raw == "":
            continue
        try:
            render_options[target] = float(raw)
        except (TypeError, ValueError):
            continue

    parts = HeatmapKeyParts(
        profile_id=profile.id,
        map_name=map_name,
        metric=kind,
        side=side,
        period=period,
        time_slice=time_slice,
        version=version,
        resolution=resolution,
    )
    cache_key = heatmap_meta_key(parts)
    force_regen = request.GET.get("force") == "1"
    if render_options:
        force_regen = True

    aggregate = HeatmapAggregate.objects.filter(
        profile=profile,
        period=period,
        map_name=map_name,
        metric=kind,
        side=side,
        analytics_version=version,
        resolution=resolution,
        time_slice=time_slice,
    ).first()

    response = {
        "status": "missing",
        "image_url": None,
        "updated_at": None,
        "resolution": resolution,
        "version": None,
        "metric": kind,
        "kind": kind,
        "side": side,
        "map": map_name,
        "period": period,
        "slice": time_slice,
        "time_bucket": bucket_value,
        "time_from": time_from,
        "time_to": time_to,
        "res": resolution,
        "meta": {
            "map": map_name,
            "side": side,
            "period": period,
            "slice": time_slice,
            "kind": kind,
            "metric": kind,
            "resolution": resolution,
            "version": version,
            "time_bucket": bucket_value,
            "time_from": time_from,
            "time_to": time_to,
            "cache_has_time_data": True,
            "time_slice_applied": parse_time_slice(time_slice) is not None,
            "missing_time_data_reason": None,
        },
    }

    render_options_payload = render_options or None

    time_meta = None
    if parse_time_slice(time_slice) is not None:
        steamid64 = (
            getattr(profile, "steamid64", None)
            or getattr(profile, "steam_id64", None)
            or getattr(profile, "steam_id", None)
        )
        if steamid64:
            demos_dir = get_demos_dir(profile, map_name)
            _points, _size, time_meta = _collect_points_from_cache(
                demos_dir,
                str(steamid64),
                map_name,
                period,
                side,
                kind,
                time_slice,
            )
            response["meta"].update(time_meta)
            if time_meta.get("missing_time_data_reason"):
                response["status"] = "processing"
                return JsonResponse(to_jsonable(response))
        if (
            aggregate
            and not force_regen
            and (aggregate.max_value is None or aggregate.max_value <= 0)
            and time_meta is not None
            and time_meta.get("points_total", 0) > 0
        ):
            force_regen = True
            try:
                cache.delete(cache_key)
            except Exception:
                pass

    if not force_regen:
        try:
            cached = cache.get(cache_key)
        except Exception:
            cached = None
        if cached:
            image_url = cached.get("image_url")
            if image_url:
                base_url = settings.MEDIA_URL or ""
                path = image_url.split("?", 1)[0]
                if base_url and path.startswith(base_url):
                    storage_path = path[len(base_url) :]
                    if not default_storage.exists(storage_path):
                        cached = None
            if cached:
                if time_meta and isinstance(cached, dict):
                    cached.setdefault("meta", {}).update(time_meta)
                return JsonResponse(to_jsonable(cached))

    try:
        if force_regen:
            aggregate = get_or_build_heatmap(
                profile_id=profile.id,
                map_name=map_name,
                metric=kind,
                side=side,
                period=period,
                time_slice=time_slice,
                version=version,
                resolution=resolution,
                force_rebuild=True,
                render_options=render_options_payload,
            )
        elif aggregate:
            aggregate = ensure_heatmap_image(aggregate, force=False, **(render_options_payload or {}))
    except HeatmapTimeDataMissing as exc:
        response["status"] = "processing"
        response["meta"]["cache_has_time_data"] = bool(exc.meta.get("cache_has_time_data"))
        response["meta"]["time_slice_applied"] = bool(exc.meta.get("time_slice_applied"))
        response["meta"]["missing_time_data_reason"] = exc.meta.get("missing_time_data_reason")
        return JsonResponse(to_jsonable(response))

    if aggregate and aggregate.image:
        updated_at = aggregate.updated_at.isoformat() if aggregate.updated_at else None
        try:
            file_version = int(Path(aggregate.image.path).stat().st_mtime)
        except Exception:
            file_version = int(time.time())
        image_url = _with_cache_buster(aggregate.image.url, file_version)
        response.update(
            {
                "status": "ready",
                "image_url": image_url,
                "updated_at": updated_at,
                "version": file_version,
            }
        )
        response = to_jsonable(response)
        try:
            cache.set(cache_key, response, DEFAULT_TTL_SECONDS)
            cache.set(heatmap_image_url_key(parts), image_url, DEFAULT_TTL_SECONDS)
        except Exception:
            pass
        return JsonResponse(response)

    job = (
        ProcessingJob.objects.filter(
            profile=profile,
            job_type=ProcessingJob.JOB_FULL_PIPELINE,
            status__in=(
                ProcessingJob.STATUS_PENDING,
                ProcessingJob.STATUS_RUNNING,
                ProcessingJob.STATUS_STARTED,
                ProcessingJob.STATUS_PROCESSING,
            ),
        )
        .order_by("-created_at")
        .first()
    )
    if job:
        response["status"] = "processing"
        try:
            cache.set(cache_key, to_jsonable(response), DEFAULT_TTL_SECONDS)
        except Exception:
            pass

    return JsonResponse(to_jsonable(response))


@login_required
@require_GET
def heatmaps_me(request):
    profile = _get_request_profile(request)
    return _heatmap_response(request, profile)


@login_required
@require_GET
def profile_heatmaps(request, profile_id: int):
    profile = get_object_or_404(PlayerProfile, id=profile_id)
    if not request.user.is_superuser and request.user != profile.user:
        return JsonResponse({"error": "forbidden"}, status=403)
    return _heatmap_response(request, profile)


@login_required
@require_GET
def analytics_me(request):
    profile = _get_request_profile(request)
    period = request.GET.get("period", "last_20").strip() or "last_20"
    map_name = request.GET.get("map", "de_mirage").strip() or "de_mirage"
    version = request.GET.get("v", ANALYTICS_VERSION).strip() or ANALYTICS_VERSION
    aggregates = list(
        AnalyticsAggregate.objects.filter(
            profile=profile,
            period=period,
            analytics_version=version,
            map_name=map_name,
        )
    )
    payload = {
        "profile_id": profile.id,
        "period": period,
        "map": map_name,
        "version": version,
        "ready": bool(aggregates),
        "aggregates": [
            {
                "map": aggregate.map_name,
                "side": aggregate.side,
                "period": aggregate.period,
                "version": aggregate.analytics_version,
                "metrics": to_jsonable(aggregate.metrics_json or {}),
                "updated_at": aggregate.updated_at.isoformat() if aggregate.updated_at else None,
            }
            for aggregate in aggregates
        ],
    }
    return JsonResponse(to_jsonable(payload))
