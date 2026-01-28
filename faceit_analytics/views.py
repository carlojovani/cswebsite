from pathlib import Path

from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.core.cache import cache
from django.http import JsonResponse
from django.shortcuts import get_object_or_404
from django.views.decorators.http import require_GET, require_POST
from requests import HTTPError

from .analyzer import build_heatmaps
from .demo_fetch import get_demo_dem_path
from .faceit_client import FaceitClient
from .models import AnalyticsAggregate, HeatmapAggregate, ProcessingJob
from .tasks import task_full_pipeline
from users.models import PlayerProfile


@require_GET
def faceit_heatmaps(request):
    """
    GET /api/faceit/heatmaps?nickname=NAME
    Возвращает JSON + ссылки на PNG в MEDIA.
    """
    nickname = request.GET.get("nickname", "").strip()
    if not nickname:
        return JsonResponse({"error": "nickname is required"}, status=400)

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

    job = ProcessingJob.objects.create(
        profile=profile,
        job_type=ProcessingJob.JOB_FULL_PIPELINE,
        status=ProcessingJob.STATUS_PENDING,
        progress=0,
        requested_by=request.user,
    )

    try:
        task = task_full_pipeline.delay(profile.id, job.id, period="last_20", resolution=64)
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


@login_required
@require_GET
def profile_heatmaps(request, profile_id: int):
    profile = get_object_or_404(PlayerProfile, id=profile_id)
    if not request.user.is_superuser and request.user != profile.user:
        return JsonResponse({"error": "forbidden"}, status=403)

    period = request.GET.get("period", "last_20").strip() or "last_20"
    map_name = request.GET.get("map", "de_mirage").strip() or "de_mirage"
    cache_key = f"heatmap:{profile.id}:{period}:{map_name}"
    try:
        cached = cache.get(cache_key)
    except Exception:
        cached = None
    if cached:
        return JsonResponse(cached)

    aggregates = HeatmapAggregate.objects.filter(
        profile=profile,
        period=period,
        map_name=map_name,
    )

    images = {}
    for side, key in (
        (AnalyticsAggregate.SIDE_ALL, "presence"),
        (AnalyticsAggregate.SIDE_CT, "presence_ct"),
        (AnalyticsAggregate.SIDE_T, "presence_t"),
    ):
        aggregate = aggregates.filter(side=side).first()
        if aggregate and aggregate.image:
            images[key] = aggregate.image.url

    response = {
        "status": "ready" if images else "missing",
        "images": images,
        "map": map_name,
        "period": period,
    }
    try:
        cache.set(cache_key, response, 120)
    except Exception:
        pass
    return JsonResponse(response)
