from pathlib import Path

from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.http import require_GET

from .analyzer import build_heatmaps, build_heatmaps_aggregate
from .demo_fetch import get_local_dem_path


@require_GET
def local_heatmaps(request):
    """
    GET /api/local/heatmaps?steamid64=XXXX
    Берёт демку из MEDIA_ROOT/local_demos/<steamid64>/match.dem(.zst) и строит heatmaps для steamid64.
    """
    steamid64 = request.GET.get("steamid64", "").strip()
    if not steamid64:
        return JsonResponse({"error": "steamid64 is required"}, status=400)

    media_root = Path(getattr(settings, "MEDIA_ROOT", "media"))
    demos_root = Path(getattr(settings, "LOCAL_DEMOS_ROOT", media_root / "local_demos"))
    demo_dir = demos_root / steamid64
    out_dir = media_root / "heatmaps_local" / steamid64

    try:
        dem_path = get_local_dem_path(demo_dir)
        stats = build_heatmaps(dem_path, out_dir, steamid64)
    except Exception as exc:
        return JsonResponse(
            {"error": "Failed to build heatmaps", "details": str(exc)}, status=500
        )

    files = stats.get("files", {})
    return JsonResponse({
        "steamid64": steamid64,
        "map": stats.get("map"),
        "counts": stats.get("counts"),
        "analyzer_version": stats.get("analyzer_version"),
        "images": {
            "presence": f"{settings.MEDIA_URL}heatmaps_local/{steamid64}/{files.get('presence')}",
            "presence_ct": f"{settings.MEDIA_URL}heatmaps_local/{steamid64}/{files.get('presence_ct')}",
            "presence_t": f"{settings.MEDIA_URL}heatmaps_local/{steamid64}/{files.get('presence_t')}",
            "kills": f"{settings.MEDIA_URL}heatmaps_local/{steamid64}/{files.get('kills')}",
            "deaths": f"{settings.MEDIA_URL}heatmaps_local/{steamid64}/{files.get('deaths')}",
        },
    })


@require_GET
def local_heatmaps_aggregate(request):
    """
    GET /api/local/heatmaps_aggregate?steamid64=XXXX&map=de_mirage&limit=5
    Берёт несколько демо в MEDIA_ROOT/local_demos/<steamid64>/<map>/ и строит агрегированные heatmaps.
    """
    steamid64 = request.GET.get("steamid64", "").strip()
    map_name = request.GET.get("map", "").strip()
    limit = request.GET.get("limit", "5").strip()
    debug = request.GET.get("debug", "").strip() == "1"

    if not steamid64:
        return JsonResponse({"error": "steamid64 is required"}, status=400)
    if not map_name:
        return JsonResponse({"error": "map is required"}, status=400)

    try:
        limit_value = max(int(limit), 1)
    except ValueError:
        return JsonResponse({"error": "limit must be an integer"}, status=400)

    media_root = Path(getattr(settings, "MEDIA_ROOT", "media"))
    demos_root = Path(getattr(settings, "LOCAL_DEMOS_ROOT", media_root / "local_demos"))
    demos_dir = demos_root / steamid64 / map_name
    out_dir = media_root / "heatmaps_local" / steamid64 / "aggregate" / map_name
    cache_dir = media_root / "heatmaps_cache"

    try:
        stats = build_heatmaps_aggregate(
            steamid64=steamid64,
            map_name=map_name,
            limit=limit_value,
            demos_dir=demos_dir,
            out_dir=out_dir,
            cache_dir=cache_dir,
        )
    except Exception as exc:
        return JsonResponse(
            {"error": "Failed to build aggregate heatmaps", "details": str(exc)}, status=500
        )

    files = stats.get("files", {})
    response = {
        "steamid64": steamid64,
        "map": stats.get("map"),
        "analyzer_version": stats.get("analyzer_version"),
        "images": {
            "presence": f"{settings.MEDIA_URL}heatmaps_local/{steamid64}/aggregate/{map_name}/{files.get('presence')}",
            "presence_ct": f"{settings.MEDIA_URL}heatmaps_local/{steamid64}/aggregate/{map_name}/{files.get('presence_ct')}",
            "presence_t": f"{settings.MEDIA_URL}heatmaps_local/{steamid64}/aggregate/{map_name}/{files.get('presence_t')}",
            "kills": f"{settings.MEDIA_URL}heatmaps_local/{steamid64}/aggregate/{map_name}/{files.get('kills')}",
            "deaths": f"{settings.MEDIA_URL}heatmaps_local/{steamid64}/aggregate/{map_name}/{files.get('deaths')}",
        },
    }
    if debug:
        response["debug"] = {
            "counts": stats.get("counts"),
            "cache_hits": stats.get("cache_hits"),
            "cached": stats.get("cached"),
        }
    return JsonResponse(response)
