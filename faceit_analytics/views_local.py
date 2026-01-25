from pathlib import Path

from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.http import require_GET

from .analyzer import build_heatmaps
from .demo_fetch import get_local_dem_path


@require_GET
def local_heatmaps(request):
    """
    GET /api/local/heatmaps?steamid64=XXXX
    Берёт демку из media/local_demos/match.dem(.zst) и строит heatmaps для steamid64.
    """
    steamid64 = request.GET.get("steamid64", "").strip()
    if not steamid64:
        return JsonResponse({"error": "steamid64 is required"}, status=400)

    media_root = Path(getattr(settings, "MEDIA_ROOT", "media"))
    demo_dir = media_root / "local_demos"
    out_dir = media_root / "heatmaps_local" / steamid64

    try:
        dem_path = get_local_dem_path(demo_dir)
        stats = build_heatmaps(dem_path, out_dir, steamid64)
    except Exception as exc:
        return JsonResponse(
            {"error": "Failed to build heatmaps", "details": str(exc)}, status=500
        )

    return JsonResponse({
        "steamid64": steamid64,
        "map": stats.get("map"),
        "counts": stats.get("counts"),
        "analyzer_version": stats.get("analyzer_version"),
        "images": {
            "presence": f"{settings.MEDIA_URL}heatmaps_local/{steamid64}/presence_heatmap.png",
            "presence_ct": f"{settings.MEDIA_URL}heatmaps_local/{steamid64}/presence_heatmap_ct.png",
            "presence_t": f"{settings.MEDIA_URL}heatmaps_local/{steamid64}/presence_heatmap_t.png",
            "kills": f"{settings.MEDIA_URL}heatmaps_local/{steamid64}/kills_heatmap.png",
            "deaths": f"{settings.MEDIA_URL}heatmaps_local/{steamid64}/deaths_heatmap.png",
        },
    })
