from pathlib import Path

from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.http import require_GET

from .analyzer import build_heatmaps
from .demo_fetch import get_demo_dem_path
from .faceit_client import FaceitClient


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
    player_id = player.get("player_id")
    games = player.get("games", {})
    steamid64 = (games.get("cs2") or {}).get("game_player_id")

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
    demo_urls = match.get("demo_url") or match.get("demo_urls") or []
    if isinstance(demo_urls, str):
        demo_urls = [demo_urls]
    if not demo_urls:
        return JsonResponse(
            {"error": "No demo_url in match details (maybe demo not available yet)"},
            status=404,
        )

    resource_url = demo_urls[0]
    signed_url = client.get_signed_download_url(resource_url)

    media_root = Path(getattr(settings, "MEDIA_ROOT", "media"))
    out_dir = media_root / "faceit_heatmaps" / nickname / match_id
    out_dir.mkdir(parents=True, exist_ok=True)

    kills_png = out_dir / "kills_heatmap.png"
    presence_png = out_dir / "presence_heatmap.png"

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
                    "kills_heatmap.png"
                ),
                "presence": (
                    f"{settings.MEDIA_URL}faceit_heatmaps/{nickname}/{match_id}/"
                    "presence_heatmap.png"
                ),
            },
        }
        return JsonResponse(summary)

    work_dir = media_root / "faceit_cache" / nickname / match_id
    dem_path = get_demo_dem_path(signed_url, work_dir)

    stats = build_heatmaps(dem_path, out_dir, steamid64)

    resp = {
        "nickname": nickname,
        "player_id": player_id,
        "steamid64": steamid64,
        "match_id": match_id,
        "cached": False,
        "map": stats["map"],
        "kills": stats["kills"],
        "images": {
            "kills": (
                f"{settings.MEDIA_URL}faceit_heatmaps/{nickname}/{match_id}/"
                "kills_heatmap.png"
            ),
            "presence": (
                f"{settings.MEDIA_URL}faceit_heatmaps/{nickname}/{match_id}/"
                "presence_heatmap.png"
            ),
        },
    }
    return JsonResponse(resp)
