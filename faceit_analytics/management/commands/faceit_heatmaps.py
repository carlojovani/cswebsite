from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand

from faceit_analytics.analyzer import build_heatmaps
from faceit_analytics.demo_fetch import get_demo_dem_path
from faceit_analytics.faceit_client import FaceitClient


class Command(BaseCommand):
    help = "Generate CS2 heatmaps (presence + kills) for latest FACEIT match by nickname"

    def add_arguments(self, parser):
        parser.add_argument("--nickname", required=True)
        parser.add_argument("--out", default=None)

    def handle(self, *args, **opts):
        nickname = opts["nickname"]
        out = opts["out"]

        client = FaceitClient(api_key=getattr(settings, "FACEIT_API_KEY", None))
        player = client.search_player(nickname, game="cs2")
        player_id = player.get("player_id")
        steamid64 = (player.get("games", {}).get("cs2") or {}).get("game_player_id")

        hist = client.player_history(player_id, game="cs2", limit=1)
        match_id = (hist.get("items") or [])[0]["match_id"]
        match = client.match_details(match_id)
        demo_urls = match.get("demo_url") or []
        resource_url = demo_urls[0]
        download_url = client.get_download_url(resource_url)

        media_root = Path(getattr(settings, "MEDIA_ROOT", "media"))
        work_dir = media_root / "faceit_cache" / nickname / match_id
        dem_path = get_demo_dem_path(download_url, work_dir)

        out_dir = Path(out) if out else (media_root / "faceit_heatmaps" / nickname / match_id)
        stats = build_heatmaps(dem_path, out_dir, steamid64)

        self.stdout.write(self.style.SUCCESS(f"OK: {nickname} {match_id} map={stats['map']}"))
        self.stdout.write(f"Saved to: {out_dir}")
