from django.core.management.base import BaseCommand

from faceit_analytics.constants import ANALYTICS_VERSION
from faceit_analytics.models import AnalyticsAggregate, HeatmapAggregate
from faceit_analytics.services.heatmaps import ensure_heatmap_image, get_or_build_heatmap


class Command(BaseCommand):
    help = "Regenerate heatmap images for aggregates."

    def add_arguments(self, parser) -> None:
        parser.add_argument("--profile-id", type=int, required=True)
        parser.add_argument("--period", default="last_20")
        parser.add_argument("--map", dest="map_name", default="de_mirage")
        parser.add_argument("--side", default=AnalyticsAggregate.SIDE_ALL)
        parser.add_argument("--res", type=int, default=64)
        parser.add_argument("--force", action="store_true")
        parser.add_argument("--analytics-version", dest="analytics_version", default=ANALYTICS_VERSION)

    def handle(self, *args, **options) -> None:
        profile_id = options["profile_id"]
        period = options["period"]
        map_name = options["map_name"]
        side = options["side"].upper()
        resolution = options["res"]
        force = options["force"]
        version = options["analytics_version"]

        aggregates = HeatmapAggregate.objects.filter(
            profile_id=profile_id,
            map_name=map_name,
            side=side,
            period=period,
            analytics_version=version,
            resolution=resolution,
        )
        if not aggregates.exists():
            aggregate = get_or_build_heatmap(
                profile_id=profile_id,
                map_name=map_name,
                side=side,
                period=period,
                version=version,
                resolution=resolution,
                force_rebuild=True,
            )
            aggregates = HeatmapAggregate.objects.filter(id=aggregate.id)

        for aggregate in aggregates:
            aggregate = ensure_heatmap_image(aggregate, force=force)
            image_name = aggregate.image.name if aggregate.image else None
            exists = aggregate.image.storage.exists(image_name) if aggregate.image else False
            self.stdout.write(
                f"heatmap id={aggregate.id} image={image_name} exists={exists}"
            )
