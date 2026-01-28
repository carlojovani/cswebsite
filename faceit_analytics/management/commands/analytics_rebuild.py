from __future__ import annotations

from django.core.management.base import BaseCommand

from faceit_analytics.constants import ANALYTICS_VERSION
from faceit_analytics.models import ProcessingJob
from faceit_analytics.services.pipeline import run_full_pipeline
from faceit_analytics.tasks import task_full_pipeline
from users.models import PlayerProfile


class Command(BaseCommand):
    help = "Trigger analytics rebuild (optionally force refresh) for a player profile."

    def add_arguments(self, parser):
        parser.add_argument("--profile-id", type=int, required=True)
        parser.add_argument("--period", type=str, default="last_20")
        parser.add_argument("--resolution", type=int, default=64)
        parser.add_argument("--force", action="store_true")
        parser.add_argument("--sync", action="store_true")

    def handle(self, *args, **options):
        profile_id = options["profile_id"]
        period = options["period"]
        resolution = options["resolution"]
        force_rebuild = options["force"]
        run_sync = options["sync"]

        profile = PlayerProfile.objects.get(id=profile_id)
        job = ProcessingJob.objects.create(
            profile=profile,
            job_type=ProcessingJob.JOB_FULL_PIPELINE,
            status=ProcessingJob.STATUS_PENDING,
            progress=0,
        )

        if run_sync:
            run_full_pipeline(
                profile_id=profile.id,
                job_id=job.id,
                period=period,
                resolution=resolution,
                force_rebuild=force_rebuild,
            )
            job.refresh_from_db()
            self.stdout.write(
                f"Analytics rebuild completed: job_id={job.id} status={job.status} version={ANALYTICS_VERSION}"
            )
        else:
            task_full_pipeline.delay(
                profile_id=profile.id,
                job_id=job.id,
                period=period,
                resolution=resolution,
                force_rebuild=force_rebuild,
            )
            self.stdout.write(
                f"Analytics rebuild queued: job_id={job.id} status={job.status} version={ANALYTICS_VERSION}"
            )
