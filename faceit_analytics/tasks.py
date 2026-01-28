from celery import shared_task

from faceit_analytics.models import ProcessingJob
from faceit_analytics.services.pipeline import run_full_pipeline


@shared_task(bind=True)
def task_full_pipeline(
    self,
    profile_id: int,
    job_id: int,
    period: str = "last_20",
    resolution: int = 64,
    force_rebuild: bool = False,
    force_heatmaps: bool = False,
    force_demo_features: bool = False,
) -> None:
    ProcessingJob.objects.filter(id=job_id).filter(
        celery_task_id__isnull=True
    ).update(celery_task_id=self.request.id)
    ProcessingJob.objects.filter(id=job_id, celery_task_id="").update(celery_task_id=self.request.id)
    run_full_pipeline(
        profile_id=profile_id,
        job_id=job_id,
        period=period,
        resolution=resolution,
        force_rebuild=force_rebuild,
        force_heatmaps=force_heatmaps,
        force_demo_features=force_demo_features,
    )
