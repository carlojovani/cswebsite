import logging
import traceback

from celery import shared_task
from django.core.cache import cache

from faceit_analytics.models import ProcessingJob
from faceit_analytics.services.aggregates import build_metrics
from faceit_analytics.services.pipeline import build_heatmaps, sync_faceit_profile
from users.models import PlayerProfile

logger = logging.getLogger(__name__)


def _load_job(job_id: int) -> ProcessingJob:
    return ProcessingJob.objects.select_related("profile").get(id=job_id)


def _set_job(job: ProcessingJob, **fields) -> None:
    for key, value in fields.items():
        setattr(job, key, value)
    job.save()


def _invalidate_cache(profile_id: int, period: str) -> None:
    cache.delete(f"agg:{profile_id}:{period}")
    if hasattr(cache, "delete_pattern"):
        cache.delete_pattern(f"heatmap:{profile_id}:{period}:*")


@shared_task
def task_sync_faceit(profile_id: int, job_id: int) -> None:
    job = _load_job(job_id)
    _set_job(job, status=ProcessingJob.STATUS_RUNNING, progress=1, error="")

    try:
        profile = PlayerProfile.objects.get(id=profile_id)
        sync_faceit_profile(profile)
        _set_job(job, status=ProcessingJob.STATUS_SUCCESS, progress=100)
    except Exception:
        logger.exception("Failed to sync Faceit for profile %s", profile_id)
        _set_job(job, status=ProcessingJob.STATUS_FAILED, progress=0, error=traceback.format_exc())
        raise


@shared_task
def task_build_aggregates(profile_id: int, job_id: int, period: str) -> None:
    job = _load_job(job_id)
    _set_job(job, status=ProcessingJob.STATUS_RUNNING, progress=1, error="")

    try:
        profile = PlayerProfile.objects.get(id=profile_id)
        build_metrics(profile, period=period)
        _invalidate_cache(profile_id, period)
        _set_job(job, status=ProcessingJob.STATUS_SUCCESS, progress=100)
    except Exception:
        logger.exception("Failed to build aggregates for profile %s", profile_id)
        _set_job(job, status=ProcessingJob.STATUS_FAILED, progress=0, error=traceback.format_exc())
        raise


@shared_task
def task_render_heatmaps(profile_id: int, job_id: int, period: str, resolution: int = 64) -> None:
    job = _load_job(job_id)
    _set_job(job, status=ProcessingJob.STATUS_RUNNING, progress=1, error="")

    try:
        profile = PlayerProfile.objects.get(id=profile_id)
        build_heatmaps(profile, period=period, resolution=resolution)
        _invalidate_cache(profile_id, period)
        _set_job(job, status=ProcessingJob.STATUS_SUCCESS, progress=100)
    except Exception:
        logger.exception("Failed to render heatmaps for profile %s", profile_id)
        _set_job(job, status=ProcessingJob.STATUS_FAILED, progress=0, error=traceback.format_exc())
        raise


@shared_task
def task_full_pipeline(profile_id: int, job_id: int, period: str = "last_20", resolution: int = 64) -> None:
    job = _load_job(job_id)
    _set_job(job, status=ProcessingJob.STATUS_RUNNING, progress=1, error="")

    try:
        profile = PlayerProfile.objects.get(id=profile_id)

        sync_faceit_profile(profile)
        _set_job(job, progress=30)

        build_metrics(profile, period=period)
        _set_job(job, progress=70)

        build_heatmaps(profile, period=period, resolution=resolution)
        _invalidate_cache(profile_id, period)
        _set_job(job, progress=100, status=ProcessingJob.STATUS_SUCCESS)
    except Exception:
        logger.exception("Failed to run full pipeline for profile %s", profile_id)
        _set_job(job, status=ProcessingJob.STATUS_FAILED, progress=0, error=traceback.format_exc())
        raise
