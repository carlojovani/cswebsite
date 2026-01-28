from django.conf import settings
from django.db import models

from users.models import PlayerProfile


class AnalyticsAggregate(models.Model):
    SIDE_ALL = "ALL"
    SIDE_CT = "CT"
    SIDE_T = "T"

    SIDE_CHOICES = (
        (SIDE_T, "T"),
        (SIDE_CT, "CT"),
        (SIDE_ALL, "ALL"),
    )

    profile = models.ForeignKey(PlayerProfile, on_delete=models.CASCADE, related_name="analytics_aggregates")
    map_name = models.CharField(max_length=64)
    side = models.CharField(max_length=3, choices=SIDE_CHOICES, default=SIDE_ALL)
    period = models.CharField(max_length=20)
    analytics_version = models.CharField(max_length=12, default="v1")
    metrics_json = models.JSONField(default=dict)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ("profile", "map_name", "side", "period", "analytics_version")

    def __str__(self) -> str:
        return f"AnalyticsAggregate({self.profile_id}, {self.map_name}, {self.side}, {self.period})"


def heatmap_upload_to(instance: "HeatmapAggregate", filename: str) -> str:
    return (
        "heatmaps/"
        f"{instance.profile_id}/{instance.map_name}/{instance.side}/"
        f"{instance.period}/{filename}"
    )


class HeatmapAggregate(models.Model):
    profile = models.ForeignKey(PlayerProfile, on_delete=models.CASCADE, related_name="heatmap_aggregates")
    map_name = models.CharField(max_length=64)
    side = models.CharField(max_length=3, choices=AnalyticsAggregate.SIDE_CHOICES, default=AnalyticsAggregate.SIDE_ALL)
    period = models.CharField(max_length=20)
    analytics_version = models.CharField(max_length=12, default="v1")
    resolution = models.PositiveSmallIntegerField(default=64)
    grid = models.JSONField(default=list)
    max_value = models.FloatField(null=True)
    image = models.ImageField(upload_to=heatmap_upload_to, blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ("profile", "map_name", "side", "period", "analytics_version", "resolution")

    def __str__(self) -> str:
        return f"HeatmapAggregate({self.profile_id}, {self.map_name}, {self.side}, {self.period})"


class ProcessingJob(models.Model):
    JOB_SYNC_FACEIT = "SYNC_FACEIT"
    JOB_BUILD_AGGREGATES = "BUILD_AGGREGATES"
    JOB_RENDER_HEATMAPS = "RENDER_HEATMAPS"
    JOB_FULL_PIPELINE = "FULL_PIPELINE"

    JOB_TYPE_CHOICES = (
        (JOB_SYNC_FACEIT, "Sync Faceit"),
        (JOB_BUILD_AGGREGATES, "Build aggregates"),
        (JOB_RENDER_HEATMAPS, "Render heatmaps"),
        (JOB_FULL_PIPELINE, "Full pipeline"),
    )

    STATUS_PENDING = "PENDING"
    STATUS_STARTED = "STARTED"
    STATUS_PROCESSING = "PROCESSING"
    STATUS_DONE = "DONE"
    STATUS_FAILED = "FAILED"
    STATUS_RUNNING = "RUNNING"
    STATUS_SUCCESS = "SUCCESS"

    STATUS_CHOICES = (
        (STATUS_PENDING, "Pending"),
        (STATUS_STARTED, "Started"),
        (STATUS_PROCESSING, "Processing"),
        (STATUS_DONE, "Done"),
        (STATUS_FAILED, "Failed"),
        (STATUS_RUNNING, "Running (legacy)"),
        (STATUS_SUCCESS, "Success (legacy)"),
    )

    profile = models.ForeignKey(PlayerProfile, on_delete=models.CASCADE, related_name="processing_jobs")
    job_type = models.CharField(max_length=32, choices=JOB_TYPE_CHOICES)
    celery_task_id = models.CharField(max_length=255, blank=True, null=True)
    status = models.CharField(max_length=16, choices=STATUS_CHOICES, default=STATUS_PENDING)
    progress = models.PositiveSmallIntegerField(default=0)
    error = models.TextField(blank=True)
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    requested_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="requested_jobs",
    )

    def __str__(self) -> str:
        return f"ProcessingJob({self.profile_id}, {self.job_type}, {self.status})"
