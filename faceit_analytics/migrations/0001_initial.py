from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


def heatmap_upload_to(instance, filename):
    return (
        f"heatmaps/{instance.profile_id}/{instance.map_name}/{instance.side}/"
        f"{instance.period}/{filename}"
    )


class Migration(migrations.Migration):
    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ("users", "0001_initial"),
    ]

    operations = [
        migrations.CreateModel(
            name="AnalyticsAggregate",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("map_name", models.CharField(max_length=64)),
                (
                    "side",
                    models.CharField(
                        choices=[("T", "T"), ("CT", "CT"), ("ALL", "ALL")],
                        default="ALL",
                        max_length=3,
                    ),
                ),
                ("period", models.CharField(max_length=20)),
                ("analytics_version", models.CharField(default="v1", max_length=12)),
                ("metrics_json", models.JSONField(default=dict)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                (
                    "profile",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="analytics_aggregates",
                        to="users.playerprofile",
                    ),
                ),
            ],
            options={
                "unique_together": {("profile", "map_name", "side", "period", "analytics_version")},
            },
        ),
        migrations.CreateModel(
            name="HeatmapAggregate",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("map_name", models.CharField(max_length=64)),
                (
                    "side",
                    models.CharField(
                        choices=[("T", "T"), ("CT", "CT"), ("ALL", "ALL")],
                        default="ALL",
                        max_length=3,
                    ),
                ),
                ("period", models.CharField(max_length=20)),
                ("analytics_version", models.CharField(default="v1", max_length=12)),
                ("resolution", models.IntegerField(default=64)),
                ("grid_json", models.JSONField(default=list)),
                ("image", models.FileField(blank=True, upload_to=heatmap_upload_to)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                (
                    "profile",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="heatmap_aggregates",
                        to="users.playerprofile",
                    ),
                ),
            ],
            options={
                "unique_together": {
                    ("profile", "map_name", "side", "period", "analytics_version", "resolution")
                },
            },
        ),
        migrations.CreateModel(
            name="ProcessingJob",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                (
                    "job_type",
                    models.CharField(
                        choices=[
                            ("SYNC_FACEIT", "Sync Faceit"),
                            ("BUILD_AGGREGATES", "Build aggregates"),
                            ("RENDER_HEATMAPS", "Render heatmaps"),
                            ("FULL_PIPELINE", "Full pipeline"),
                        ],
                        max_length=32,
                    ),
                ),
                ("celery_task_id", models.CharField(blank=True, max_length=255, null=True)),
                (
                    "status",
                    models.CharField(
                        choices=[
                            ("PENDING", "Pending"),
                            ("RUNNING", "Running"),
                            ("SUCCESS", "Success"),
                            ("FAILED", "Failed"),
                        ],
                        default="PENDING",
                        max_length=16,
                    ),
                ),
                ("progress", models.PositiveSmallIntegerField(default=0)),
                ("error", models.TextField(blank=True)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                (
                    "profile",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="processing_jobs",
                        to="users.playerprofile",
                    ),
                ),
                (
                    "requested_by",
                    models.ForeignKey(
                        blank=True,
                        null=True,
                        on_delete=django.db.models.deletion.SET_NULL,
                        related_name="requested_jobs",
                        to=settings.AUTH_USER_MODEL,
                    ),
                ),
            ],
        ),
    ]
