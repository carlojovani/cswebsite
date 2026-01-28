from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("faceit_analytics", "0002_alter_heatmapaggregate_image"),
    ]

    operations = [
        migrations.AddField(
            model_name="processingjob",
            name="started_at",
            field=models.DateTimeField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="processingjob",
            name="finished_at",
            field=models.DateTimeField(blank=True, null=True),
        ),
        migrations.AlterField(
            model_name="processingjob",
            name="status",
            field=models.CharField(
                choices=[
                    ("PENDING", "Pending"),
                    ("STARTED", "Started"),
                    ("PROCESSING", "Processing"),
                    ("DONE", "Done"),
                    ("FAILED", "Failed"),
                    ("RUNNING", "Running (legacy)"),
                    ("SUCCESS", "Success (legacy)"),
                ],
                default="PENDING",
                max_length=16,
            ),
        ),
    ]
