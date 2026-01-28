from django.db import migrations, models
from django.utils import timezone


def heatmap_upload_to(instance, filename):
    return (
        "heatmaps/"
        f"{instance.profile_id}/{instance.map_name}/{instance.side}/"
        f"{instance.period}/{instance.analytics_version}.png"
    )


class Migration(migrations.Migration):
    dependencies = [
        ("faceit_analytics", "0003_processingjob_status_fields"),
    ]

    operations = [
        migrations.RenameField(
            model_name="heatmapaggregate",
            old_name="grid_json",
            new_name="grid",
        ),
        migrations.AddField(
            model_name="heatmapaggregate",
            name="max_value",
            field=models.FloatField(null=True),
        ),
        migrations.AddField(
            model_name="heatmapaggregate",
            name="created_at",
            field=models.DateTimeField(auto_now_add=True, default=timezone.now),
        ),
        migrations.AlterField(
            model_name="heatmapaggregate",
            name="image",
            field=models.ImageField(blank=True, null=True, upload_to=heatmap_upload_to),
        ),
        migrations.AlterField(
            model_name="heatmapaggregate",
            name="resolution",
            field=models.PositiveSmallIntegerField(default=64),
        ),
    ]
