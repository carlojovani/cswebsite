from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("faceit_analytics", "0006_alter_heatmapaggregate_unique_together_and_more"),
    ]

    operations = [
        migrations.AddField(
            model_name="heatmapaggregate",
            name="time_slice",
            field=models.CharField(default="all", max_length=20),
        ),
        migrations.AlterUniqueTogether(
            name="heatmapaggregate",
            unique_together={
                (
                    "profile",
                    "map_name",
                    "metric",
                    "side",
                    "period",
                    "time_slice",
                    "analytics_version",
                    "resolution",
                )
            },
        ),
    ]
