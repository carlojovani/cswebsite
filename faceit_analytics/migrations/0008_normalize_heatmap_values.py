from django.db import migrations, models


def normalize_values(apps, schema_editor):
    AnalyticsAggregate = apps.get_model("faceit_analytics", "AnalyticsAggregate")
    HeatmapAggregate = apps.get_model("faceit_analytics", "HeatmapAggregate")

    for model in (AnalyticsAggregate, HeatmapAggregate):
        model.objects.filter(side__iexact="all").update(side="all")
        model.objects.filter(side__iexact="ct").update(side="ct")
        model.objects.filter(side__iexact="t").update(side="t")

    HeatmapAggregate.objects.filter(metric__iexact="kills").update(metric="kills")
    HeatmapAggregate.objects.filter(metric__iexact="deaths").update(metric="deaths")
    HeatmapAggregate.objects.filter(metric__iexact="presence").update(metric="presence")


def noop_reverse(apps, schema_editor):
    pass


class Migration(migrations.Migration):
    dependencies = [
        ("faceit_analytics", "0007_heatmapaggregate_time_slice"),
    ]

    operations = [
        migrations.AlterField(
            model_name="analyticsaggregate",
            name="side",
            field=models.CharField(
                choices=[("t", "T"), ("ct", "CT"), ("all", "ALL")],
                default="all",
                max_length=3,
            ),
        ),
        migrations.AlterField(
            model_name="heatmapaggregate",
            name="side",
            field=models.CharField(
                choices=[("t", "T"), ("ct", "CT"), ("all", "ALL")],
                default="all",
                max_length=3,
            ),
        ),
        migrations.RunPython(normalize_values, noop_reverse),
    ]
