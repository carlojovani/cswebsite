from django.urls import path
from .views import (
    analytics_job_status,
    faceit_heatmaps,
    profile_heatmaps,
    start_analytics_processing,
)
from .views_local import local_heatmaps, local_heatmaps_aggregate

urlpatterns = [
    path("faceit/heatmaps", faceit_heatmaps, name="faceit_heatmaps"),
    path("local/heatmaps", local_heatmaps, name="local_heatmaps"),
    path("local/heatmaps_aggregate", local_heatmaps_aggregate, name="local_heatmaps_aggregate"),
    path("analytics/<int:profile_id>/start", start_analytics_processing, name="start_analytics_processing"),
    path("analytics/jobs/<int:job_id>", analytics_job_status, name="analytics_job_status"),
    path("analytics/<int:profile_id>/heatmaps", profile_heatmaps, name="profile_heatmaps"),
]
