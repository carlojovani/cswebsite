from django.urls import path
from .views import faceit_heatmaps
from .views_local import local_heatmaps, local_heatmaps_aggregate

urlpatterns = [
    path("faceit/heatmaps", faceit_heatmaps, name="faceit_heatmaps"),
    path("local/heatmaps", local_heatmaps, name="local_heatmaps"),
    path("local/heatmaps_aggregate", local_heatmaps_aggregate, name="local_heatmaps_aggregate"),
]
