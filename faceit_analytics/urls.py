from django.urls import path

from .views import faceit_heatmaps

urlpatterns = [
    path("faceit/heatmaps", faceit_heatmaps, name="faceit_heatmaps"),
]
