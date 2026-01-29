import os
import tempfile

import django
from django.contrib.auth import get_user_model
from django.test import Client, TestCase, override_settings

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backend.settings")
django.setup()

from faceit_analytics.models import HeatmapAggregate  # noqa: E402
from faceit_analytics.services.heatmaps import ensure_heatmap_image  # noqa: E402
from users.models import PlayerProfile  # noqa: E402


@override_settings(ALLOWED_HOSTS=["testserver", "localhost"])
class HeatmapMeApiTests(TestCase):
    def setUp(self):
        User = get_user_model()
        self.user1 = User.objects.create_user(
            username="user1",
            email="user1@example.com",
            password="pass",
            user_type="player",
        )
        self.user2 = User.objects.create_user(
            username="user2",
            email="user2@example.com",
            password="pass",
            user_type="player",
        )
        self.profile1 = PlayerProfile.objects.create(user=self.user1, steam_id="76561198000000001")
        self.profile2 = PlayerProfile.objects.create(user=self.user2, steam_id="76561198000000002")
        self.client = Client()

    def _create_heatmap(self, profile_id: int, tmp_dir: str) -> HeatmapAggregate:
        grid = [[0.0, 1.0], [0.0, 0.0]]
        aggregate = HeatmapAggregate.objects.create(
            profile_id=profile_id,
            map_name="de_mirage",
            metric=HeatmapAggregate.METRIC_KILLS,
            side="ALL",
            period="last_20",
            analytics_version="v2",
            resolution=64,
            grid=grid,
            max_value=1.0,
        )
        with override_settings(MEDIA_ROOT=tmp_dir, MEDIA_URL="/media/"):
            aggregate = ensure_heatmap_image(aggregate, force=True)
        return aggregate

    def test_heatmap_me_requires_login(self):
        response = self.client.get("/api/heatmaps/me?map=de_mirage")
        assert response.status_code == 302

    def test_heatmap_me_uses_logged_in_profile(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            aggregate1 = self._create_heatmap(self.profile1.id, tmp_dir)
            aggregate2 = self._create_heatmap(self.profile2.id, tmp_dir)

            assert self.client.login(username="user1", password="pass") is True
            with override_settings(MEDIA_ROOT=tmp_dir, MEDIA_URL="/media/"):
                response = self.client.get(
                    "/api/heatmaps/me?map=de_mirage&metric=kills&side=ALL&period=last_20&v=v2&res=64"
                )
            assert response.status_code == 200
            payload = response.json()
            assert payload["status"] == "ready"
            assert str(self.profile1.id) in payload["image_url"]
            assert aggregate1.image.url in payload["image_url"]
            assert aggregate2.image.url not in payload["image_url"]

    def test_missing_file_regenerates_heatmap(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            aggregate = self._create_heatmap(self.profile1.id, tmp_dir)
            image_path = os.path.join(tmp_dir, aggregate.image.name)
            os.remove(image_path)
            assert self.client.login(username="user1", password="pass") is True

            with override_settings(MEDIA_ROOT=tmp_dir, MEDIA_URL="/media/"):
                response = self.client.get(
                    "/api/heatmaps/me?map=de_mirage&metric=kills&side=ALL&period=last_20&v=v2&res=64"
                )
            assert response.status_code == 200
            payload = response.json()
            assert payload["status"] == "ready"
            regenerated_path = os.path.join(tmp_dir, payload["image_url"].split("/media/")[-1].split("?")[0])
            assert os.path.exists(regenerated_path)
