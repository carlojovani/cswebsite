from __future__ import annotations

from pathlib import Path
from uuid import uuid4
from django.contrib.auth import get_user_model
from django.test import override_settings

from faceit_analytics.services.heatmaps import DEFAULT_MAPS
from faceit_analytics.services.paths import ensure_profile_dirs
from users.models import PlayerProfile


def test_profile_dirs_created_and_idempotent(tmp_path):
    media_root = tmp_path / "media"
    with override_settings(MEDIA_ROOT=media_root):
        unique_suffix = uuid4().hex
        unique_email = f"demo-user-{unique_suffix}@example.com"
        user = get_user_model().objects.create_user(
            username=f"demo-user-{unique_suffix}",
            email=unique_email,
            password="pass",
            user_type="player",
        )
        profile = PlayerProfile.objects.create(user=user, steam_id="76561198000000010")

        ensure_profile_dirs(profile)
        ensure_profile_dirs(profile)

        map_name = list(DEFAULT_MAPS)[0]
        expected_dirs = [
            media_root / "local_demos" / profile.steam_id / map_name,
            media_root / "heatmaps_cache" / profile.steam_id / map_name,
            media_root / "heatmaps_local" / profile.steam_id / "aggregate" / map_name,
            media_root / "heatmaps" / str(profile.id) / map_name / "all" / "last_20",
        ]
        for path in expected_dirs:
            assert Path(path).exists()
