from django.contrib.auth import get_user_model
from django.test import override_settings

from faceit_analytics.services.demo_events import discover_demo_files
from users.models import PlayerProfile
from uuid import uuid4


def test_discover_demo_files_uses_media_local_demos_only(tmp_path):
    media_root = tmp_path / "media"
    other_root = tmp_path / "other_demos"
    other_root.mkdir(parents=True, exist_ok=True)
    other_demo = other_root / "outside.dem"
    other_demo.write_bytes(b"outside")

    with override_settings(MEDIA_ROOT=media_root):
        unique_suffix = uuid4().hex
        user = get_user_model().objects.create_user(
            username=f"demo-user-{unique_suffix}",
            email=f"demo-user-{unique_suffix}@example.com",
            password="pass",
            user_type="player",
        )
        profile = PlayerProfile.objects.create(user=user, steam_id="76561198000000088")
        demo_dir = media_root / "local_demos" / profile.steam_id / "de_mirage"
        demo_dir.mkdir(parents=True, exist_ok=True)
        local_demo = demo_dir / "local.dem"
        local_demo.write_bytes(b"local")

        demos = discover_demo_files(profile, "last_20", "de_mirage", demos_dir=other_root)
        assert local_demo in demos
        assert other_demo not in demos
