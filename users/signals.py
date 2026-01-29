from __future__ import annotations

import logging

from django.db.models.signals import post_save
from django.dispatch import receiver

from faceit_analytics.services.paths import ensure_profile_dirs
from users.models import PlayerProfile

logger = logging.getLogger(__name__)


@receiver(post_save, sender=PlayerProfile)
def ensure_profile_directories(sender, instance: PlayerProfile, **kwargs) -> None:
    try:
        ensure_profile_dirs(instance)
    except Exception:  # pragma: no cover - filesystem edge cases
        logger.exception("Failed to ensure profile directories for profile %s", instance.id)
