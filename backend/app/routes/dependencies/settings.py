from functools import lru_cache

from app.core.config import Settings, settings


@lru_cache
def get_app_settings() -> Settings:
    return settings
