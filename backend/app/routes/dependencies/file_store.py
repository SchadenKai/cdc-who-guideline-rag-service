from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from app.core.config import Settings
from app.routes.dependencies.settings import get_app_settings
from app.services.file_store.db import S3Service


@lru_cache
def get_s3_service(
    settings: Annotated[Settings, Depends(get_app_settings)],
) -> S3Service:
    return S3Service(settings=settings)
