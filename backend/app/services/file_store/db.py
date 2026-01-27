import boto3

from app.core.config import Settings


class S3Service:
    def __init__(self, settings: Settings):
        self._s3_client: any | None = None
        self.settings: Settings = settings

    @property
    def client(self):
        if self._s3_client:
            return self._s3_client
        s3_client = boto3.client(
            "s3",
            endpoint_url=self.settings.minio_endpoint_url,
            aws_access_key_id=self.settings.minio_username,
            aws_secret_access_key=self.settings.minio_password,
        )
        self._s3_client = s3_client
        return self._s3_client
