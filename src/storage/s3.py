import os
import boto3
from pathlib import Path
from src.config import get_settings


class S3Client:
    def __init__(self, bucket: str | None = None):
        settings = get_settings()
        self.bucket = bucket or settings.s3_bucket_tiles
        self._client = boto3.client(
            "s3",
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
            region_name=settings.aws_default_region,
        )

    def upload_tile(self, local_path: str, region_id: int, date: str) -> str:
        filename = Path(local_path).name
        key = f"sentinel2/{region_id}/{date}/{filename}"
        self._client.upload_file(local_path, self.bucket, key)
        return key

    def upload_processed_tile(self, local_path: str, region_id: int, date: str) -> str:
        filename = Path(local_path).name
        key = f"sentinel2_processed/{region_id}/{date}/{filename}"
        self._client.upload_file(local_path, self.bucket, key)
        return key

    def download_tile(self, s3_key: str, dest_path: str) -> None:
        if dirname := os.path.dirname(dest_path):
            os.makedirs(dirname, exist_ok=True)
        self._client.download_file(self.bucket, s3_key, dest_path)

    def upload_pdf(self, local_path: str, report_id: str) -> str:
        settings = get_settings()
        key = f"{report_id}.pdf"
        self._client.upload_file(local_path, settings.s3_bucket_pdfs, key)
        return key
