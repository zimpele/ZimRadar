import os
import shutil
from pathlib import Path


DATA_ROOT = Path(os.getenv("DATA_ROOT", "/app/data"))


class S3Client:
    """Local-disk file store with the same interface as the former S3 client.

    Files are written under DATA_ROOT (default /app/data), which should be
    mounted as a Docker volume for persistence.
    """

    def __init__(self, bucket: str | None = None):
        self._root = DATA_ROOT
        self._root.mkdir(parents=True, exist_ok=True)

    def upload_tile(self, local_path: str, region_id: int, date: str) -> str:
        filename = Path(local_path).name
        key = f"sentinel2/{region_id}/{date}/{filename}"
        dest = self._root / key
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, dest)
        return key

    def upload_processed_tile(self, local_path: str, region_id: int, date: str) -> str:
        filename = Path(local_path).name
        key = f"sentinel2_processed/{region_id}/{date}/{filename}"
        dest = self._root / key
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, dest)
        return key

    def download_tile(self, key: str, dest_path: str) -> None:
        if dirname := os.path.dirname(dest_path):
            os.makedirs(dirname, exist_ok=True)
        shutil.copy2(self._root / key, dest_path)

    def upload_pdf(self, local_path: str, report_id: str) -> str:
        key = f"pdfs/{report_id}.pdf"
        dest = self._root / key
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, dest)
        return key

    def upload_model(self, local_path: str, s3_key: str) -> str:
        dest = self._root / s3_key
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, dest)
        return s3_key

    def download_model(self, s3_key: str, dest_path: str) -> None:
        if dirname := os.path.dirname(dest_path):
            os.makedirs(dirname, exist_ok=True)
        shutil.copy2(self._root / s3_key, dest_path)
