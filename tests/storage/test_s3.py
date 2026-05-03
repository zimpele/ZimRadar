import os
import pytest
import src.storage.s3 as s3_mod
from src.storage.s3 import S3Client


@pytest.fixture
def s3(tmp_path, monkeypatch):
    monkeypatch.setattr(s3_mod, "DATA_ROOT", tmp_path)
    return S3Client()


def test_upload_tile_returns_s3_path(s3, tmp_path):
    tile_path = tmp_path / "tile.tif"
    tile_path.write_bytes(b"fake tif data")

    result = s3.upload_tile(str(tile_path), region_id=1, date="2024-01-15")

    assert result == "sentinel2/1/2024-01-15/tile.tif"
    assert (s3._root / result).exists()


def test_download_tile_writes_file(s3, tmp_path):
    src_file = tmp_path / "tile.tif"
    src_file.write_bytes(b"tile data")
    key = s3.upload_tile(str(src_file), region_id=1, date="2024-01-15")

    dest = str(tmp_path / "downloaded.tif")
    s3.download_tile(key, dest)

    assert os.path.exists(dest)
    assert open(dest, "rb").read() == b"tile data"


def test_upload_model_uses_given_key(s3, tmp_path):
    model_file = tmp_path / "model.json"
    model_file.write_text("{}")

    key = s3.upload_model(str(model_file), "models/xgboost.json")

    assert key == "models/xgboost.json"
    assert (s3._root / "models/xgboost.json").exists()


def test_download_model_creates_parent_dirs(s3, tmp_path):
    model_file = tmp_path / "model.json"
    model_file.write_text("{}")
    s3.upload_model(str(model_file), "models/xgboost.json")

    dest = str(tmp_path / "subdir" / "model.json")
    s3.download_model("models/xgboost.json", dest)

    assert os.path.exists(dest)
