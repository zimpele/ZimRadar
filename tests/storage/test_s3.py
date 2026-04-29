from unittest.mock import patch, MagicMock
from src.storage.s3 import S3Client


def test_upload_tile_returns_s3_path(tmp_path):
    tile_path = tmp_path / "tile.tif"
    tile_path.write_bytes(b"fake tif data")

    with patch("src.storage.s3.boto3.client") as mock_boto:
        mock_client = MagicMock()
        mock_boto.return_value = mock_client
        s3 = S3Client(bucket="zimradar-tiles")

        result = s3.upload_tile(str(tile_path), region_id=1, date="2024-01-15")

    assert result == "sentinel2/1/2024-01-15/tile.tif"
    mock_client.upload_file.assert_called_once()


def test_download_tile_writes_file(tmp_path):
    with patch("src.storage.s3.boto3.client") as mock_boto:
        mock_client = MagicMock()
        mock_boto.return_value = mock_client
        s3 = S3Client(bucket="zimradar-tiles")

        dest = str(tmp_path / "downloaded.tif")
        s3.download_tile("sentinel2/1/2024-01-15/tile.tif", dest)

    mock_client.download_file.assert_called_once_with(
        "zimradar-tiles", "sentinel2/1/2024-01-15/tile.tif", dest
    )


def test_upload_model_uses_tiles_bucket(tmp_path):
    with patch("src.storage.s3.boto3.client") as mock_boto:
        with patch("src.storage.s3.get_settings") as mock_settings:
            mock_client = MagicMock()
            mock_boto.return_value = mock_client
            mock_settings_obj = MagicMock()
            mock_settings_obj.s3_bucket_tiles = "zimradar-tiles"
            mock_settings.return_value = mock_settings_obj

            s3 = S3Client()
            tmp_file = tmp_path / "model.json"
            tmp_file.write_text("{}")

            key = s3.upload_model(str(tmp_file), "models/xgboost.json")

            assert key == "models/xgboost.json"
            mock_client.upload_file.assert_called_with(
                str(tmp_file), "zimradar-tiles", "models/xgboost.json"
            )


def test_download_model_creates_parent_dirs(tmp_path):
    with patch("src.storage.s3.boto3.client") as mock_boto:
        with patch("src.storage.s3.get_settings") as mock_settings:
            mock_client = MagicMock()
            mock_boto.return_value = mock_client
            mock_settings_obj = MagicMock()
            mock_settings_obj.s3_bucket_tiles = "zimradar-tiles"
            mock_settings.return_value = mock_settings_obj

            s3 = S3Client()
            dest = str(tmp_path / "subdir" / "model.json")
            s3.download_model("models/xgboost.json", dest)

            mock_client.download_file.assert_called_with(
                "zimradar-tiles", "models/xgboost.json", dest
            )
