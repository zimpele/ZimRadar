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
