import json
import numpy as np
import os
import pytest
import rasterio
import tempfile
import torch
from rasterio.transform import from_bounds
from unittest.mock import AsyncMock, MagicMock, patch


def _make_test_tile(path: str) -> None:
    data = np.random.default_rng(0).random((3, 64, 64)).astype(np.float32)
    transform = from_bounds(0, 0, 1, 1, 64, 64)
    with rasterio.open(
        path, "w", driver="GTiff", height=64, width=64, count=3,
        dtype="float32", crs="EPSG:4326", transform=transform,
    ) as dst:
        dst.write(data)


def test_depth_pipeline_estimate_returns_flood_zone_geojson():
    from src.pipeline.depth import DepthPipeline, MODEL_VERSION

    fake_depth = torch.tensor(np.random.default_rng(1).random((64, 64)).astype(np.float32))
    mock_pipe = MagicMock(return_value={"predicted_depth": fake_depth})

    with tempfile.TemporaryDirectory() as tmpdir:
        tile_path = os.path.join(tmpdir, "tile.tif")
        _make_test_tile(tile_path)

        dp = DepthPipeline.__new__(DepthPipeline)
        dp._pipe = mock_pipe
        result = dp.estimate(tile_path)

    assert result["model_version"] == MODEL_VERSION
    assert result["flood_zone_geojson"]["type"] == "FeatureCollection"
    assert isinstance(result["flood_zone_geojson"]["features"], list)


def test_depth_pipeline_flood_mask_covers_top_10_percent():
    from src.pipeline.depth import DepthPipeline

    # Depth map: values 1..64*64, top 10% should be the highest values
    depth_vals = np.arange(1, 64 * 64 + 1, dtype=np.float32).reshape(64, 64)
    fake_depth = torch.from_numpy(depth_vals)
    mock_pipe = MagicMock(return_value={"predicted_depth": fake_depth})

    with tempfile.TemporaryDirectory() as tmpdir:
        tile_path = os.path.join(tmpdir, "tile.tif")
        _make_test_tile(tile_path)

        dp = DepthPipeline.__new__(DepthPipeline)
        dp._pipe = mock_pipe
        result = dp.estimate(tile_path)

    # Flood zone features are present for the highest-depth pixels
    assert len(result["flood_zone_geojson"]["features"]) > 0
    # The flood mask must cover exactly the bottom 10% = 410 pixels of a 64x64 map
    # with monotonically increasing values 1..4096 the threshold is deterministic
    threshold = np.percentile(depth_vals, 10)
    expected_flood_pixels = int(np.sum(depth_vals <= threshold))
    assert expected_flood_pixels == pytest.approx(64 * 64 * 0.10, abs=5)


@pytest.mark.asyncio
async def test_run_depth_for_tile_skips_download_when_cached():
    from src.pipeline.depth import run_depth_for_tile, MODEL_VERSION

    cached_result = {
        "flood_zone_geojson": {"type": "FeatureCollection", "features": []},
        "model_version": MODEL_VERSION,
    }

    mock_session = MagicMock()
    mock_session.execute = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    with (
        patch("src.pipeline.depth.get_cached", return_value=cached_result),
        patch("src.pipeline.depth.S3Client") as mock_s3,
        patch("src.pipeline.depth.DepthPipeline") as mock_dp,
        patch("src.pipeline.depth.get_async_session", return_value=mock_session),
    ):
        await run_depth_for_tile(tile_id=1, processed_s3_path="s3/path.tif")

    mock_s3.assert_not_called()
    mock_dp.assert_not_called()
    mock_session.execute.assert_called_once()


@pytest.mark.asyncio
async def test_run_depth_for_tile_caches_result_on_miss():
    from src.pipeline.depth import run_depth_for_tile, MODEL_VERSION

    fresh_result = {
        "flood_zone_geojson": {"type": "FeatureCollection", "features": []},
        "model_version": MODEL_VERSION,
    }

    mock_session = MagicMock()
    mock_session.execute = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    set_cached_calls = []

    # get_cached / set_cached are called via asyncio.to_thread; patching the underlying
    # function is sufficient because to_thread simply invokes the mock in a thread pool.
    with (
        patch("src.pipeline.depth.get_cached", return_value=None),
        patch("src.pipeline.depth.set_cached", side_effect=lambda k, v: set_cached_calls.append(v)),
        patch("src.pipeline.depth.S3Client"),
        patch("src.pipeline.depth._get_pipeline") as mock_get_pipeline,
        patch("src.pipeline.depth.get_async_session", return_value=mock_session),
    ):
        mock_get_pipeline.return_value.estimate.return_value = fresh_result
        await run_depth_for_tile(tile_id=2, processed_s3_path="s3/other.tif")

    assert len(set_cached_calls) == 1
    assert set_cached_calls[0]["model_version"] == MODEL_VERSION
