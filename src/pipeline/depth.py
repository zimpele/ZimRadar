import asyncio
import json
import logging
import os
import tempfile
import numpy as np
import rasterio
from datetime import datetime, timezone
from PIL import Image
from sqlalchemy import text
from src.storage.db import get_async_session
from src.storage.cache import make_cache_key, get_cached, set_cached
from src.storage.s3 import S3Client

logger = logging.getLogger(__name__)

MODEL_ID = "Intel/zoedepth-nyu"
MODEL_VERSION = "zoedepth-nyu-v1"


class DepthPipeline:
    def __init__(self):
        from transformers import pipeline as hf_pipeline
        self._pipe = hf_pipeline("depth-estimation", model=MODEL_ID, device="cpu")

    def estimate(self, tile_path: str) -> dict:
        with rasterio.open(tile_path) as src:
            data = src.read([1, 2, 3])  # (3, H, W) float32 [0, 1]
            transform = src.transform

        rgb = (data.transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
        image = Image.fromarray(rgb)

        result = self._pipe(image)
        depth_map = result["predicted_depth"].squeeze().numpy()  # (H, W)

        # Low-lying terrain = smallest depth values (overhead sensor: low elevation = closest = smallest depth)
        threshold = np.percentile(depth_map, 10)
        flood_mask = (depth_map <= threshold).astype(np.uint8)

        from rasterio import features
        shapes = list(features.shapes(flood_mask, transform=transform))
        flood_features = [
            {
                "type": "Feature",
                "geometry": geom,
                "properties": {"label": "flood_accumulation_zone"},
            }
            for geom, val in shapes
            if val == 1
        ]

        return {
            "flood_zone_geojson": {"type": "FeatureCollection", "features": flood_features},
            "model_version": MODEL_VERSION,
        }


_pipeline: "DepthPipeline | None" = None


def _get_pipeline() -> "DepthPipeline":
    global _pipeline
    if _pipeline is None:
        _pipeline = DepthPipeline()
    return _pipeline


async def run_depth_for_tile(tile_id: int, processed_s3_path: str) -> None:
    cache_key = make_cache_key(processed_s3_path, MODEL_VERSION)
    result = await asyncio.to_thread(get_cached, cache_key)

    if result is None:
        s3 = S3Client()
        pipeline = await asyncio.to_thread(_get_pipeline)
        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = os.path.join(tmpdir, "tile.tif")
            await asyncio.to_thread(s3.download_tile, processed_s3_path, local_path)
            result = await asyncio.to_thread(pipeline.estimate, local_path)
        await asyncio.to_thread(set_cached, cache_key, result)

    async with get_async_session() as session:
        await session.execute(
            text("""
                INSERT INTO depth_results
                    (tile_id, flood_zone_geojson, model_version, created_at)
                VALUES
                    (:tile_id, :flood_zone_geojson::jsonb, :model_version, :created_at)
                ON CONFLICT (tile_id, model_version) DO NOTHING
            """),
            {
                "tile_id": tile_id,
                "flood_zone_geojson": json.dumps(result["flood_zone_geojson"]),
                "model_version": result["model_version"],
                "created_at": datetime.now(timezone.utc),
            },
        )
