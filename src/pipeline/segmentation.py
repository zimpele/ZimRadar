import asyncio
import json
import logging
import numpy as np
import torch
import rasterio
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from datetime import datetime, timezone
from sqlalchemy import text
from src.storage.db import get_async_session

logger = logging.getLogger(__name__)

MODEL_ID = "nvidia/segformer-b2-finetuned-ade-512-512"
MODEL_VERSION = "segformer-b2-ade-v1"

LAND_USE_CLASSES = ["water", "vegetation", "urban", "bare_soil", "burn_scar"]

# Mapping from ADE20K class indices to our 5 land-use categories.
ADE20K_TO_LANDUSE = {
    6: "water",
    26: "water",
    60: "water",
    9: "vegetation",
    12: "vegetation",
    4: "vegetation",
    17: "urban",
    11: "urban",
    29: "bare_soil",
    94: "burn_scar",
}


class SegmentationPipeline:
    def __init__(self):
        self.processor = SegformerImageProcessor.from_pretrained(MODEL_ID)
        self.model = SegformerForSemanticSegmentation.from_pretrained(MODEL_ID)
        self.model.eval()

    def segment(self, tile_path: str) -> dict:
        with rasterio.open(tile_path) as src:
            data = src.read([1, 2, 3])  # RGB bands, float32 [0,1]
            transform = src.transform

        # Convert to uint8 PIL image for SegFormer processor
        rgb = (data.transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
        image = Image.fromarray(rgb)

        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)

        seg_map = self.processor.post_process_semantic_segmentation(
            outputs, target_sizes=[image.size[::-1]]
        )[0].numpy()

        # Remap ADE20K labels to our 5 classes; default to bare_soil (index 3)
        landuse_map = np.full(seg_map.shape, 3, dtype=np.uint8)
        for ade_idx, landuse in ADE20K_TO_LANDUSE.items():
            landuse_map[seg_map == ade_idx] = LAND_USE_CLASSES.index(landuse)

        total_pixels = landuse_map.size
        area_stats = {
            cls: float((landuse_map == i).sum() / total_pixels)
            for i, cls in enumerate(LAND_USE_CLASSES)
        }

        flood_mask = (landuse_map == LAND_USE_CLASSES.index("water")).astype(np.uint8)

        def mask_to_geojson(mask: np.ndarray, label: str) -> dict:
            from rasterio import features

            shapes = list(features.shapes(mask, transform=transform))
            features_list = [
                {"type": "Feature", "geometry": geom, "properties": {"label": label}}
                for geom, val in shapes
                if val == 1
            ]
            return {"type": "FeatureCollection", "features": features_list}

        geojson: dict = {"type": "FeatureCollection", "features": []}
        for i, cls in enumerate(LAND_USE_CLASSES):
            mask = (landuse_map == i).astype(np.uint8)
            cls_geojson = mask_to_geojson(mask, cls)
            geojson["features"].extend(cls_geojson["features"])

        flood_zone_geojson = mask_to_geojson(flood_mask, "flood_zone")

        return {
            "geojson": geojson,
            "area_stats": area_stats,
            "flood_zone_geojson": flood_zone_geojson,
            "model_version": MODEL_VERSION,
        }


async def run_segmentation_for_tile(tile_id: int, processed_s3_path: str) -> None:
    import tempfile
    import os
    from src.storage.s3 import S3Client

    s3 = S3Client()
    pipeline = SegmentationPipeline()

    with tempfile.TemporaryDirectory() as tmpdir:
        local_path = os.path.join(tmpdir, "tile.tif")
        await asyncio.to_thread(s3.download_tile, processed_s3_path, local_path)
        result = await asyncio.to_thread(pipeline.segment, local_path)

    async with get_async_session() as session:
        await session.execute(
            text("""
                INSERT INTO segmentation_results
                    (tile_id, geojson, area_stats, flood_zone_geojson, model_version, created_at)
                VALUES
                    (:tile_id, :geojson::jsonb, :area_stats::jsonb, :flood_zone_geojson::jsonb,
                     :model_version, :created_at)
                ON CONFLICT DO NOTHING
            """),
            {
                "tile_id": tile_id,
                "geojson": json.dumps(result["geojson"]),
                "area_stats": json.dumps(result["area_stats"]),
                "flood_zone_geojson": json.dumps(result["flood_zone_geojson"]),
                "model_version": result["model_version"],
                "created_at": datetime.now(timezone.utc),
            },
        )
