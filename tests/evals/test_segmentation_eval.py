"""
Segmentation eval against a subset of EuroSAT.
EuroSAT is a scene-level classification dataset; we use it to measure
per-class accuracy of our patch-level land-use predictions.
Eval passes if mean accuracy across 5 land-use classes >= 0.60 (baseline).
Raise threshold in Phase 2 after fine-tuning.
"""

import os
import tempfile
import numpy as np
import pytest
import rasterio
from datasets import load_dataset
from rasterio.transform import from_bounds
from src.pipeline.segmentation import SegmentationPipeline

# EuroSAT class → our land-use class mapping
EUROSAT_TO_LANDUSE = {
    "AnnualCrop": "vegetation",
    "Forest": "vegetation",
    "HerbaceousVegetation": "vegetation",
    "Highway": "urban",
    "Industrial": "urban",
    "Pasture": "vegetation",
    "PermanentCrop": "vegetation",
    "Residential": "urban",
    "River": "water",
    "SeaLake": "water",
}

EVAL_SAMPLES = 50


@pytest.mark.slow
def test_segmentation_eurosat_baseline():
    dataset = load_dataset("torchgeo/eurosat", split="test", trust_remote_code=True)
    pipeline = SegmentationPipeline()

    correct = 0
    total = 0

    for sample in dataset.select(range(EVAL_SAMPLES)):
        label_name = dataset.features["label"].int2str(sample["label"])
        expected_class = EUROSAT_TO_LANDUSE.get(label_name)
        if expected_class is None:
            continue

        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            tmp_path = f.name

        try:
            img = sample["image"].convert("RGB")
            img_resized = img.resize((256, 256))
            arr = np.array(img_resized).transpose(2, 0, 1).astype(np.float32) / 255.0

            with rasterio.open(
                tmp_path,
                "w",
                driver="GTiff",
                height=256,
                width=256,
                count=3,
                dtype="float32",
                crs="EPSG:4326",
                transform=from_bounds(0, 0, 1, 1, 256, 256),
            ) as dst:
                dst.write(arr)

            result = pipeline.segment(tmp_path)
            predicted_class = max(result["area_stats"], key=lambda k: result["area_stats"][k])

            if predicted_class == expected_class:
                correct += 1
            total += 1
        finally:
            os.unlink(tmp_path)

    accuracy = correct / total if total > 0 else 0.0
    print(f"\nSegmentation eval: {correct}/{total} = {accuracy:.2%}")
    assert accuracy >= 0.60, f"Baseline accuracy {accuracy:.2%} below 0.60 threshold"
