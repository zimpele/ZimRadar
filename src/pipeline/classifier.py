# src/pipeline/classifier.py
import asyncio
import logging
import numpy as np
import xgboost as xgb
from datetime import datetime, timezone
from sqlalchemy import text
from src.config import get_settings
from src.storage.db import get_async_session

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "flood_events_5yr",
    "avg_precipitation_trend",
    "vegetation_loss_pct",
    "urban_density",
    "elevation_variance",
    "infrastructure_age_proxy",
]
RISK_TIERS = ["low", "moderate", "high", "critical"]
MODEL_S3_KEY = "models/xgboost_risk_classifier.json"


def train_classifier(X: np.ndarray, y: np.ndarray) -> xgb.XGBClassifier:
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        eval_metric="mlogloss",
        random_state=42,
    )
    model.fit(X, y)
    return model


def save_classifier_to_s3(model: xgb.XGBClassifier) -> None:
    import tempfile
    import os
    from src.storage.s3 import S3Client

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        temp_path = f.name
    try:
        model.save_model(temp_path)
        S3Client().upload_model(temp_path, MODEL_S3_KEY)
    finally:
        os.unlink(temp_path)


def load_classifier_from_s3() -> xgb.XGBClassifier:
    import tempfile
    import os
    from src.storage.s3 import S3Client

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        temp_path = f.name
    try:
        S3Client().download_model(MODEL_S3_KEY, temp_path)
        model = xgb.XGBClassifier()
        model.load_model(temp_path)
        return model
    finally:
        os.unlink(temp_path)


def classify_region_features(features: dict[str, float]) -> tuple[str, float]:
    model = load_classifier_from_s3()
    X = np.array([[features[k] for k in FEATURE_NAMES]])
    proba = model.predict_proba(X)[0]
    tier_idx = int(np.argmax(proba))
    return RISK_TIERS[tier_idx], float(proba[tier_idx])


def _composite_score(confidence: float, flood_flag: bool, fire_flag: bool) -> float:
    w1, w2, w3 = get_settings().risk_weights
    return w1 * confidence + w2 * float(flood_flag) + w3 * float(fire_flag)


async def build_features_for_region(region_id: int) -> dict[str, float]:
    async with get_async_session() as session:
        flood_result = await session.execute(
            text("""
                SELECT COUNT(*) FROM fema_declarations
                WHERE disaster_type ILIKE '%flood%'
                AND declaration_date >= NOW() - INTERVAL '5 years'
            """),
        )
        flood_events = int(flood_result.scalar() or 0)

        precip_result = await session.execute(
            text("""
                SELECT precipitation_mm FROM noaa_observations
                WHERE region_id = :rid AND date >= NOW() - INTERVAL '2 years'
                ORDER BY date
            """),
            {"rid": region_id},
        )
        precip_rows = precip_result.fetchall()

    precip_values = [float(r.precipitation_mm or 0.0) for r in precip_rows]
    precip_trend = (
        float(np.polyfit(range(len(precip_values)), precip_values, 1)[0])
        if len(precip_values) >= 2
        else 0.0
    )

    async with get_async_session() as session:
        seg_result = await session.execute(
            text("""
                SELECT sr.area_stats
                FROM segmentation_results sr
                JOIN sentinel2_tiles t ON t.id = sr.tile_id
                WHERE t.region_id = :rid
                ORDER BY t.date DESC
                LIMIT 5
            """),
            {"rid": region_id},
        )
        seg_rows = seg_result.fetchall()

    veg_fracs = [float((r.area_stats or {}).get("vegetation", 0.3)) for r in seg_rows]
    urban_fracs = [float((r.area_stats or {}).get("urban", 0.1)) for r in seg_rows]

    # seg_rows ordered DESC, so index 0 is newest, -1 is oldest
    vegetation_loss_pct = max(0.0, veg_fracs[0] - veg_fracs[-1]) if len(veg_fracs) >= 2 else 0.0
    urban_density = float(np.mean(urban_fracs)) if urban_fracs else 0.1

    async with get_async_session() as session:
        depth_result = await session.execute(
            text("""
                SELECT COUNT(*) FROM depth_results dr
                JOIN sentinel2_tiles t ON t.id = dr.tile_id
                WHERE t.region_id = :rid
            """),
            {"rid": region_id},
        )
        depth_count = int(depth_result.scalar() or 0)

    elevation_variance = min(100.0, float(depth_count) * 10.0)
    infrastructure_age_proxy = 0.5  # static until OSM feature extraction is added in Phase 3

    return {
        "flood_events_5yr": float(flood_events),
        "avg_precipitation_trend": precip_trend,
        "vegetation_loss_pct": vegetation_loss_pct,
        "urban_density": urban_density,
        "elevation_variance": elevation_variance,
        "infrastructure_age_proxy": infrastructure_age_proxy,
    }


async def run_classification_for_region(region_id: int) -> None:
    features = await build_features_for_region(region_id)
    tier, confidence = await asyncio.to_thread(classify_region_features, features)

    async with get_async_session() as session:
        forecast_result = await session.execute(
            text("""
                SELECT flood_risk_flag, fire_risk_flag FROM forecasts
                WHERE region_id = :rid ORDER BY created_at DESC LIMIT 1
            """),
            {"rid": region_id},
        )
        forecast_row = forecast_result.fetchone()

    flood_flag = bool(forecast_row.flood_risk_flag) if forecast_row else False
    fire_flag = bool(forecast_row.fire_risk_flag) if forecast_row else False
    composite = _composite_score(confidence, flood_flag, fire_flag)

    async with get_async_session() as session:
        await session.execute(
            text("""
                INSERT INTO risk_assessments
                    (region_id, risk_tier, confidence, composite_score, assessed_at)
                VALUES
                    (:region_id, :risk_tier, :confidence, :composite_score, :assessed_at)
            """),
            {
                "region_id": region_id,
                "risk_tier": tier,
                "confidence": confidence,
                "composite_score": composite,
                "assessed_at": datetime.now(timezone.utc),
            },
        )
