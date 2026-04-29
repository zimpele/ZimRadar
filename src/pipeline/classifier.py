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
TIER_WEIGHTS = {"low": 0.15, "moderate": 0.45, "high": 0.75, "critical": 1.0}
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


def _bootstrap_model() -> xgb.XGBClassifier:
    """Train a synthetic model so the pipeline can run before real data exists."""
    logger.warning("No trained model found — bootstrapping XGBoost on synthetic data.")
    rng = np.random.default_rng(42)
    n = 500
    X = np.column_stack([
        rng.poisson(2, n),          # flood_events_5yr
        rng.normal(0, 1, n),        # avg_precipitation_trend
        rng.uniform(0, 0.5, n),     # vegetation_loss_pct
        rng.uniform(0, 1, n),       # urban_density
        rng.uniform(0, 100, n),     # elevation_variance
        rng.uniform(0, 1, n),       # infrastructure_age_proxy
    ])
    risk_score = X[:, 0] * 0.3 + np.clip(X[:, 1], 0, None) * 0.2 + X[:, 2] * 0.3 + X[:, 3] * 0.2
    y = np.digitize(risk_score, bins=[0.5, 1.0, 1.8]).clip(0, 3)
    model = xgb.XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        eval_metric="mlogloss", random_state=42,
    )
    model.fit(X, y)
    save_classifier_to_s3(model)
    logger.info("Bootstrap model saved to %s", MODEL_S3_KEY)
    return model


def load_classifier_from_s3() -> xgb.XGBClassifier:
    import tempfile
    import os
    from src.storage.s3 import S3Client

    client = S3Client()
    from src.storage.s3 import DATA_ROOT
    if not (DATA_ROOT / MODEL_S3_KEY).exists():
        return _bootstrap_model()

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        temp_path = f.name
    try:
        client.download_model(MODEL_S3_KEY, temp_path)
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


def _composite_score(tier: str, confidence: float, flood_flag: bool, fire_flag: bool) -> float:
    w1, w2, w3 = get_settings().risk_weights
    tier_weight = TIER_WEIGHTS.get(tier, 0.5)
    return w1 * confidence * tier_weight + w2 * float(flood_flag) + w3 * float(fire_flag)


def _bbox_to_state_code(bbox: dict) -> str | None:
    """Rough centroid-based US state lookup for FEMA filtering."""
    cx = (bbox["min_lon"] + bbox["max_lon"]) / 2
    cy = (bbox["min_lat"] + bbox["max_lat"]) / 2
    # Bounding boxes for common states (approximate)
    STATE_BOXES = {
        "LA": (-94.0, 28.9, -88.8, 33.0),
        "TX": (-106.6, 25.8, -93.5, 36.5),
        "CA": (-124.4, 32.5, -114.1, 42.0),
        "FL": (-87.6, 24.5, -80.0, 31.0),
        "NY": (-79.8, 40.5, -71.8, 45.0),
        "MS": (-91.7, 30.2, -88.1, 35.0),
        "AL": (-88.5, 30.2, -84.9, 35.0),
        "GA": (-85.6, 30.4, -80.8, 35.0),
        "SC": (-83.4, 32.0, -78.5, 35.2),
        "NC": (-84.3, 33.8, -75.4, 36.6),
        "VA": (-83.7, 36.5, -75.2, 39.5),
    }
    for state, (min_lon, min_lat, max_lon, max_lat) in STATE_BOXES.items():
        if min_lon <= cx <= max_lon and min_lat <= cy <= max_lat:
            return state
    return None


async def build_features_for_region(region_id: int) -> dict[str, float]:
    async with get_async_session() as session:
        # Derive state code from region bbox centroid longitude/latitude
        bbox_result = await session.execute(
            text("SELECT bbox FROM regions WHERE id = :rid"), {"rid": region_id}
        )
        bbox_row = bbox_result.fetchone()
        state_code = None
        if bbox_row:
            import json as _j
            bbox = bbox_row[0] if not isinstance(bbox_row[0], str) else _j.loads(bbox_row[0])
            state_code = _bbox_to_state_code(bbox)

        if state_code:
            flood_result = await session.execute(
                text("""
                    SELECT COUNT(*) FROM fema_declarations
                    WHERE disaster_type ILIKE '%flood%'
                    AND state = :state
                    AND declaration_date >= NOW() - INTERVAL '5 years'
                """),
                {"state": state_code},
            )
        else:
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
                SELECT dr.flood_zone_geojson
                FROM depth_results dr
                JOIN sentinel2_tiles t ON t.id = dr.tile_id
                WHERE t.region_id = :rid
            """),
            {"rid": region_id},
        )
        depth_rows = depth_result.fetchall()

    import json as _json
    flood_zone_feature_counts = []
    for row in depth_rows:
        fz = row.flood_zone_geojson
        if isinstance(fz, str):
            fz = _json.loads(fz)
        if fz:
            flood_zone_feature_counts.append(len(fz.get("features", [])))
    # High feature count = fragmented low-lying terrain = higher elevation variance proxy
    elevation_variance = min(100.0, float(np.mean(flood_zone_feature_counts)) if flood_zone_feature_counts else 0.0)
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
    composite = _composite_score(tier, confidence, flood_flag, fire_flag)

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
