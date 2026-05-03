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
    # FEMA NRI features (0.0 when county not in NRI table)
    "nri_risk_score",
    "nri_eal_score",
    "nri_sovi_score",
    "nri_flood_risks",  # max(cfld_risks, rfld_risks)
    "nri_fire_risks",
    "nri_heat_risks",
]
RISK_TIERS = ["low", "moderate", "high", "critical"]
TIER_WEIGHTS = {"low": 0.15, "moderate": 0.45, "high": 0.75, "critical": 1.0}
MODEL_S3_KEY = "models/xgboost_risk_classifier.json"


def train_classifier(X: np.ndarray, y: np.ndarray) -> xgb.XGBClassifier:
    from sklearn.utils.class_weight import compute_sample_weight

    sample_weights = compute_sample_weight(class_weight="balanced", y=y)
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        eval_metric="mlogloss",
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    model.fit(X, y, sample_weight=sample_weights)
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
    X = np.column_stack(
        [
            rng.poisson(2, n),  # flood_events_5yr
            rng.normal(0, 1, n),  # avg_precipitation_trend
            rng.uniform(0, 0.5, n),  # vegetation_loss_pct
            rng.uniform(0, 1, n),  # urban_density
            rng.uniform(0, 100, n),  # elevation_variance
            rng.uniform(0, 1, n),  # infrastructure_age_proxy
            rng.uniform(0, 100, n),  # nri_risk_score
            rng.uniform(0, 1e9, n),  # nri_eal_score
            rng.uniform(0, 1, n),  # nri_sovi_score
            rng.uniform(0, 100, n),  # nri_flood_risks
            rng.uniform(0, 100, n),  # nri_fire_risks
            rng.uniform(0, 100, n),  # nri_heat_risks
        ]
    )
    risk_score = X[:, 0] * 0.3 + np.clip(X[:, 1], 0, None) * 0.2 + X[:, 2] * 0.3 + X[:, 3] * 0.2
    y = np.digitize(risk_score, bins=[0.5, 1.0, 1.8]).clip(0, 3)
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        eval_metric="mlogloss",
        random_state=42,
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


async def build_features_for_region(region_id: int) -> dict[str, float]:
    async with get_async_session() as session:
        state_result = await session.execute(
            text("SELECT state_code FROM regions WHERE id = :rid"), {"rid": region_id}
        )
        state_row = state_result.fetchone()
        state_code = state_row[0] if state_row else None

        if state_code and not state_code.startswith("DE-"):
            flood_result = await session.execute(
                text("""
                    SELECT COUNT(*) FROM fema_declarations
                    WHERE disaster_type IN ('Flood','Hurricane','Severe Storm','Tropical Storm')
                    AND state = :state
                    AND declaration_date >= NOW() - INTERVAL '5 years'
                """),
                {"state": state_code},
            )
        else:
            flood_result = await session.execute(
                text("SELECT 0"),
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
    elevation_variance = min(
        100.0, float(np.mean(flood_zone_feature_counts)) if flood_zone_feature_counts else 0.0
    )
    infrastructure_age_proxy = 0.5  # static until OSM feature extraction is added in Phase 3

    async with get_async_session() as session:
        fips_result = await session.execute(
            text("SELECT county_fips FROM regions WHERE id = :rid"), {"rid": region_id}
        )
        fips_row = fips_result.fetchone()
        county_fips = fips_row[0] if fips_row else None

        nri_row = None
        if county_fips:
            nri_result = await session.execute(
                text("""
                    SELECT risk_score, eal_score, sovi_score,
                           cfld_risks, rfld_risks, wfir_risks, hwav_risks
                    FROM fema_nri_county WHERE county_fips = :fips
                """),
                {"fips": county_fips},
            )
            nri_row = nri_result.fetchone()

    nri_risk_score = float(nri_row.risk_score or 0.0) if nri_row else 0.0
    nri_eal_score = float(nri_row.eal_score or 0.0) if nri_row else 0.0
    nri_sovi_score = float(nri_row.sovi_score or 0.0) if nri_row else 0.0
    nri_flood_risks = (
        max(float(nri_row.cfld_risks or 0.0), float(nri_row.rfld_risks or 0.0)) if nri_row else 0.0
    )
    nri_fire_risks = float(nri_row.wfir_risks or 0.0) if nri_row else 0.0
    nri_heat_risks = float(nri_row.hwav_risks or 0.0) if nri_row else 0.0

    return {
        "flood_events_5yr": float(flood_events),
        "avg_precipitation_trend": precip_trend,
        "vegetation_loss_pct": vegetation_loss_pct,
        "urban_density": urban_density,
        "elevation_variance": elevation_variance,
        "infrastructure_age_proxy": infrastructure_age_proxy,
        "nri_risk_score": nri_risk_score,
        "nri_eal_score": nri_eal_score,
        "nri_sovi_score": nri_sovi_score,
        "nri_flood_risks": nri_flood_risks,
        "nri_fire_risks": nri_fire_risks,
        "nri_heat_risks": nri_heat_risks,
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
