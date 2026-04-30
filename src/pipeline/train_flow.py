"""Prefect flow: train XGBoost risk classifier from real FEMA data."""
import logging
import numpy as np
from prefect import flow, task, get_run_logger

logger = logging.getLogger(__name__)

FLOOD_TYPES = ("Flood", "Hurricane", "Severe Storm", "Tropical Storm",
               "Coastal Storm", "Dam/Levee Break", "Typhoon")
FIRE_TYPES = ("Fire", "Wildfire")
ALL_RISK_TYPES = FLOOD_TYPES + FIRE_TYPES + ("Tornado", "Earthquake", "Drought")


def _label_from_count(n: int) -> int:
    """Map total FEMA disaster declarations (10 yr) → risk tier index."""
    if n == 0:
        return 0   # low
    if n <= 2:
        return 1   # moderate
    if n <= 5:
        return 2   # high
    return 3       # critical


@task(name="build-training-data", log_prints=True)
async def build_training_data() -> tuple[np.ndarray, np.ndarray]:
    from sqlalchemy import text
    from src.storage.db import get_async_session

    log = get_run_logger()
    log.info("Querying FEMA declarations for training dataset…")

    async with get_async_session() as session:
        # One row per (state, county_fips) with feature counts
        rows = await session.execute(text("""
            SELECT
                state,
                county_fips,
                COUNT(*) FILTER (
                    WHERE incident_begin >= NOW() - INTERVAL '10 years'
                ) AS total_10yr,
                COUNT(*) FILTER (
                    WHERE disaster_type = ANY(:flood_types)
                    AND declaration_date >= NOW() - INTERVAL '5 years'
                ) AS flood_events_5yr
            FROM fema_declarations
            WHERE state IS NOT NULL
            GROUP BY state, county_fips
            HAVING COUNT(*) >= 1
        """), {"flood_types": list(FLOOD_TYPES)})
        fema_rows = rows.fetchall()

    if not fema_rows:
        raise RuntimeError("No FEMA data found — run ingest_fema first.")

    log.info("Building features for %d counties…", len(fema_rows))

    # For tracked regions, pull real NOAA + segmentation features
    async with get_async_session() as session:
        # Prefer county_climate_summary (bulk NOAA), fall back to region-based observations
        climate_rows = await session.execute(text("""
            SELECT county_fips, avg_precip_mm, precip_trend
            FROM county_climate_summary
            WHERE avg_precip_mm IS NOT NULL
        """))
        climate_by_fips = {row.county_fips: row for row in climate_rows.fetchall()}

        region_rows = await session.execute(text("""
            SELECT r.state_code, r.county_fips,
                   AVG(o.precipitation_mm) AS avg_precip,
                   COUNT(o.id)             AS noaa_obs_count
            FROM regions r
            JOIN noaa_observations o ON o.region_id = r.id
            WHERE r.state_code IS NOT NULL AND r.county_fips IS NOT NULL
            GROUP BY r.state_code, r.county_fips
        """))
        noaa_by_county = {
            (row.state_code, row.county_fips): row
            for row in region_rows.fetchall()
        }

        seg_rows = await session.execute(text("""
            SELECT r.state_code, r.county_fips,
                   AVG((sr.area_stats->>'vegetation')::float) AS avg_veg,
                   AVG((sr.area_stats->>'urban')::float)      AS avg_urban
            FROM regions r
            JOIN sentinel2_tiles t ON t.region_id = r.id
            JOIN segmentation_results sr ON sr.tile_id = t.id
            WHERE r.state_code IS NOT NULL AND r.county_fips IS NOT NULL
              AND sr.area_stats IS NOT NULL
            GROUP BY r.state_code, r.county_fips
        """))
        seg_by_county = {
            (row.state_code, row.county_fips): row
            for row in seg_rows.fetchall()
        }

        nri_rows = await session.execute(text("""
            SELECT county_fips, risk_score, eal_score, sovi_score,
                   cfld_risks, rfld_risks, wfir_risks, hwav_risks
            FROM fema_nri_county
        """))
        nri_by_fips = {row.county_fips: row for row in nri_rows.fetchall()}

    log.info(
        "Lookup tables: climate_summary=%d, NOAA_region=%d, seg=%d, NRI=%d",
        len(climate_by_fips), len(noaa_by_county), len(seg_by_county), len(nri_by_fips),
    )

    X_rows, y_rows = [], []
    for row in fema_rows:
        state = row.state
        fips = row.county_fips
        flood_5yr = float(row.flood_events_5yr or 0)
        label = _label_from_count(int(row.total_10yr or 0))

        noaa = noaa_by_county.get((state, fips))
        seg = seg_by_county.get((state, fips))
        nri = nri_by_fips.get(fips) if fips else None
        climate = climate_by_fips.get(fips) if fips else None

        if climate:
            precip_trend = float(climate.precip_trend or 0.0)
        elif noaa:
            precip_trend = float(noaa.avg_precip or 0) * 0.01
        else:
            precip_trend = 0.0
        veg_loss = max(0.0, 0.3 - float(seg.avg_veg or 0.3)) if seg else 0.0
        urban = float(seg.avg_urban or 0.1) if seg else 0.1

        nri_risk   = float(nri.risk_score or 0.0) if nri else 0.0
        nri_eal    = float(nri.eal_score  or 0.0) if nri else 0.0
        nri_sovi   = float(nri.sovi_score or 0.0) if nri else 0.0
        nri_flood  = max(
            float(nri.cfld_risks or 0.0), float(nri.rfld_risks or 0.0)
        ) if nri else 0.0
        nri_fire   = float(nri.wfir_risks or 0.0) if nri else 0.0
        nri_heat   = float(nri.hwav_risks or 0.0) if nri else 0.0

        X_rows.append([
            flood_5yr,
            precip_trend,
            veg_loss,
            urban,
            0.0,   # elevation_variance — needs depth results
            0.5,   # infrastructure_age_proxy — static until OSM extraction
            nri_risk,
            nri_eal,
            nri_sovi,
            nri_flood,
            nri_fire,
            nri_heat,
        ])
        y_rows.append(label)

    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_rows, dtype=np.int32)

    counts = np.bincount(y, minlength=4)
    log.info(
        "Dataset: %d samples | low=%d moderate=%d high=%d critical=%d",
        len(y), counts[0], counts[1], counts[2], counts[3],
    )
    return X, y


@task(name="train-and-save", log_prints=True)
def train_and_save(X: np.ndarray, y: np.ndarray) -> dict:
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import label_binarize
    from src.pipeline.classifier import train_classifier, save_classifier_to_s3, RISK_TIERS

    log = get_run_logger()
    log.info("Starting 5-fold CV evaluation…")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_aucs = []
    for train_idx, val_idx in skf.split(X, y):
        model = train_classifier(X[train_idx], y[train_idx])
        proba = model.predict_proba(X[val_idx])
        y_bin = label_binarize(y[val_idx], classes=list(range(len(RISK_TIERS))))
        auc = roc_auc_score(y_bin, proba, multi_class="ovr", average="macro")
        fold_aucs.append(auc)

    mean_auc = float(np.mean(fold_aucs))
    log.info("5-fold mean AUC-ROC: %.4f", mean_auc)

    log.info("Training final model on full dataset…")
    final_model = train_classifier(X, y)
    save_classifier_to_s3(final_model)
    log.info("Model saved to S3 at models/xgboost_risk_classifier.json")

    return {"mean_auc": mean_auc, "n_samples": len(y)}


@flow(name="train_xgboost_classifier", log_prints=True)
async def train_classifier_flow() -> dict:
    """Train the XGBoost risk classifier from real FEMA data and save to S3."""
    X, y = await build_training_data()
    result = train_and_save(X, y)  # sync task — Prefect runs it in a thread pool
    log = get_run_logger()
    log.info(
        "Training complete — AUC: %.4f on %d samples",
        result["mean_auc"], result["n_samples"],
    )
    return result
