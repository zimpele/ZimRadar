# src/pipeline/forecasting.py
import asyncio
import json
import logging
import threading
import numpy as np
import torch
from datetime import datetime, timezone
from sqlalchemy import text
from src.storage.db import get_async_session

logger = logging.getLogger(__name__)

MODEL_ID = "amazon/chronos-t5-small"
MODEL_VERSION = "chronos-t5-small-v1"
FORECAST_HORIZONS = [30, 60, 90]
NUM_SAMPLES = 20
MIN_HISTORY_ROWS = 30


class _ChronosPipeline:
    def __init__(self):
        from chronos import ChronosPipeline as _CP
        self._pipe = _CP.from_pretrained(
            MODEL_ID, device_map="cpu", torch_dtype=torch.float32
        )

    def forecast(self, series: list[float], horizon: int) -> dict:
        context = torch.tensor(series, dtype=torch.float32).unsqueeze(0)
        samples = self._pipe.predict(context, prediction_length=horizon, num_samples=NUM_SAMPLES)
        arr = samples.squeeze(0).numpy()  # (NUM_SAMPLES, horizon)
        return {
            "mean": arr.mean(axis=0).tolist(),
            "q10": np.percentile(arr, 10, axis=0).tolist(),
            "q90": np.percentile(arr, 90, axis=0).tolist(),
            "samples": arr.tolist(),
        }


_chronos_pipeline: _ChronosPipeline | None = None
_chronos_pipeline_lock = threading.Lock()


def _get_chronos_pipeline() -> _ChronosPipeline:
    global _chronos_pipeline
    if _chronos_pipeline is None:
        with _chronos_pipeline_lock:
            if _chronos_pipeline is None:
                _chronos_pipeline = _ChronosPipeline()
    return _chronos_pipeline


def _compute_flood_risk_flag(samples_30d: np.ndarray, historical_precip: list[float]) -> bool:
    """P(any 3 consecutive days exceed the 95th historical percentile) > 0.3"""
    if not historical_precip:
        return False
    if len(samples_30d) == 0:
        return False
    p95 = float(np.percentile(historical_precip, 95))
    count = 0
    for sample in samples_30d:
        for i in range(len(sample) - 2):
            if sample[i] > p95 and sample[i + 1] > p95 and sample[i + 2] > p95:
                count += 1
                break
    return (count / len(samples_30d)) > 0.3


def _compute_fire_risk_flag(temp_samples_30d: np.ndarray, vegetation_trend: float) -> bool:
    """P(any 7 consecutive days > 40°C AND vegetation declining) > 0.3"""
    if len(temp_samples_30d) == 0:
        return False
    if vegetation_trend >= 0:
        return False
    count = 0
    for sample in temp_samples_30d:
        for i in range(len(sample) - 6):
            if all(sample[i + j] > 40.0 for j in range(7)):
                count += 1
                break
    return (count / len(temp_samples_30d)) > 0.3


async def run_forecast_for_region(region_id: int) -> None:
    async with get_async_session() as session:
        rows_result = await session.execute(
            text("""
                SELECT date, precipitation_mm, temp_max_c
                FROM noaa_observations
                WHERE region_id = :rid
                ORDER BY date
            """),
            {"rid": region_id},
        )
        obs = rows_result.fetchall()

    if len(obs) < MIN_HISTORY_ROWS:
        logger.warning(
            "Insufficient NOAA data for region %d (%d rows, need %d)",
            region_id,
            len(obs),
            MIN_HISTORY_ROWS,
        )
        return

    precip_series = [float(r.precipitation_mm or 0.0) for r in obs]
    temp_series = [float(r.temp_max_c or 20.0) for r in obs]

    chronos = await asyncio.to_thread(_get_chronos_pipeline)

    forecasts: dict = {}
    precip_fc_30d: dict = {}
    for horizon in FORECAST_HORIZONS:
        fc = await asyncio.to_thread(chronos.forecast, precip_series, horizon)
        if horizon == 30:
            precip_fc_30d = fc
        forecasts[f"forecast_{horizon}d"] = {k: v for k, v in fc.items() if k != "samples"}

    temp_fc_full = await asyncio.to_thread(chronos.forecast, temp_series, 30)

    precip_samples_30d = np.array(precip_fc_30d["samples"])
    temp_samples_30d = np.array(temp_fc_full["samples"])

    # Vegetation trend from segmentation history
    async with get_async_session() as session:
        veg_result = await session.execute(
            text("""
                SELECT (sr.area_stats->>'vegetation')::float AS veg_pct
                FROM segmentation_results sr
                JOIN sentinel2_tiles t ON t.id = sr.tile_id
                WHERE t.region_id = :rid
                ORDER BY t.date DESC
                LIMIT 10
            """),
            {"rid": region_id},
        )
        veg_data = [float(r.veg_pct) for r in veg_result if r.veg_pct is not None]

    vegetation_trend = (
        float(np.polyfit(range(len(veg_data)), veg_data, 1)[0])
        if len(veg_data) >= 2
        else 0.0
    )

    flood_risk_flag = _compute_flood_risk_flag(precip_samples_30d, precip_series)
    fire_risk_flag = _compute_fire_risk_flag(temp_samples_30d, vegetation_trend)

    async with get_async_session() as session:
        await session.execute(
            text("""
                INSERT INTO forecasts
                    (region_id, forecast_30d, forecast_60d, forecast_90d,
                     flood_risk_flag, fire_risk_flag, model_version, created_at)
                VALUES
                    (:region_id, :forecast_30d::jsonb, :forecast_60d::jsonb, :forecast_90d::jsonb,
                     :flood_risk_flag, :fire_risk_flag, :model_version, :created_at)
            """),
            {
                "region_id": region_id,
                "forecast_30d": json.dumps(forecasts["forecast_30d"]),
                "forecast_60d": json.dumps(forecasts["forecast_60d"]),
                "forecast_90d": json.dumps(forecasts["forecast_90d"]),
                "flood_risk_flag": flood_risk_flag,
                "fire_risk_flag": fire_risk_flag,
                "model_version": MODEL_VERSION,
                "created_at": datetime.now(timezone.utc),
            },
        )
