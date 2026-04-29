# tests/pipeline/test_forecasting.py
import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


def test_compute_flood_risk_flag_triggers_when_threshold_exceeded():
    from src.pipeline.forecasting import _compute_flood_risk_flag

    historical = list(range(1, 101))  # p95 ≈ 95
    high = [96.0, 97.0, 98.0] + [1.0] * 27  # has 3 consecutive > p95
    low = [1.0] * 30
    # 12/20 = 0.6 > 0.3 → True
    samples = np.array([high] * 12 + [low] * 8)
    assert _compute_flood_risk_flag(samples, historical) is True


def test_compute_flood_risk_flag_does_not_trigger_below_threshold():
    from src.pipeline.forecasting import _compute_flood_risk_flag

    historical = list(range(1, 101))
    high = [96.0, 97.0, 98.0] + [1.0] * 27
    low = [1.0] * 30
    # 5/20 = 0.25 < 0.3 → False
    samples = np.array([high] * 5 + [low] * 15)
    assert _compute_flood_risk_flag(samples, historical) is False


def test_compute_fire_risk_flag_triggers_when_hot_and_vegetation_declining():
    from src.pipeline.forecasting import _compute_fire_risk_flag

    hot = [41.0] * 30  # all 30 days > 40°C → any 7 consecutive are > 40°C
    samples = np.array([hot] * 10)  # P = 1.0 > 0.3
    assert _compute_fire_risk_flag(samples, vegetation_trend=-0.01) is True


def test_compute_fire_risk_flag_no_trigger_when_vegetation_stable():
    from src.pipeline.forecasting import _compute_fire_risk_flag

    hot = [41.0] * 30
    samples = np.array([hot] * 10)
    assert _compute_fire_risk_flag(samples, vegetation_trend=0.0) is False


def test_compute_fire_risk_flag_no_trigger_when_cool():
    from src.pipeline.forecasting import _compute_fire_risk_flag

    cool = [20.0] * 30  # never exceeds 40°C
    samples = np.array([cool] * 10)
    assert _compute_fire_risk_flag(samples, vegetation_trend=-0.05) is False


@pytest.mark.asyncio
async def test_run_forecast_for_region_skips_when_insufficient_data():
    from src.pipeline.forecasting import run_forecast_for_region

    mock_rows = MagicMock()
    mock_rows.fetchall.return_value = [
        MagicMock(precipitation_mm=1.0, temp_max_c=20.0)
    ] * 10  # only 10 rows, < 30 minimum

    mock_session = MagicMock()
    mock_session.execute = AsyncMock(return_value=mock_rows)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    with patch("src.pipeline.forecasting.get_async_session", return_value=mock_session):
        await run_forecast_for_region(region_id=1)

    # Only the initial SELECT is called; no INSERT occurs
    assert mock_session.execute.call_count == 1
