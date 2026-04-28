# tests/evals/test_forecast_eval.py
import csv
import numpy as np
import pytest
import torch
from pathlib import Path

CRPS_THRESHOLD = 1.60  # baseline from chronos-t5-small on NOAA synthetic holdout (90-day horizon)
HOLDOUT_PATH = Path(__file__).parent.parent / "fixtures" / "noaa_holdout.csv"
TRAIN_SIZE = 410   # ~82% of 500
FORECAST_HORIZON = 90


def _crps_ensemble(samples: np.ndarray, observation: float) -> float:
    """CRPS for a single timestep: E|X-y| - 0.5*E|X-X'|"""
    n = len(samples)
    term1 = np.mean(np.abs(samples - observation))
    sorted_s = np.sort(samples)
    k = np.arange(1, n + 1)
    term2 = np.sum((2 * k - n - 1) * sorted_s) / (n * n)
    return float(term1 - term2)


def _mean_crps(forecast_samples: np.ndarray, actuals: np.ndarray) -> float:
    """
    forecast_samples: (n_samples, horizon)
    actuals: (horizon,)
    """
    return float(np.mean([
        _crps_ensemble(forecast_samples[:, t], actuals[t])
        for t in range(len(actuals))
    ]))


@pytest.mark.slow
def test_chronos_crps_below_threshold():
    from chronos import ChronosPipeline

    rows = []
    with open(HOLDOUT_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(float(row["precipitation_mm"]))

    assert len(rows) == 500, f"Expected 500 rows, got {len(rows)}"

    train_series = rows[:TRAIN_SIZE]
    actuals = np.array(rows[TRAIN_SIZE : TRAIN_SIZE + FORECAST_HORIZON])

    pipeline = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-small",
        device_map="cpu",
        torch_dtype=torch.float32,
    )
    context = torch.tensor(train_series, dtype=torch.float32).unsqueeze(0)
    samples = pipeline.predict(context, prediction_length=FORECAST_HORIZON, num_samples=100)
    forecast_samples = samples.squeeze(0).numpy()  # (100, FORECAST_HORIZON)

    crps = _mean_crps(forecast_samples, actuals)
    print(f"\nChronos CRPS on 90-day NOAA holdout: {crps:.4f} (regression threshold: {CRPS_THRESHOLD})")

    assert crps < CRPS_THRESHOLD, (
        f"Chronos CRPS {crps:.4f} exceeds regression threshold {CRPS_THRESHOLD}. "
        "This blocks merge — check model or data."
    )
