# tests/pipeline/test_classifier.py
import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


def test_train_classifier_produces_valid_probabilities():
    from src.pipeline.classifier import train_classifier, FEATURE_NAMES, RISK_TIERS

    rng = np.random.default_rng(42)
    X = rng.random((80, len(FEATURE_NAMES))).astype(np.float32)
    y = rng.integers(0, len(RISK_TIERS), size=80)

    model = train_classifier(X, y)

    proba = model.predict_proba(X[:1])
    assert proba.shape == (1, len(RISK_TIERS))
    assert abs(proba[0].sum() - 1.0) < 1e-5


def test_classify_region_features_returns_valid_tier():
    from src.pipeline.classifier import (
        classify_region_features,
        train_classifier,
        FEATURE_NAMES,
        RISK_TIERS,
    )

    rng = np.random.default_rng(0)
    X = rng.random((80, len(FEATURE_NAMES))).astype(np.float32)
    y = rng.integers(0, len(RISK_TIERS), size=80)
    model = train_classifier(X, y)

    with patch("src.pipeline.classifier.load_classifier_from_s3", return_value=model):
        features = {name: 0.5 for name in FEATURE_NAMES}
        tier, confidence, shap_dict = classify_region_features(features)

    assert tier in RISK_TIERS
    assert 0.0 <= confidence <= 1.0
    assert set(shap_dict.keys()) == set(FEATURE_NAMES)


@pytest.mark.asyncio
async def test_run_classification_for_region_writes_risk_assessment():
    from src.pipeline.classifier import run_classification_for_region

    mock_features = {
        "flood_events_5yr": 2.0,
        "avg_precipitation_trend": 0.1,
        "vegetation_loss_pct": 0.05,
        "urban_density": 0.3,
        "elevation_variance": 20.0,
        "infrastructure_age_proxy": 0.5,
    }

    mock_forecast_row = MagicMock(flood_risk_flag=True, fire_risk_flag=False)
    mock_fc_result = MagicMock(fetchone=MagicMock(return_value=mock_forecast_row))

    insert_calls = []
    mock_session = MagicMock()
    mock_session.execute = AsyncMock(
        side_effect=[
            mock_fc_result,
            AsyncMock(side_effect=lambda *a, **kw: insert_calls.append(kw)),
        ]
    )
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    with (
        patch("src.pipeline.classifier.build_features_for_region", return_value=mock_features),
        patch(
            "src.pipeline.classifier.classify_region_features", return_value=("moderate", 0.7, {})
        ),
        patch("src.pipeline.classifier.get_async_session", return_value=mock_session),
    ):
        await run_classification_for_region(region_id=1)

    # Both SELECT (forecast) and INSERT (risk_assessment) were executed
    assert mock_session.execute.call_count == 2


def test_composite_score_formula():
    from src.pipeline.classifier import _composite_score
    from src.config import get_settings

    from src.pipeline.classifier import TIER_WEIGHTS

    w1, w2, w3 = get_settings().risk_weights
    tier = "high"
    score = _composite_score(tier=tier, confidence=0.8, flood_flag=True, fire_flag=False)
    expected = w1 * 0.8 * TIER_WEIGHTS[tier] + w2 * 1.0 + w3 * 0.0
    assert abs(score - expected) < 1e-6
