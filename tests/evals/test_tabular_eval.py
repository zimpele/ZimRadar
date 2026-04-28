# tests/evals/test_tabular_eval.py
import json
import numpy as np
import pytest
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize

AUC_THRESHOLD = 0.80
LABELS_PATH = Path(__file__).parent.parent / "fixtures" / "xgboost_labels.json"


@pytest.mark.slow
def test_xgboost_auc_roc_above_threshold():
    from src.pipeline.classifier import train_classifier, FEATURE_NAMES, RISK_TIERS

    with open(LABELS_PATH) as f:
        records = json.load(f)

    X = np.array([[r[k] for k in FEATURE_NAMES] for r in records], dtype=np.float32)
    y = np.array([r["label"] for r in records])
    classes = list(range(len(RISK_TIERS)))

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_aucs = []

    for train_idx, val_idx in skf.split(X, y):
        model = train_classifier(X[train_idx], y[train_idx])
        proba = model.predict_proba(X[val_idx])
        y_bin = label_binarize(y[val_idx], classes=classes)
        auc = roc_auc_score(y_bin, proba, multi_class="ovr", average="macro")
        fold_aucs.append(auc)

    mean_auc = float(np.mean(fold_aucs))
    print(f"\nXGBoost 5-fold mean AUC-ROC: {mean_auc:.4f} (threshold: {AUC_THRESHOLD})")

    assert mean_auc >= AUC_THRESHOLD, (
        f"XGBoost AUC-ROC {mean_auc:.4f} is below regression threshold {AUC_THRESHOLD}. "
        "This blocks merge — check features or model config."
    )
