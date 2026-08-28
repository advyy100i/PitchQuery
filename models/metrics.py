"""Scoring functions for probabilistic predictions.

Split out of models/evaluate_xg.py so that eval/score_xg.py — which CI runs on
every pull request — does not import matplotlib to compute a log-loss. The CI
job installs numpy, scikit-learn and lightgbm and nothing else heavier; a
plotting stack is not part of a metric gate.
"""
import numpy as np
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


def expected_calibration_error(y, p, bins: int = 10) -> float:
    """Mean |predicted - observed| across equal-count bins, weighted by size.

    Equal-count, not equal-width. xG is overwhelmingly concentrated below 0.2, so
    ten equal-width bins put almost every shot in the first one and then report a
    calibration error computed from a handful of rebounds and tap-ins.
    """
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, dtype=float)
    edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return float(abs(p.mean() - y.mean()))
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, len(edges) - 2)
    total, err = 0, 0.0
    for b in range(len(edges) - 1):
        m = idx == b
        n = int(m.sum())
        if n:
            err += n * abs(p[m].mean() - y[m].mean())
            total += n
    return float(err / total) if total else 0.0


def metrics(y, p) -> dict:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {
        "log_loss": log_loss(y, p, labels=[0, 1]),
        "brier": brier_score_loss(y, p),
        "auc": roc_auc_score(y, p),
        # Discrimination without calibration is not an xG model, and neither AUC
        # nor log-loss reports the level error on its own.
        "ece": expected_calibration_error(y, p),
        "expected": float(p.sum()),
        "observed": int(y.sum()),
    }
