"""Score a shot with the trained xG model, using lightgbm and numpy only.

This is the serving half of models/export_xg_portable.py. It reads the exported
artefact — three LightGBM ensembles in LightGBM's own text format, each with a
two-parameter sigmoid — and reproduces `CalibratedClassifierCV.predict_proba`
exactly, without unpickling anything and without pandas.

    p(goal) = mean over members of  1 / (1 + exp(a * margin + b))

`margin` is the raw score, NOT a probability. scikit-learn calibrates whatever
`decision_function` returns and lightgbm's classifier defines it as the margin;
calibrating the probability instead yields numbers wedged between 0.43 and 0.62
which look like plausible xG values on a page. tests/test_xg_portable.py pins
this against the pickle over all 11,185 shots for that reason.

Deliberately absent: any fitting, any dataframe, any sklearn import. This module
runs on a 512 MB box.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np

DEFAULT_PATH = Path(__file__).resolve().parents[1] / "models" / "xg_portable.json.gz"

# Penalties are excluded from training (models/train_xg.py rule 2): fixed
# geometry, a ~0.78 conversion rate, and no freeze-frame context that means
# anything. Scoring one would be extrapolation dressed up as a prediction, so
# the model declines instead and the caller shows StatsBomb's figure alone.
UNSUPPORTED_SHOT_TYPES = {"Penalty"}

SUPPORTED_FORMATS = {"pitchquery-xg/1"}


class XGModel:
    """The calibrated ensemble, ready to score."""

    def __init__(self, doc: dict):
        if doc.get("format") not in SUPPORTED_FORMATS:
            raise ValueError(
                f"unknown artefact format {doc.get('format')!r}; "
                f"this build reads {sorted(SUPPORTED_FORMATS)}")
        import lightgbm as lgb        # imported here so `import core.xg` stays cheap

        self.features: list[str] = doc["features"]
        self.numeric = set(doc["numeric"])
        self.boolean = set(doc["boolean"])
        self.categorical = set(doc["categorical"])
        self.test_comps: list[str] = doc["test_comps"]
        self.trained_with: dict = doc.get("trained_with", {})

        self.members = [{
            "booster": lgb.Booster(model_str=m["booster"]),
            "a": m["a"], "b": m["b"], "levels": m["levels"],
        } for m in doc["members"]]

    @classmethod
    def load(cls, path: Path | str = DEFAULT_PATH) -> "XGModel":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return cls(json.load(f))

    # -- scoring ---------------------------------------------------------------

    def _matrix(self, rows: list[dict], levels: dict) -> np.ndarray:
        """One member's feature matrix. Anything unknown becomes NaN.

        Missing values are not imputed. LightGBM routes them natively, and a
        shot with no freeze frame genuinely has no defender count — filling in a
        mean would invent defenders that were not on the pitch.
        """
        X = np.full((len(rows), len(self.features)), np.nan)
        for j, col in enumerate(self.features):
            if col in self.categorical:
                table = levels[col]
                X[:, j] = [table.get(r.get(col), np.nan) for r in rows]
            elif col in self.boolean:
                X[:, j] = [1.0 if r.get(col) else 0.0 for r in rows]
            else:
                for i, r in enumerate(rows):
                    v = r.get(col)
                    if v is not None:
                        X[i, j] = float(v)
        return X

    def predict(self, rows: list[dict]) -> np.ndarray:
        """Goal probability for each row. Keys are the `shots` table columns."""
        if not rows:
            return np.empty(0)
        stacked = np.empty((len(self.members), len(rows)))
        for k, m in enumerate(self.members):
            margin = m["booster"].predict(self._matrix(rows, m["levels"]), raw_score=True)
            stacked[k] = 1.0 / (1.0 + np.exp(m["a"] * margin + m["b"]))
        return stacked.mean(axis=0)

    def predict_one(self, row: dict) -> float | None:
        """One shot, or None when the model has no business answering."""
        if row.get("shot_type") in UNSUPPORTED_SHOT_TYPES:
            return None
        if row.get("distance") is None or row.get("angle") is None:
            return None       # the two features every tree in the ensemble uses
        return float(self.predict([row])[0])
