"""Re-serialise the trained xG model into something safe to deploy.

`train_xg.py` writes a pickle, which is the right format for a local experiment
and the wrong one for a hosted service. Two reasons:

  1. It is fragile. The pickle holds a `CalibratedClassifierCV` wrapping three
     `LGBMClassifier` objects, so restoring it requires scikit-learn and
     lightgbm to be close enough to the versions that wrote it. Nothing warns
     you when they are not — you get an exception at import, or worse, a silent
     behaviour change. Pinning both libraries on the server to match a laptop is
     a standing maintenance cost for a model that does not change.

  2. It drags pandas onto the serving box. The pickle is fitted on a pandas
     frame with categorical dtypes, so scoring one shot means reproducing that
     frame. pandas is ~70 MB resident for what is, at serving time, twelve
     numbers in a row.

So the model is exported to what it actually is: three gradient-boosted
ensembles and a two-parameter sigmoid on top of each.

    p(goal) = mean over the 3 members of  1 / (1 + exp(a_i * margin_i + b_i))

LightGBM's own text format is a documented, versioned interchange format that
newer releases read — unlike a pickle, which is a snapshot of Python objects.
The sigmoids are two floats each. Everything else is a lookup table.

The catch is that scikit-learn feeds the calibrator `decision_function`, not
`predict_proba` — the raw margin, not a probability. Calibrating the
probability instead gives outputs pinned between 0.43 and 0.62 that still look
plausible on a page, which is exactly the kind of error a smoke test misses.
`tests/test_xg_portable.py` therefore asserts the exported model matches the
pickle on every shot in the corpus rather than on a handful.

Run:
  python models/export_xg_portable.py           # writes models/xg_portable.json.gz
  python models/export_xg_portable.py --check   # also diff against the pickle
"""
import argparse
import gzip
import json
import pickle
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.train_xg import (BOOLEAN, CATEGORICAL, CONTEXT_NUMERIC,  # noqa: E402
                             MODEL_PATH, NUMERIC)

# Committed to the repo, unlike the pickle. It is ~1 MB gzipped, and shipping it
# is what lets the hosted API serve my_xg without a build step that needs a
# database, a training run and 8 GB of RAM.
PORTABLE_PATH = Path(__file__).resolve().parent / "xg_portable.json.gz"

FORMAT = "pitchquery-xg/1"


def _levels(booster, categories: dict) -> dict:
    """Map each raw category string to the integer LightGBM expects.

    Two hops, and both matter:

      value -> code      the position in the category list fitted on the
                         TRAINING competitions (models/train_xg.py), so a body
                         part seen only in the held-out data stays unknown
                         rather than quietly becoming a new column
      code  -> slot      LightGBM re-indexed those codes when it consumed the
                         pandas categorical dtype, and recorded the order in
                         `pandas_categorical`

    Composing them here means serving does one dict lookup and cannot get the
    order wrong. A value missing from either hop maps to null, which the scorer
    turns into NaN — LightGBM routes missing values natively, which is the right
    answer for a category the model never saw.
    """
    out = {}
    for k, col in enumerate(CATEGORICAL):
        order = booster.pandas_categorical[k]
        slot = {code: i for i, code in enumerate(order)}
        out[col] = {value: slot[code] for code, value in enumerate(categories[col])
                    if code in slot}
    return out


def export(pickle_path: Path = None, out_path: Path = None) -> Path:
    pickle_path = pickle_path or MODEL_PATH
    out_path = out_path or PORTABLE_PATH

    with open(pickle_path, "rb") as f:
        blob = pickle.load(f)
    mine = blob["mine"]

    # The exporter only knows how to flatten a sigmoid ensemble. If the CV in
    # train_xg.py ever picks isotonic instead, that is a step function per
    # member and this file must learn to carry it — fail loudly rather than
    # export something that scores differently from what was evaluated.
    if getattr(mine, "method", None) != "sigmoid":
        raise SystemExit(
            f"expected a sigmoid-calibrated model, found {getattr(mine, 'method', type(mine))}. "
            "Re-run models/train_xg.py or extend this exporter.")

    members = []
    for cc in mine.calibrated_classifiers_:
        booster = cc.estimator.booster_
        cal = cc.calibrators[0]
        members.append({
            "booster": booster.model_to_string(),
            # p = 1 / (1 + exp(a * margin + b)) — sklearn's _SigmoidCalibration.
            "a": float(cal.a_),
            "b": float(cal.b_),
            "levels": _levels(booster, blob["categories"]),
        })

    import lightgbm
    import sklearn

    doc = {
        "format": FORMAT,
        "exported": date.today().isoformat(),
        # Recorded for provenance, NOT required at load time — that is the whole
        # point of not shipping the pickle.
        "trained_with": {"lightgbm": lightgbm.__version__,
                         "scikit-learn": sklearn.__version__},
        "calibration": "sigmoid",
        "test_comps": blob["test_comps"],
        "features": blob["features"],
        "numeric": NUMERIC + CONTEXT_NUMERIC,
        "boolean": BOOLEAN,
        "categorical": CATEGORICAL,
        "members": members,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out_path, "wt", encoding="utf-8") as f:
        json.dump(doc, f)

    kb = out_path.stat().st_size / 1024
    print(f"wrote {out_path.name}  {kb:,.0f} KB  "
          f"({len(members)} members, {len(doc['features'])} features)")
    return out_path


def check(pickle_path: Path = None, out_path: Path = None) -> float:
    """Score every non-penalty shot both ways and report the largest gap."""
    import numpy as np
    import pandas as pd

    from core import db
    from core.xg import XGModel
    from models.train_xg import load

    pickle_path = pickle_path or MODEL_PATH
    with open(pickle_path, "rb") as f:
        blob = pickle.load(f)

    conn = db.connect()
    df = load(conn)
    conn.close()

    X = df[NUMERIC + CONTEXT_NUMERIC].astype(float).copy()
    for c in BOOLEAN:
        X[c] = df[c].fillna(False).astype(int)
    for c in CATEGORICAL:
        X[c] = pd.Categorical(df[c], categories=blob["categories"][c]).codes
        X[c] = X[c].astype("category")
    reference = blob["mine"].predict_proba(X[blob["features"]])[:, 1]

    model = XGModel.load(out_path or PORTABLE_PATH)
    ported = model.predict(df.to_dict("records"))

    worst = float(np.abs(reference - ported).max())
    print(f"checked {len(df):,} shots — largest difference {worst:.3e}")
    return worst


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="score the whole corpus both ways and diff (needs the database)")
    args = ap.parse_args()
    export()
    if args.check:
        check()
