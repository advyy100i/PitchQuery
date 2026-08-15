"""The deployed xG model must be the model that was evaluated.

docs/benchmark.md reports a number produced by the pickle that models/train_xg.py
writes. The hosted API serves models/xg_portable.json.gz instead, which is a
hand-written re-serialisation — three LightGBM ensembles in text form and a
sigmoid each. Nothing about that translation is checked by the type system, and
its most likely failure is silent: feed the calibrator a probability instead of
a raw margin and every prediction lands between 0.43 and 0.62, which reads as
believable xG on a page while being wrong on every shot.

So the two are compared numerically, over the whole corpus rather than a
sample, and the tolerance is exact. They are the same arithmetic on the same
floats; anything other than a zero difference means the translation drifted.

The comparison needs the pickle and the database, so it skips on a machine that
has neither — a deployment box, or a checkout that has not run the ingest. The
tests that need nothing always run.
"""
import gzip
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.xg import UNSUPPORTED_SHOT_TYPES, XGModel  # noqa: E402
from models.export_xg_portable import PORTABLE_PATH  # noqa: E402
from models.train_xg import (BOOLEAN, CATEGORICAL, CONTEXT_NUMERIC,  # noqa: E402
                             MODEL_PATH, NUMERIC)

pytestmark = pytest.mark.skipif(
    not PORTABLE_PATH.exists(),
    reason=f"{PORTABLE_PATH.name} not built — run models/train_xg.py")


@pytest.fixture(scope="module")
def model():
    return XGModel.load(PORTABLE_PATH)


def test_artefact_declares_its_provenance():
    """A model file that cannot say what produced it is not reproducible."""
    with gzip.open(PORTABLE_PATH, "rt", encoding="utf-8") as f:
        doc = json.load(f)
    assert doc["format"] == "pitchquery-xg/1"
    assert doc["calibration"] == "sigmoid"
    assert doc["test_comps"], "the held-out competitions must travel with the model"
    assert doc["trained_with"]["lightgbm"]


def test_features_are_the_trained_features(model):
    """statsbomb_xg is the comparison column and must never be an input."""
    assert set(model.features) == set(
        NUMERIC + CONTEXT_NUMERIC + BOOLEAN + CATEGORICAL)
    assert "statsbomb_xg" not in model.features
    assert "is_goal" not in model.features


def test_a_close_shot_beats_a_long_one(model):
    """A sanity check that survives a retrain: geometry still dominates."""
    base = {"body_part": "Right Foot", "technique": "Normal", "shot_type": "Open Play",
            "play_pattern": "Regular Play", "first_time": False, "under_pressure": False,
            "n_def_in_cone": 1, "dist_nearest_def": 3.0,
            "gk_dist_to_goal": 2.0, "gk_off_line": 2.0}
    close = model.predict_one({**base, "distance": 6.0, "angle": 1.0})
    far = model.predict_one({**base, "distance": 32.0, "angle": 0.15})
    assert 0.0 < far < close < 1.0


def test_penalties_are_declined_not_guessed(model):
    """Trained without them, so it must not extrapolate into them."""
    pen = {"distance": 12.0, "angle": 0.9, "body_part": "Right Foot",
           "technique": "Normal", "shot_type": "Penalty", "play_pattern": "Other",
           "first_time": False, "under_pressure": False, "n_def_in_cone": 0,
           "dist_nearest_def": None, "gk_dist_to_goal": None, "gk_off_line": None}
    assert "Penalty" in UNSUPPORTED_SHOT_TYPES
    assert model.predict_one(pen) is None


def test_missing_freeze_frame_is_not_imputed(model):
    """A shot with no freeze frame still scores — LightGBM routes the NaNs.

    Half the corpus has no freeze frame. If missing context silently became a
    zero, those shots would be scored as though nobody was defending.
    """
    bare = {"distance": 14.0, "angle": 0.5, "body_part": "Right Foot",
            "technique": "Normal", "shot_type": "Open Play",
            "play_pattern": "Regular Play", "first_time": False,
            "under_pressure": False, "n_def_in_cone": None, "dist_nearest_def": None,
            "gk_dist_to_goal": None, "gk_off_line": None}
    crowded = {**bare, "n_def_in_cone": 4, "dist_nearest_def": 0.4,
               "gk_dist_to_goal": 1.0, "gk_off_line": 1.0}
    p_bare, p_crowded = model.predict_one(bare), model.predict_one(crowded)
    assert 0.0 < p_bare < 1.0
    assert p_bare != p_crowded, "context features are being ignored"


def test_unknown_category_does_not_shift_the_other_columns(model):
    """A body part the model never saw must go missing, not become column zero."""
    base = {"distance": 14.0, "angle": 0.5, "technique": "Normal",
            "shot_type": "Open Play", "play_pattern": "Regular Play",
            "first_time": False, "under_pressure": False, "n_def_in_cone": 1,
            "dist_nearest_def": 3.0, "gk_dist_to_goal": 2.0, "gk_off_line": 2.0}
    unseen = model.predict_one({**base, "body_part": "Shoulder Blade"})
    head = model.predict_one({**base, "body_part": "Head"})
    assert 0.0 < unseen < 1.0
    assert unseen != head


def test_matches_the_pickle_on_every_shot():
    """The one that matters: the deployed model IS the evaluated model."""
    if not MODEL_PATH.exists():
        pytest.skip(f"{MODEL_PATH.name} not present — nothing to compare against")
    pd = pytest.importorskip("pandas", reason="the pickle needs pandas to score")
    try:
        from core import db
        from models.train_xg import load
        conn = db.connect()
    except Exception as exc:
        pytest.skip(f"no database: {exc}")

    try:
        df = load(conn)
    finally:
        conn.close()

    with open(MODEL_PATH, "rb") as f:
        blob = pickle.load(f)

    X = df[NUMERIC + CONTEXT_NUMERIC].astype(float).copy()
    for c in BOOLEAN:
        X[c] = df[c].fillna(False).astype(int)
    for c in CATEGORICAL:
        X[c] = pd.Categorical(df[c], categories=blob["categories"][c]).codes
        X[c] = X[c].astype("category")
    reference = blob["mine"].predict_proba(X[blob["features"]])[:, 1]

    ported = XGModel.load(PORTABLE_PATH).predict(df.to_dict("records"))

    assert len(reference) == len(ported) > 1000
    np.testing.assert_array_equal(ported, reference)
