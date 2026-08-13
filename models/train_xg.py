"""Train the two xG models and hold out whole competitions (plan §8).

Three models are compared:
  baseline    logistic regression on distance and angle only
  mine        LightGBM on geometry + situation + freeze-frame context
  statsbomb   the shipped statsbomb_xg column — not trained here, the ceiling

Four rules from §8 are enforced in code rather than trusted to discipline:

  1. Split by competition/season, never by shot. Shots inside one match share a
     game state, a pitch and often a shooter; a random shot split leaks and
     inflates every metric. GROUPS below are whole competition/seasons.
  2. Penalties and own goals are dropped. A penalty is a fixed-geometry event
     with a ~0.78 conversion rate and it flatters AUC enormously.
  3. statsbomb_xg is NEVER a feature. The SELECT lists columns explicitly for
     this reason — `SELECT *` into a dataframe is exactly how that leak happens.
  4. Nothing is fitted on the test competitions, including the imputation and
     the category encodings.

Run:
  python models/train_xg.py
  python models/train_xg.py --test-comps 43:106 55:282
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from core import db  # noqa: E402
from core.config import INDEX_DIR  # noqa: E402

MODEL_PATH = INDEX_DIR / "xg_models.pkl"

# Held out entirely. Chosen before looking at any score: one men's and one
# women's major tournament, so the test set is not a single style of football.
DEFAULT_TEST = ["43:106", "72:107"]

NUMERIC = ["distance", "angle"]
CONTEXT_NUMERIC = ["n_def_in_cone", "dist_nearest_def", "gk_dist_to_goal", "gk_off_line"]
CATEGORICAL = ["body_part", "technique", "shot_type", "play_pattern"]
BOOLEAN = ["first_time", "under_pressure"]

# Explicit whitelist. statsbomb_xg is fetched only as a comparison column and is
# removed from the feature frame before anything is fitted.
SELECT = """
SELECT s.event_id, s.competition_id, s.season_id, s.is_goal, s.statsbomb_xg,
       s.distance, s.angle, s.body_part, s.technique, s.shot_type,
       s.first_time, s.under_pressure, s.play_pattern,
       s.n_def_in_cone, s.dist_nearest_def, s.gk_dist_to_goal, s.gk_off_line
FROM shots s
WHERE s.shot_type <> 'Penalty'
  AND s.distance IS NOT NULL
  AND s.angle IS NOT NULL
"""


def load(conn) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(SELECT)
        cols = [d.name for d in cur.description]
        df = pd.DataFrame(cur.fetchall(), columns=cols)
    df["group"] = df["competition_id"].astype(str) + ":" + df["season_id"].astype(str)
    for c in NUMERIC + CONTEXT_NUMERIC + ["statsbomb_xg"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["is_goal"] = df["is_goal"].astype(int)
    return df


def feature_frame(df: pd.DataFrame, categories: dict = None):
    """Build the model matrix. Returns (X, categories).

    `categories` is fitted on train only and reused for test, so a body part
    that appears only in the held-out competitions cannot silently become a new
    column and shift every other column along.
    """
    X = df[NUMERIC + CONTEXT_NUMERIC].copy()
    for c in BOOLEAN:
        X[c] = df[c].fillna(False).astype(int)

    fitted = {}
    for c in CATEGORICAL:
        cats = categories[c] if categories else sorted(df[c].dropna().unique())
        fitted[c] = cats
        X[c] = pd.Categorical(df[c], categories=cats).codes   # -1 for unseen
        X[c] = X[c].astype("category")
    return X, fitted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-comps", nargs="+", default=DEFAULT_TEST,
                    metavar="COMP:SEASON", help="competition/seasons held out")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from lightgbm import LGBMClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    conn = db.connect()
    df = load(conn)
    conn.close()

    test_mask = df["group"].isin(args.test_comps)
    if not test_mask.any():
        print(f"none of {args.test_comps} are in the data. Available:\n"
              f"  {sorted(df['group'].unique())}")
        return

    train, test = df[~test_mask].copy(), df[test_mask].copy()
    print(f"train {len(train):,} shots over {train['group'].nunique()} competition/seasons")
    print(f"test  {len(test):,} shots over {test['group'].nunique()}: {args.test_comps}")
    print(f"goal rate: train {train['is_goal'].mean():.3f}, test {test['is_goal'].mean():.3f}")

    # Sanity: the split must be clean. If a group appears in both, rule 1 is broken.
    overlap = set(train["group"]) & set(test["group"])
    assert not overlap, f"competition/season in both splits: {overlap}"

    X_train, cats = feature_frame(train)
    X_test, _ = feature_frame(test, cats)
    y_train, y_test = train["is_goal"].values, test["is_goal"].values

    # -- baseline: distance and angle only -----------------------------------
    base = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    base.fit(X_train[NUMERIC], y_train)
    p_base = base.predict_proba(X_test[NUMERIC])[:, 1]

    # -- mine: geometry + situation + freeze-frame context --------------------
    # Freeze-frame columns keep their NaNs: LightGBM routes missing values
    # natively, and imputing them would invent defenders that were not there.
    #
    # Raw gradient boosting on 7.8k shots at a 10% base rate comes out sharp but
    # over-confident: it ranks chances well (AUC) while getting the level wrong
    # (O/E), and log-loss charges for exactly that. So the probabilities are
    # calibrated, and the calibrator is CHOSEN by grouped cross-validation on the
    # training competitions only — never on the held-out ones.
    def make_lgbm():
        return LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=15,
            min_child_samples=60, subsample=0.9, subsample_freq=1,
            colsample_bytree=0.9, reg_lambda=5.0,
            random_state=args.seed, verbose=-1)

    folds = list(GroupKFold(n_splits=5).split(X_train, y_train, train["group"]))
    candidates = {
        "raw": lambda: make_lgbm(),
        "isotonic": lambda: CalibratedClassifierCV(make_lgbm(), method="isotonic", cv=3),
        "sigmoid": lambda: CalibratedClassifierCV(make_lgbm(), method="sigmoid", cv=3),
    }
    cv_scores = {}
    for name, build in candidates.items():
        oof = np.zeros(len(y_train))
        for tr, va in folds:
            m = build()
            m.fit(X_train.iloc[tr], y_train[tr])
            oof[va] = m.predict_proba(X_train.iloc[va])[:, 1]
        cv_scores[name] = log_loss(y_train, np.clip(oof, 1e-6, 1 - 1e-6), labels=[0, 1])
        print(f"  cv log-loss ({name:8}): {cv_scores[name]:.4f}")

    best = min(cv_scores, key=cv_scores.get)
    print(f"  -> using '{best}' (chosen on training competitions only)")
    mine = candidates[best]()
    mine.fit(X_train, y_train)
    p_mine = mine.predict_proba(X_test)[:, 1]

    p_sb = test["statsbomb_xg"].values

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"baseline": base, "mine": mine, "categories": cats,
                     "test_comps": args.test_comps,
                     "features": list(X_train.columns)}, f)

    out = pd.DataFrame({
        "event_id": test["event_id"].astype(str).values,
        "group": test["group"].values,
        "is_goal": y_test,
        "p_base": p_base,
        "p_mine": p_mine,
        "p_sb": p_sb,
        "distance": test["distance"].values,
        "shot_type": test["shot_type"].values,
        "has_freeze_frame": test["n_def_in_cone"].notna().values,
    })
    # CSV, not parquet: a few thousand rows does not justify a pyarrow dependency.
    pred_path = INDEX_DIR / "xg_test_predictions.csv"
    out.to_csv(pred_path, index=False)

    print(f"\nwrote {MODEL_PATH.name} and {pred_path.name}")

    # A calibrated model wraps the booster, so importances come from a plain
    # LightGBM refit on the same features — reported for interpretation only.
    ref = make_lgbm()
    ref.fit(X_train, y_train)
    importances = sorted(zip(X_train.columns, ref.feature_importances_),
                         key=lambda t: -t[1])
    print("feature importances (top 8):")
    for name, v in importances[:8]:
        print(f"  {name:20} {v}")
    print("\nnow run: python models/evaluate_xg.py")

    (INDEX_DIR / "xg_split.json").write_text(json.dumps({
        "test_comps": args.test_comps,
        "train_groups": sorted(train["group"].unique()),
        "n_train": int(len(train)), "n_test": int(len(test)),
        "calibration": best,
        "cv_log_loss": {k: round(v, 4) for k, v in cv_scores.items()},
        "importances": [[n, int(v)] for n, v in importances],
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
