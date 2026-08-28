"""Serving half of Phase 7: load models/ranker.json.gz and reorder a pool.

Mirrors core/xg.py deliberately. The artefact is LightGBM's own text format
plus the constants the features need, so loading it costs lightgbm and numpy and
nothing else — no scikit-learn, no pickle, no version lock. A 512 MB box can
serve this.

Reordering only. The fused list decides WHICH hundred possessions are in play;
this decides the order of those hundred. Scoring the corpus would mean building
features for 67k rows per query, which is minutes, not milliseconds.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np

from core.rank_features import FEATURES, QueryContext, build_matrix

DEFAULT_PATH = Path(__file__).resolve().parents[1] / "models" / "ranker.json.gz"
SUPPORTED_FORMATS = {"pitchquery-ranker/1"}


class Ranker:
    def __init__(self, doc: dict):
        if doc.get("format") not in SUPPORTED_FORMATS:
            raise ValueError(
                f"unknown artefact format {doc.get('format')!r}; "
                f"this build reads {sorted(SUPPORTED_FORMATS)}")
        # A model trained against a different feature list would still predict —
        # LightGBM only checks the column count — and would silently rank on the
        # wrong columns. Refuse instead.
        if doc.get("features") != FEATURES:
            raise ValueError(
                "the artefact was trained on a different feature list than "
                "core/rank_features.FEATURES. Retrain with "
                "`python models/train_ranker.py --promote`.\n"
                f"  artefact: {doc.get('features')}\n"
                f"  code:     {FEATURES}")

        import lightgbm as lgb

        self.booster = lgb.Booster(model_str=doc["booster"])
        # The corpus statistic n_tokens_ratio was divided by at training time.
        # It travels with the model because a corpus that grew would otherwise
        # change what the model's inputs mean without changing the model.
        self.median_tokens = float(doc["median_tokens"])
        self.pool = int(doc.get("pool", 100))
        self.loo = doc.get("loo", {})
        self.trained_with = doc.get("trained_with", {})

    @classmethod
    def load(cls, path: Path | str = DEFAULT_PATH) -> "Ranker":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return cls(json.load(f))

    def rerank(self, rows: list, sequence_hint: str, sparse: dict,
               dense: dict) -> list:
        """Return `rows` reordered, best first.

        `sparse` and `dense` map uid -> (score, rank), exactly as the trainer
        builds them, because both call core.rank_features.build_matrix.
        """
        if not rows:
            return rows
        ctx = QueryContext(sequence_hint, self.median_tokens)
        X = np.array(build_matrix(rows, ctx, sparse, dense), dtype=float)
        scores = self.booster.predict(X)
        order = np.argsort(-scores)
        return [rows[i] for i in order]

    @property
    def summary(self) -> str:
        loo = self.loo
        if not loo:
            return "loaded"
        return (f"loaded (leave-one-query-out NDCG@10 {loo.get('ndcg_at_10_learned'):.3f} "
                f"vs RRF {loo.get('ndcg_at_10_rrf'):.3f} over "
                f"{loo.get('n_queries')} queries"
                + ("" if loo.get("significant") else ", not significant") + ")")
