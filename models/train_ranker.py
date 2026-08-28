"""Phase 7: replace fixed RRF with a learned fusion.

Reciprocal rank fusion has one parameter and no idea what it is looking at. It
cannot express "this passage is four tokens long and the query described a
twelve-token build-up", which is the short-possession bias documented in
docs/retrieval_eval.md. A ranker with `n_tokens / corpus_median` in its feature
set can.

Read the numbers this produces sceptically, and say so wherever they are quoted.
Thirty queries is a very small training set for a pairwise ranker. That is why:

  * evaluation is leave-one-query-out, never a random split. A random split over
    (query, candidate) pairs would put candidates from the same query on both
    sides and report a number two or three times better than the truth;
  * the comparison is against RRF on exactly the same candidate pool, not
    against a number from a different run;
  * `--promote` refuses to move the champion alias unless LOO NDCG@10 actually
    beats RRF, and the artefact is only written when it does.

The real fix is more labelled queries, which is what Phase 8's click log is for.

Run:
  python models/train_ranker.py                 # train, evaluate, report
  python models/train_ranker.py --promote       # ...and write the artefact
"""
import argparse
import gzip
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from core import db  # noqa: E402
from core.config import DOCS_DIR, REPO_ROOT  # noqa: E402
from core.rank_features import FEATURES, QueryContext, build_matrix  # noqa: E402
from core.retrieval import Filters, Retriever, hydrate, rrf  # noqa: E402
from eval.judge import grade  # noqa: E402
from models import tracking  # noqa: E402

ARTEFACT = REPO_ROOT / "models" / "ranker.json.gz"
QUERIES = REPO_ROOT / "eval" / "queries.yaml"

# The pool the ranker is trained on and, at serve time, reorders. 100 is not
# arbitrary: scoring the whole corpus per query would cost a feature build over
# 67k rows and blow the latency budget the CI gate defends, and the recall of
# the fused top 100 is the ceiling either way.
POOL = 100

# label_gain from the plan. The jump from 1 to 3 to 7 is what tells lambdarank
# that pulling a goal above a shot is worth more than pulling a shot above a
# merely-relevant passage; a linear gain would treat those as equal trades.
LABEL_GAIN = [0, 1, 3, 7]


def corpus_median_tokens(conn) -> float:
    with conn.cursor() as cur:
        cur.execute("SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY n_events) "
                    "FROM possessions")
        return float(cur.fetchone()[0] or 1.0)


def candidates(r: Retriever, conn, q: dict, median: float, use_dense: bool):
    """Feature rows, labels and uids for one query's pool.

    The pool is the union of what each ranker returned, not the fusion's top N.
    Training only on what RRF already liked would teach the model to agree with
    RRF, and the one thing this model has to be able to do is disagree with it.
    """
    f = Filters(**(q.get("filters") or {}))
    allowed = r.candidates(conn, f)
    sparse = r.sparse_scored(q["sequence_hint"], allowed, limit=POOL)
    dense = (r.dense_scored(conn, q["sequence_hint"], filters=f, limit=POOL)
             if use_dense else [])

    s_map = {uid: (score, i + 1.0) for i, (uid, score) in enumerate(sparse)}
    d_map = {uid: (score, i + 1.0) for i, (uid, score) in enumerate(dense)}
    uids = list(dict.fromkeys([u for u, _ in sparse] + [u for u, _ in dense]))
    if not uids:
        return [], [], [], []

    rows = hydrate(conn, uids)
    ctx = QueryContext(q["sequence_hint"], median)
    X = build_matrix(rows, ctx, s_map, d_map)
    y = [grade(q["id"], row) for row in rows]
    ordered = [row["possession_uid"] for row in rows]
    baseline = rrf([u for u, _ in sparse], [u for u, _ in dense]) if dense \
        else [u for u, _ in sparse]
    return X, y, ordered, baseline


def dcg(gains: list, k: int) -> float:
    return sum(g / np.log2(i + 2) for i, g in enumerate(gains[:k]))


def ndcg_at(labels_in_order: list, k: int = 10) -> float:
    """NDCG@k with the same exponential gain the model is trained on.

    Using a different gain here than in `label_gain` would report a model
    against a target it was never optimising, which is a good way to conclude
    that lambdarank does not work.
    """
    gains = [LABEL_GAIN[v] for v in labels_in_order]
    ideal = sorted(gains, reverse=True)
    best = dcg(ideal, k)
    return (dcg(gains, k) / best) if best > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--promote", action="store_true",
                    help="write models/ranker.json.gz if the model beats RRF")
    ap.add_argument("--sparse-only", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from lightgbm import LGBMRanker

    queries = yaml.safe_load(QUERIES.read_text(encoding="utf-8"))["queries"]
    conn = db.connect()
    r = Retriever(conn=conn)
    use_dense = not args.sparse_only and r.dense_available
    median = corpus_median_tokens(conn)
    print(f"{len(queries)} queries, pool {POOL}, corpus median {median:.0f} tokens, "
          f"{'sparse+dense' if use_dense else 'sparse only'}")

    per_query = []
    for q in queries:
        X, y, uids, baseline = candidates(r, conn, q, median, use_dense)
        if not X:
            print(f"  {q['id']}: no candidates, skipped")
            continue
        per_query.append({"id": q["id"], "X": np.array(X, dtype=float),
                          "y": np.array(y, dtype=int), "uids": uids,
                          "baseline": baseline})
    conn.close()

    labelled = [p for p in per_query if p["y"].max() > 0]
    print(f"{len(labelled)}/{len(per_query)} queries have at least one relevant "
          f"candidate in the pool — the rest cannot train or score anything")

    def build():
        return LGBMRanker(
            objective="lambdarank", metric="ndcg", label_gain=LABEL_GAIN,
            # Small on purpose. Thirty groups is not enough data to justify a
            # deep model, and the failure mode of over-fitting here is a ranker
            # that memorises which uids were relevant.
            n_estimators=120, learning_rate=0.06, num_leaves=7,
            min_child_samples=20, subsample=0.9, subsample_freq=1,
            colsample_bytree=0.8, reg_lambda=5.0,
            random_state=args.seed, verbose=-1)

    # -- leave one query out --------------------------------------------------
    model_ndcg, rrf_ndcg, rerank_ms = [], [], []
    for held in labelled:
        train = [p for p in labelled if p["id"] != held["id"]]
        X = np.vstack([p["X"] for p in train])
        y = np.concatenate([p["y"] for p in train])
        groups = [len(p["y"]) for p in train]

        m = build()
        m.fit(X, y, group=groups)

        t0 = time.perf_counter()
        scores = m.predict(held["X"])
        rerank_ms.append((time.perf_counter() - t0) * 1000)

        order = np.argsort(-scores)
        label_of = dict(zip(held["uids"], held["y"]))
        model_ndcg.append(ndcg_at([int(held["y"][i]) for i in order]))
        rrf_ndcg.append(ndcg_at([label_of.get(u, 0) for u in held["baseline"]]))

    won = sum(1 for a, b in zip(model_ndcg, rrf_ndcg) if a > b + 1e-9)
    lost = sum(1 for a, b in zip(model_ndcg, rrf_ndcg) if a < b - 1e-9)
    mean_model, mean_rrf = float(np.mean(model_ndcg)), float(np.mean(rrf_ndcg))
    delta = mean_model - mean_rrf

    print()
    print(f"leave-one-query-out over {len(labelled)} queries")
    print(f"  NDCG@10  RRF     {mean_rrf:.4f}")
    print(f"  NDCG@10  learned {mean_model:.4f}   ({delta:+.4f})")
    print(f"  better on {won}, worse on {lost}, tied on "
          f"{len(labelled) - won - lost}")
    print(f"  rerank of {POOL} candidates: {np.mean(rerank_ms):.2f} ms mean, "
          f"{np.percentile(rerank_ms, 95):.2f} ms p95")

    # A mean over 25-odd queries with this much spread is not significant on its
    # own, so report the spread rather than letting the mean stand alone.
    diffs = np.array(model_ndcg) - np.array(rrf_ndcg)
    se = diffs.std(ddof=1) / np.sqrt(len(diffs)) if len(diffs) > 1 else 0.0
    print(f"  paired difference {diffs.mean():+.4f} +/- {1.96 * se:.4f} (95% CI)")
    significant = abs(diffs.mean()) > 1.96 * se > 0

    # -- final fit on everything ---------------------------------------------
    final = build()
    final.fit(np.vstack([p["X"] for p in labelled]),
              np.concatenate([p["y"] for p in labelled]),
              group=[len(p["y"]) for p in labelled])
    importances = sorted(zip(FEATURES, final.feature_importances_),
                         key=lambda t: -t[1])
    print("\nfeature importances:")
    for name, v in importances:
        print(f"  {name:20} {v}")

    report = {
        "n_queries": len(labelled),
        "pool": POOL,
        "ndcg_at_10_learned": round(mean_model, 6),
        "ndcg_at_10_rrf": round(mean_rrf, 6),
        "delta": round(delta, 6),
        "ci95": round(float(1.96 * se), 6),
        "significant": bool(significant),
        "better_on": won, "worse_on": lost,
        "rerank_p95_ms": round(float(np.percentile(rerank_ms, 95)), 3),
        "median_tokens": median,
        "dense": use_dense,
        "importances": [[n, int(v)] for n, v in importances],
    }
    write_markdown(report)

    with tracking.run("pitchquery-ranker", run_name="lambdarank-loo") as mlrun:
        mlrun.log_params({"n_estimators": 120, "num_leaves": 7, "pool": POOL,
                          "label_gain": LABEL_GAIN, "n_queries": len(labelled),
                          "dense": use_dense, "seed": args.seed})
        mlrun.log_metrics({"ndcg_at_10": mean_model, "ndcg_at_10_rrf": mean_rrf,
                           "delta": delta, "rerank_p95_ms": report["rerank_p95_ms"]})
        mlrun.set_tags({"git_commit": tracking.git_commit(),
                        "evaluation": "leave-one-query-out"})
        mlrun.log_dict(report, "loo_report.json")

    if delta <= 0:
        print(f"\nnot promoting: the learned ranker does not beat RRF "
              f"({mean_model:.4f} vs {mean_rrf:.4f}). RRF stays the default.")
        return
    if not args.promote:
        print(f"\nbeats RRF by {delta:+.4f}. Re-run with --promote to write "
              f"{ARTEFACT.name}.")
        return

    save(final, report)
    print(f"\nwrote {ARTEFACT} ({ARTEFACT.stat().st_size / 1e3:.0f} KB)")


def save(model, report: dict) -> Path:
    """LightGBM's own text format, gzipped, with the constants beside it.

    Same shape as models/xg_portable.json.gz and for the same reason: a pickle
    ties the artefact to the exact scikit-learn that wrote it, and this file has
    to be loadable by an API that installs neither scikit-learn nor a matching
    LightGBM minor version.

    `median_tokens` travels inside the artefact because `n_tokens_ratio` is
    meaningless without the constant it was divided by, and a corpus that grows
    would otherwise change what the model's inputs mean without changing the
    model.
    """
    doc = {
        "format": "pitchquery-ranker/1",
        "features": FEATURES,
        "median_tokens": report["median_tokens"],
        "pool": report["pool"],
        "label_gain": LABEL_GAIN,
        "booster": model.booster_.model_to_string(),
        "loo": {k: report[k] for k in
                ("n_queries", "ndcg_at_10_learned", "ndcg_at_10_rrf", "delta",
                 "ci95", "significant")},
        "trained_with": {"lightgbm": __import__("lightgbm").__version__},
    }
    ARTEFACT.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(ARTEFACT, "wt", encoding="utf-8") as f:
        json.dump(doc, f)
    return ARTEFACT


def write_markdown(rep: dict) -> Path:
    verdict = ("beats" if rep["delta"] > 0 else "does not beat")
    confidence = ("The paired difference clears its own 95% interval, so the "
                  "direction is probably real."
                  if rep["significant"] else
                  "**The paired difference does not clear its own 95% confidence "
                  "interval.** With this many queries the result is a direction, "
                  "not a measurement, and it should be quoted that way.")
    lines = [
        "# Learned ranker",
        "",
        f"An `LGBMRanker` with `objective=lambdarank` reorders the top "
        f"{rep['pool']} candidates that sparse and dense retrieval return. It "
        f"{verdict} the fixed reciprocal rank fusion it replaces.",
        "",
        "| | NDCG@10 |",
        "|---|--:|",
        f"| reciprocal rank fusion | {rep['ndcg_at_10_rrf']:.4f} |",
        f"| learned ranker | **{rep['ndcg_at_10_learned']:.4f}** |",
        f"| difference | {rep['delta']:+.4f} ± {rep['ci95']:.4f} |",
        "",
        f"Better on {rep['better_on']} queries, worse on {rep['worse_on']}, out "
        f"of {rep['n_queries']}.",
        "",
        "## How much to trust this",
        "",
        f"**{rep['n_queries']} training queries is a very small set for a pairwise "
        f"ranker.** Evaluation is leave-one-query-out rather than a random split, "
        f"because a random split over (query, candidate) pairs puts candidates "
        f"from the same query on both sides of it and reports a number several "
        f"times better than the truth. Even so: " + confidence,
        "",
        "The fix is more labelled queries, not a better model. `search_log` and "
        "`click_log` (Phase 8) exist to grow this set out of real use — a result "
        "clicked at rank 5 or below is a query the ranking got wrong, and those "
        "are collected for hand-grading rather than invented.",
        "",
        "## Labels",
        "",
        "Graded 0-3 by `eval.judge.grade`: 0 not relevant, 1 relevant, 2 relevant "
        "and produced a shot, 3 relevant and produced a goal. Because the label "
        "depends on the outcome columns, `ended_in_shot` and `ended_in_goal` are "
        "deliberately absent from the feature set — a model given them would "
        "learn the label instead of the ranking.",
        "",
        "## Features",
        "",
        "| feature | importance |",
        "|---|--:|",
    ]
    lines += [f"| `{n}` | {v} |" for n, v in rep["importances"]]
    lines += [
        "",
        f"Reranking {rep['pool']} candidates costs {rep['rerank_p95_ms']:.1f} ms at "
        f"p95, which is why the ranker reorders a pool rather than scoring the "
        f"corpus.",
        "",
        "Two features from the original plan are absent — `filter match count` and "
        "`competition id match`. Retrieval filters in SQL before it ranks, so both "
        "are constant across every candidate in a query and carry no gradient in a "
        "within-group objective. See `core/rank_features.py`.",
        "",
        "Data source: StatsBomb.",
    ]
    out = DOCS_DIR / "ranker_eval.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return out


if __name__ == "__main__":
    db.cli(main)()
