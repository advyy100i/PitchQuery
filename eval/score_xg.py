"""Phase 6: score the shipped xG artefact against whatever is in the database.

Deliberately not a training run. `models/train_xg.py` fits and evaluates in one
go and writes docs/benchmark.md; this scores `models/xg_portable.json.gz` — the
599 KB file the hosted API actually loads — and writes eval/out/xg.json for
ci/compare_metrics.py.

That distinction is what makes it usable as a CI gate. Training in CI on a
2,000-possession fixture would produce a different model every time the fixture
changed and a log-loss dominated by sampling noise, so the gate would fire on
nothing. Scoring a fixed artefact on a fixed fixture is deterministic: the
number only moves when the feature extraction, the artefact, or the fixture
moves, which is exactly the set of changes worth blocking a merge over.

Restricted to the competitions recorded inside the artefact as held out, so the
number is always out of sample. If none of them are present — a fixture that
happens to contain no held-out shots — it says so and scores nothing rather
than quietly reporting a training-set score.

Run:
  python eval/score_xg.py
  python eval/score_xg.py --all-comps      # in-sample too, for a sanity check
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402

from core import db  # noqa: E402
from core.xg import UNSUPPORTED_SHOT_TYPES, XGModel  # noqa: E402
from eval import report  # noqa: E402
from models.metrics import metrics  # noqa: E402

# The same explicit column list the trainer uses, and for the same reason:
# statsbomb_xg comes back as a comparison column and must never reach the
# feature dict the model reads.
SELECT = """
SELECT event_id, competition_id, season_id, is_goal, statsbomb_xg,
       distance, angle, body_part, technique, shot_type,
       first_time, under_pressure, play_pattern,
       n_def_in_cone, dist_nearest_def, gk_dist_to_goal, gk_off_line
FROM shots
WHERE shot_type <> 'Penalty' AND distance IS NOT NULL AND angle IS NOT NULL
"""


def load_shots(conn, comps: list = None) -> list:
    sql, params = SELECT, []
    if comps:
        pairs = [tuple(int(v) for v in c.split(":")) for c in comps]
        sql += (" AND (competition_id, season_id) IN "
                "(SELECT * FROM unnest(%s::int[], %s::int[]))")
        params = [[c for c, _ in pairs], [s for _, s in pairs]]
    with conn.cursor() as cur:
        cur.execute(sql, params)
        cols = [d.name for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all-comps", action="store_true",
                    help="score every competition, not just the held-out ones "
                         "(in-sample; useful for a smoke test, never for a claim)")
    ap.add_argument("--model", default=None, help="path to an xg_portable.json.gz")
    args = ap.parse_args()

    model = XGModel.load(args.model) if args.model else XGModel.load()
    comps = None if args.all_comps else model.test_comps

    conn = db.connect()
    rows = load_shots(conn, comps)
    if not rows:
        scope = "any competition" if args.all_comps else f"held-out {model.test_comps}"
        print(f"no scorable shots for {scope} in this database. "
              f"Nothing written — a metrics file with no measurement behind it "
              f"is worse than a missing one.")
        conn.close()
        raise SystemExit(1)

    p_mine = model.predict(rows)
    y = np.array([1 if r["is_goal"] else 0 for r in rows])

    # StatsBomb's own number, where it exists. Not a feature, and not a target —
    # the ceiling this project measures itself against.
    has_sb = np.array([r["statsbomb_xg"] is not None for r in rows])
    p_sb = np.array([r["statsbomb_xg"] if r["statsbomb_xg"] is not None else np.nan
                     for r in rows], dtype=float)

    m = metrics(y, p_mine)
    payload = {
        "logloss": round(m["log_loss"], 6),
        "brier": round(m["brier"], 6),
        "ece": round(m["ece"], 6),
        "auc": round(m["auc"], 6),
        "observed_over_expected": round(m["observed"] / m["expected"], 6),
        "n_shots": len(rows),
        "scope": "all" if args.all_comps else "held-out",
        "test_comps": model.test_comps,
        "unsupported_shot_types": sorted(UNSUPPORTED_SHOT_TYPES),
    }
    if has_sb.sum() >= 50:
        m_sb = metrics(y[has_sb], p_sb[has_sb])
        payload["statsbomb_logloss"] = round(m_sb["log_loss"], 6)
        payload["logloss_gap_to_statsbomb"] = round(
            metrics(y[has_sb], p_mine[has_sb])["log_loss"] - m_sb["log_loss"], 6)

    report.write("xg", payload, conn=conn)
    conn.close()

    print(f"{len(rows):,} shots ({payload['scope']}): log-loss {m['log_loss']:.4f}, "
          f"Brier {m['brier']:.4f}, ECE {m['ece']:.4f}, AUC {m['auc']:.4f}")
    if "statsbomb_logloss" in payload:
        print(f"StatsBomb on the same shots: {payload['statsbomb_logloss']:.4f} "
              f"(gap {payload['logloss_gap_to_statsbomb']:+.4f})")


if __name__ == "__main__":
    db.cli(main)()
