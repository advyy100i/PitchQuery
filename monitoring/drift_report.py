"""Phase 9: measure the difference between what the model trained on and what it
is asked about.

The xG model holds out the 2022 men's World Cup and the 2023 Women's World Cup.
docs/benchmark.md reports how well it does on them. This answers the other half
of that question — *why*, and specifically which features look different in the
held-out football than in the training football.

That turns a known limitation into a measurement. "The model is trained mostly
on European club and men's international football and the held-out set includes
a women's World Cup" is a caveat. "Keeper distance and defenders-in-cone are the
two columns that shift, by this much, in this direction" is a finding, and it is
the one an interviewer can actually ask about.

Reference is the training competitions, current is the held-out ones. Not
train-vs-time: nothing here is a production stream, and pretending otherwise
would be the kind of monitoring theatre that measures a clock rather than a
model.

Run:
  python monitoring/drift_report.py
  python monitoring/drift_report.py --split gender      # men's vs women's football
"""
import argparse
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd  # noqa: E402

from core import db  # noqa: E402
from core.config import DOCS_DIR  # noqa: E402
from core.xg import XGModel  # noqa: E402

OUT_DIR = DOCS_DIR / "drift"

# Exactly the columns the model reads. Drift in a column the model does not use
# is not drift that matters, and including one would put a number in the report
# that cannot affect anything.
NUMERIC = ["distance_m", "angle_rad", "n_def_in_cone", "dist_nearest_def",
           "gk_dist_to_goal", "gk_off_line"]
CATEGORICAL = ["body_part", "technique", "shot_type", "play_pattern"]

# The women's competitions in this corpus. Used by --split gender, which asks a
# different and more pointed question than the held-out split: the held-out set
# mixes a men's and a women's tournament, so drift there is confounded.
WOMENS = [(72, 107), (53, 106), (53, 315)]

SELECT = f"""
SELECT comp_season, competition_id, season_id,
       {', '.join(NUMERIC)}, {', '.join(CATEGORICAL)}
FROM analytics.mart_xg_features
"""


def load(conn) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(SELECT)
        cols = [d.name for d in cur.description]
        df = pd.DataFrame(cur.fetchall(), columns=cols)
    for c in NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def split_masks(df: pd.DataFrame, how: str):
    """(reference mask, current mask, label for each)."""
    if how == "gender":
        womens = df.apply(
            lambda r: (r["competition_id"], r["season_id"]) in WOMENS, axis=1)
        return ~womens, womens, ("men's football", "women's football")
    heldout = df["comp_season"].isin(XGModel.load().test_comps)
    return ~heldout, heldout, ("training competitions", "held-out competitions")


def population_shift(ref: pd.DataFrame, cur: pd.DataFrame) -> list:
    """Standardised mean difference per numeric column, largest first.

    Evidently decides drifted/not-drifted with a statistical test, which on
    thousands of shots calls almost everything drifted — with n this large a
    trivial difference is still significant. Cohen's d says how BIG the shift is
    rather than how confident we are that it is nonzero, and size is the only
    version of the question that has an answer worth writing down.
    """
    out = []
    for col in NUMERIC:
        a, b = ref[col].dropna(), cur[col].dropna()
        if len(a) < 30 or len(b) < 30:
            continue
        pooled = ((a.var(ddof=1) * (len(a) - 1) + b.var(ddof=1) * (len(b) - 1))
                  / max(len(a) + len(b) - 2, 1)) ** 0.5
        d = (b.mean() - a.mean()) / pooled if pooled else 0.0
        out.append({"feature": col, "reference_mean": float(a.mean()),
                    "current_mean": float(b.mean()), "cohens_d": float(d),
                    "n_reference": int(len(a)), "n_current": int(len(b))})
    return sorted(out, key=lambda r: -abs(r["cohens_d"]))


def evidently_html(ref: pd.DataFrame, cur: pd.DataFrame, path: Path) -> bool:
    """Write the Evidently drift report. Returns whether it was produced."""
    try:
        from evidently import DataDefinition, Dataset, Report
        from evidently.presets import DataDriftPreset
    except ImportError:
        print("evidently is not installed — skipping the HTML report "
              "(pip install evidently). The table below is computed here and "
              "does not depend on it.")
        return False

    cols = NUMERIC + CATEGORICAL
    definition = DataDefinition(numerical_columns=NUMERIC,
                                categorical_columns=CATEGORICAL)
    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(
        current_data=Dataset.from_pandas(cur[cols], data_definition=definition),
        reference_data=Dataset.from_pandas(ref[cols], data_definition=definition))
    path.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(path))
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["heldout", "gender"], default="heldout",
                    help="heldout: training vs the competitions the model never "
                         "saw. gender: men's vs women's football, which is the "
                         "sharper question the held-out split confounds.")
    ap.add_argument("--date", default=date.today().isoformat())
    args = ap.parse_args()

    conn = db.connect()
    try:
        df = load(conn)
    finally:
        conn.close()

    ref_mask, cur_mask, (ref_label, cur_label) = split_masks(df, args.split)
    ref, cur = df[ref_mask], df[cur_mask]
    print(f"reference: {len(ref):,} shots ({ref_label})")
    print(f"current:   {len(cur):,} shots ({cur_label})")
    if len(cur) < 100 or len(ref) < 100:
        print("not enough shots on one side to say anything")
        return

    html = OUT_DIR / f"{args.date}-{args.split}.html"
    produced = evidently_html(ref, cur, html)
    shifts = population_shift(ref, cur)

    print(f"\nlargest population shifts ({cur_label} relative to {ref_label}):")
    for row in shifts:
        direction = "higher" if row["cohens_d"] > 0 else "lower"
        print(f"  {row['feature']:20} d = {row['cohens_d']:+.3f}  "
              f"({row['reference_mean']:.2f} -> {row['current_mean']:.2f}, "
              f"{direction})")

    top = shifts[:2]
    lines = [
        f"# Feature drift — {cur_label} vs {ref_label}",
        "",
        f"Generated {args.date} by `monitoring/drift_report.py`. Reference is "
        f"{len(ref):,} shots from the {ref_label}; current is {len(cur):,} from "
        f"the {cur_label}.",
        "",
        "## The two features that move most",
        "",
    ]
    for row in top:
        direction = "higher" if row["cohens_d"] > 0 else "lower"
        lines.append(
            f"- **`{row['feature']}`** is {direction} in {cur_label}: "
            f"{row['reference_mean']:.2f} → {row['current_mean']:.2f} "
            f"(Cohen's d {row['cohens_d']:+.2f}).")
    lines += [
        "",
        "Reported as effect size, not as a drift verdict. Over several thousand "
        "shots a statistical test calls nearly every column drifted, because with "
        "n this large a difference of no consequence is still significant. "
        "Cohen's d answers the question that matters — how far apart are they — "
        "and a |d| under about 0.2 is a difference you would struggle to see.",
        "",
        "## Every feature",
        "",
        "| feature | reference mean | current mean | Cohen's d |",
        "|---|--:|--:|--:|",
    ]
    for row in shifts:
        lines.append(f"| `{row['feature']}` | {row['reference_mean']:.3f} | "
                     f"{row['current_mean']:.3f} | {row['cohens_d']:+.3f} |")
    if produced:
        lines += ["", f"Full Evidently report: [`{html.name}`]({html.name}) — "
                      f"distributions, per-column tests and the drift summary.", ""]
    lines += ["", "Data source: StatsBomb."]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    md = OUT_DIR / f"{args.date}-{args.split}.md"
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT_DIR / f"{args.date}-{args.split}.json").write_text(
        json.dumps({"split": args.split, "date": args.date,
                    "reference": ref_label, "current": cur_label,
                    "n_reference": int(len(ref)), "n_current": int(len(cur)),
                    "shifts": shifts}, indent=2), encoding="utf-8")
    print(f"\nwrote {md}")
    if produced:
        print(f"wrote {html}")


if __name__ == "__main__":
    db.cli(main)()
