"""Phase 1 sanity check: plot one loaded match's shots straight from Postgres.

If the dots cluster in front of one goal and the big markers (goals) sit closer
in than the small ones, the coordinate convention and the geometry features
survived the round trip. If they scatter over both halves, A2 is wrong and
everything downstream is corrupt.

Run:
  python ingest/_shotmap.py                 # busiest loaded match
  python ingest/_shotmap.py --match 3869118
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mplsoccer import Pitch  # noqa: E402

from core import db  # noqa: E402
from core.config import DOCS_DIR  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", type=int, default=None)
    ap.add_argument("--out", default=str(DOCS_DIR / "shotmap_check.png"))
    args = ap.parse_args()

    conn = db.connect()
    with conn.cursor() as cur:
        if args.match is None:
            cur.execute("SELECT match_id FROM shots GROUP BY match_id "
                        "ORDER BY count(*) DESC LIMIT 1")
            row = cur.fetchone()
            if not row:
                print("no shots in the database — run ingest/03_load_events.py first")
                return
            args.match = row[0]

        cur.execute("SELECT competition, season, home_team, away_team, home_score, away_score "
                    "FROM matches WHERE match_id = %s", (args.match,))
        meta = cur.fetchone()
        cur.execute("SELECT x, y, is_goal, statsbomb_xg, team, shot_type "
                    "FROM shots WHERE match_id = %s", (args.match,))
        shots = cur.fetchall()
    conn.close()

    if not shots:
        print(f"match {args.match} has no shots loaded")
        return

    teams = sorted({s[4] for s in shots})
    colours = {teams[0]: "#e4572e", teams[-1]: "#2e86ab"}

    pitch = Pitch(pitch_type="statsbomb", line_color="#555", pitch_color="#f7f7f5")
    fig, ax = pitch.draw(figsize=(11, 7))
    for x, y, is_goal, xg, team, stype in shots:
        pitch.scatter(x, y, ax=ax, s=120 + 1600 * (xg or 0),
                      c=colours.get(team, "#888"),
                      marker="*" if is_goal else "o",
                      edgecolors="black", linewidth=0.8,
                      alpha=0.95 if is_goal else 0.65, zorder=3)

    title = (f"{meta[2]} {meta[4]}-{meta[5]} {meta[3]}  —  {meta[0]} {meta[1]}"
             if meta else f"match {args.match}")
    ax.set_title(f"{title}\n{len(shots)} shots, marker area ∝ StatsBomb xG, ★ = goal",
                 fontsize=11)
    fig.text(0.5, 0.02, "Data source: StatsBomb", ha="center", fontsize=8, color="#666")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"match {args.match}: {len(shots)} shots -> {args.out}")
    print("check: every shot should be in the attacking half (x > 60) of the same goal.")
    print(f"  x range {min(s[0] for s in shots):.1f} - {max(s[0] for s in shots):.1f}")


if __name__ == "__main__":
    main()
