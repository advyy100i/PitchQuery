"""Phase 8: read the query log, and turn the bad rankings in it into work.

Two jobs.

`candidates()` finds searches where the result someone actually opened was at
rank 5 or below. That is the definition of a ranking the system got wrong: the
answer was there and was not near the top. Those queries are written to
eval/candidates.json to be graded by hand and folded into eval/queries.yaml.
It is how the eval set grows past thirty without anybody inventing a query, and
it is the honest fix for the confidence problem docs/ranker_eval.md reports.

`vocabulary()` ranks the words `core/planner.py` could not place. A parser's
blind spots written down by its users, in frequency order.

Neither of these writes to eval/queries.yaml. A query is only worth adding once
somebody has written a rubric for it, and a rubric is a judgement — automating
it would produce an eval set that agrees with the ranker by construction.

Run:
  python -m pipeline.telemetry                  # print both reports
  python -m pipeline.telemetry --write          # ...and write eval/candidates.json
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from core import db  # noqa: E402
from core.config import REPO_ROOT as ROOT  # noqa: E402

CANDIDATES = ROOT / "eval" / "candidates.json"
DDL = ROOT / "sql" / "004_telemetry.sql"

# A click at rank 5 or below means the result was found despite the ranking
# rather than because of it. Rank is 1-based, as shown to the user.
DEEP_CLICK_RANK = 5


def ensure(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(DDL.read_text(encoding="utf-8"))
    conn.commit()


DEEP_CLICKS_SQL = """
SELECT s.id, s.ts, s.query_text, s.sequence_hint, s.parsed_filters, s.ranker,
       c.possession_uid, c.rank, s.top_uids
FROM click_log c
JOIN search_log s ON s.id = c.search_id
WHERE c.rank >= %s
  AND s.query_text IS NOT NULL
ORDER BY c.rank DESC, s.ts DESC
LIMIT %s
"""


def candidates(conn, limit: int = 200) -> list:
    """Searches whose opened result sat at rank >= DEEP_CLICK_RANK."""
    ensure(conn)
    with conn.cursor() as cur:
        cur.execute(DEEP_CLICKS_SQL, (DEEP_CLICK_RANK, limit))
        cols = [d.name for d in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]

    out = []
    for r in rows:
        out.append({
            "search_id": r["id"],
            "ts": r["ts"].isoformat(),
            "text": r["query_text"],
            "sequence_hint": r["sequence_hint"],
            "filters": r["parsed_filters"],
            "ranker": r["ranker"],
            # What to grade: the possession the user chose, and where the
            # ranking put it.
            "clicked_uid": r["possession_uid"],
            "clicked_rank": r["rank"],
            "shown": (r["top_uids"] or [])[:10],
            # Filled in by hand. Left explicitly null rather than omitted so
            # that the file reads as a work queue rather than a result.
            "rubric": None,
            "graded": False,
        })
    return out


def vocabulary(conn, limit: int = 50) -> list:
    """[(word, times seen)] for words the planner could not place."""
    ensure(conn)
    with conn.cursor() as cur:
        cur.execute("SELECT unparsed_words FROM search_log "
                    "WHERE unparsed_words IS NOT NULL")
        counter = Counter(w for (words,) in cur for w in (words or []))
    return counter.most_common(limit)


def volume(conn, days: int = 30) -> list:
    """[(day, searches, clicks, p95 latency)] — what the dashboard plots."""
    ensure(conn)
    with conn.cursor() as cur:
        cur.execute("""
            SELECT date_trunc('day', s.ts)::date AS day,
                   count(*) AS searches,
                   count(c.id) AS clicks,
                   percentile_cont(0.95) WITHIN GROUP (ORDER BY s.latency_ms) AS p95
            FROM search_log s
            LEFT JOIN click_log c ON c.search_id = s.id
            WHERE s.ts > now() - make_interval(days => %s)
            GROUP BY 1 ORDER BY 1
        """, (days,))
        return cur.fetchall()


def main(write: bool = False, limit: int = 200) -> dict:
    conn = db.connect()
    try:
        cands = candidates(conn, limit)
        vocab = vocabulary(conn)
        vol = volume(conn)
    finally:
        conn.close()

    print(f"{len(cands)} searches where the opened result sat at rank "
          f">= {DEEP_CLICK_RANK} — these are the rankings that were wrong")
    for c in cands[:10]:
        print(f"  rank {c['clicked_rank']:>3}  {c['text'][:60]!r} -> {c['clicked_uid']}")

    print(f"\nvocabulary the parser did not recognise ({len(vocab)} distinct):")
    for word, n in vocab[:15]:
        print(f"  {n:>5}  {word}")
    if not vocab:
        print("  (nothing yet — the log is empty, or the parser understood everything)")

    print(f"\n{len(vol)} days with traffic")

    if write:
        CANDIDATES.parent.mkdir(parents=True, exist_ok=True)
        CANDIDATES.write_text(json.dumps({
            "generated_from": "search_log + click_log",
            "deep_click_rank": DEEP_CLICK_RANK,
            "note": ("Each entry is a query whose useful result was ranked low. "
                     "Write a rubric, add it to eval/queries.yaml and "
                     "eval/judge.py, then rerun models/train_ranker.py. "
                     "Nothing here is added to the eval set automatically — a "
                     "rubric is a judgement, and an eval set generated from the "
                     "ranker's own output agrees with it by construction."),
            "unparsed_vocabulary": [{"word": w, "n": n} for w, n in vocab],
            "candidates": cands,
        }, indent=2), encoding="utf-8")
        print(f"\nwrote {CANDIDATES}")

    return {"candidates": len(cands), "vocabulary": len(vocab), "days": len(vol)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="write eval/candidates.json")
    ap.add_argument("--limit", type=int, default=200)
    args = ap.parse_args()
    db.cli(main)(write=args.write, limit=args.limit)
