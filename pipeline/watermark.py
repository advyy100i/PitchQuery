"""Phase 2: how far the loader got, per competition/season.

The point of this table is that adding one competition costs one competition,
not 1.6M events. Two rules make that safe:

  1. The fetch step filters on it, so work is skipped before it is done rather
     than after — an incremental load that still downloads and parses every
     match and then upserts it away is not incremental, it is just quiet.

  2. `advance()` takes a cursor, not a connection, so the caller runs it on the
     same transaction as the inserts. If the load dies halfway the watermark
     rolls back with the rows it was describing, and the next run redoes exactly
     the matches that did not land.

`last_match_id` is a high-water mark, which only means anything if matches are
loaded in ascending id order. `ingest/03_load_events.py` sorts by match id for
that reason, and refuses to advance the watermark for an out-of-order
`--match` load.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.config import REPO_ROOT  # noqa: E402

DDL_PATH = REPO_ROOT / "sql" / "003_watermark.sql"


def ensure(conn) -> None:
    """Create the table if it is not there yet. Cheap enough to call every run."""
    with conn.cursor() as cur:
        cur.execute(DDL_PATH.read_text(encoding="utf-8"))
    conn.commit()


def read_all(conn) -> dict:
    """(competition_id, season_id) -> {last_match_id, last_run_at, rows_loaded}."""
    ensure(conn)
    with conn.cursor() as cur:
        cur.execute("SELECT competition_id, season_id, last_match_id, last_run_at, "
                    "rows_loaded FROM ingest_watermark")
        return {(r[0], r[1]): {"last_match_id": r[2], "last_run_at": r[3],
                               "rows_loaded": r[4]} for r in cur.fetchall()}


def since(conn, comps: list = None) -> dict:
    """(competition_id, season_id) -> last_match_id, for ingest/02_fetch.py.

    `comps` is the ['55:43', ...] form the flow is parameterised with; pass it
    to restrict the mapping to the competitions this run cares about.
    """
    marks = {k: v["last_match_id"] for k, v in read_all(conn).items()
             if v["last_match_id"] is not None}
    if comps is None:
        return marks
    wanted = {tuple(int(v) for v in spec.split(":")) for spec in comps}
    return {k: v for k, v in marks.items() if k in wanted}


ADVANCE_SQL = """
INSERT INTO ingest_watermark (competition_id, season_id, last_match_id,
                              last_run_at, rows_loaded)
VALUES (%s, %s, %s, now(), %s)
ON CONFLICT (competition_id, season_id) DO UPDATE SET
  -- GREATEST, not EXCLUDED: re-loading an old match must never rewind the mark
  -- and make the next run redo everything after it.
  last_match_id = GREATEST(ingest_watermark.last_match_id, EXCLUDED.last_match_id),
  last_run_at   = EXCLUDED.last_run_at,
  rows_loaded   = ingest_watermark.rows_loaded + EXCLUDED.rows_loaded
"""


def advance(cur, competition_id: int, season_id: int, last_match_id: int,
            rows_loaded: int) -> None:
    """Move the mark. Takes a CURSOR so this lands in the caller's transaction."""
    cur.execute(ADVANCE_SQL, (competition_id, season_id, last_match_id, rows_loaded))


def reset(conn, comps: list = None) -> int:
    """Forget the marks, so the next run reloads from scratch. Returns rows deleted."""
    ensure(conn)
    with conn.cursor() as cur:
        if comps is None:
            cur.execute("DELETE FROM ingest_watermark")
        else:
            pairs = [tuple(int(v) for v in spec.split(":")) for spec in comps]
            cur.execute("DELETE FROM ingest_watermark WHERE (competition_id, season_id) "
                        "IN (SELECT * FROM unnest(%s::int[], %s::int[]))",
                        ([c for c, _ in pairs], [s for _, s in pairs]))
        n = cur.rowcount
    conn.commit()
    return n


def report(conn) -> str:
    """One line per competition, for the flow log."""
    rows = read_all(conn)
    if not rows:
        return "watermark: empty (next run is a full load)"
    out = ["watermark:"]
    for (cid, sid), v in sorted(rows.items()):
        out.append(f"  {cid}:{sid}  last match {v['last_match_id']}  "
                   f"{v['rows_loaded']:,} rows  {v['last_run_at']:%Y-%m-%d %H:%M}")
    return "\n".join(out)


if __name__ == "__main__":
    from core import db

    conn = db.connect()
    print(report(conn))
    conn.close()
