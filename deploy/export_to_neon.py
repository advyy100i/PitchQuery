"""Copy the serving subset of the local database into a hosted Postgres.

Streams table by table straight from one connection to the other with COPY, so
nothing is buffered in Python and no dump file touches disk.

What ships is defined by deploy/schema_deploy.sql: the raw event JSONB, the
MiniLM embeddings and the tsvector stay behind. That is 1.45 GB left at home,
and it is the difference between the entire 431-match corpus fitting a 500 MB
free tier and having to cut the demo down to a hundred matches.

Run (target is any Postgres URL — Neon, Supabase, a VM):
  python deploy/export_to_neon.py --target "postgresql://user:pw@host/db?sslmode=require"
  python deploy/export_to_neon.py --target "..." --matches 150   # smaller demo
  python deploy/export_to_neon.py --target "..." --dry-run       # just size it
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import psycopg  # noqa: E402

from core import db  # noqa: E402

HERE = Path(__file__).resolve().parent

# Column lists are explicit. `SELECT *` here would silently start shipping the
# raw JSONB again the moment anyone adds it back to the source schema.
TABLES = [
    ("matches", """match_id, competition_id, season_id, competition, season,
                   match_date, home_team, away_team, home_score, away_score, has_360"""),
    ("possessions", """possession_uid, match_id, possession, team, opponent,
                       competition, season, play_pattern, start_idx, end_idx,
                       n_events, duration_s, start_zone, end_zone, zone_path,
                       token_string, ended_in_shot, xg_sum, ended_in_goal"""),
    ("shots", """event_id, match_id, competition_id, season_id, team, player,
                 x, y, distance, angle, body_part, technique, shot_type,
                 first_time, under_pressure, play_pattern, is_goal, statsbomb_xg,
                 freeze_frame, n_def_in_cone, dist_nearest_def, gk_dist_to_goal,
                 gk_off_line"""),
    ("events", """event_id, match_id, idx, period, minute, second, type,
                  play_pattern, possession, possession_team, team, player,
                  position, x, y, end_x, end_y, under_pressure, duration, token"""),
]

# Only what the API actually queries. The GIN and HNSW indexes are for features
# that do not ship, and every index costs storage on a tier measured in MB.
INDEXES = [
    "CREATE INDEX IF NOT EXISTS events_match_poss ON events (match_id, possession)",
    "CREATE INDEX IF NOT EXISTS possessions_team ON possessions (team)",
    "CREATE INDEX IF NOT EXISTS possessions_pattern ON possessions (play_pattern)",
    "CREATE INDEX IF NOT EXISTS possessions_shot ON possessions (ended_in_shot)",
]


def match_filter(conn, limit):
    """The most recent N matches, or all of them."""
    if not limit:
        return None
    with conn.cursor() as cur:
        cur.execute("SELECT match_id FROM matches ORDER BY match_date DESC NULLS LAST "
                    "LIMIT %s", (limit,))
        return [r[0] for r in cur.fetchall()]


def where_for(table, ids):
    if ids is None:
        return ""
    col = "match_id"
    return f" WHERE {col} = ANY(%s)" if table != "matches" else " WHERE match_id = ANY(%s)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, help="destination Postgres URL")
    ap.add_argument("--matches", type=int, default=None,
                    help="ship only the N most recent matches (default: all)")
    ap.add_argument("--dry-run", action="store_true", help="report sizes, copy nothing")
    args = ap.parse_args()

    src = db.connect()
    ids = match_filter(src, args.matches)
    scope = f"{len(ids)} matches" if ids else "all matches"

    print(f"source: {db.database_url().rsplit('@', 1)[-1]}  ({scope})")
    total = 0
    for table, cols in TABLES:
        with src.cursor() as cur:
            cur.execute(f"SELECT count(*) FROM {table}{where_for(table, ids)}",
                        (ids,) if ids else None)
            n = cur.fetchone()[0]
        print(f"  {table:12} {n:>10,} rows")
        total += n
    print(f"  {'total':12} {total:>10,} rows")

    if args.dry_run:
        print("\ndry run — nothing copied")
        return

    print(f"\ntarget: {args.target.rsplit('@', 1)[-1]}")
    with psycopg.connect(args.target) as dst:
        with dst.cursor() as cur:
            cur.execute((HERE / "schema_deploy.sql").read_text(encoding="utf-8"))
        dst.commit()
        print("  schema applied")

        for table, cols in TABLES:
            t0 = time.time()
            with dst.cursor() as out:
                # Truncate so the script is re-runnable without duplicate keys.
                out.execute(f"TRUNCATE {table} CASCADE")
                flat = " ".join(cols.split())
                query = f"COPY (SELECT {flat} FROM {table}{where_for(table, ids)}) TO STDOUT (FORMAT BINARY)"
                with src.cursor() as inp, \
                        out.copy(f"COPY {table} ({flat}) FROM STDIN (FORMAT BINARY)") as writer:
                    with inp.copy(query, (ids,) if ids else None) as reader:
                        for block in reader:
                            writer.write(block)
            dst.commit()
            print(f"  {table:12} copied in {time.time() - t0:.0f}s")

        with dst.cursor() as cur:
            for stmt in INDEXES:
                cur.execute(stmt)
            cur.execute("ANALYZE")
        dst.commit()
        print("  indexes built")

        with dst.cursor() as cur:
            cur.execute("SELECT pg_size_pretty(pg_database_size(current_database()))")
            size = cur.fetchone()[0]
    print(f"\ndone — hosted database is {size}")
    print("Set DATABASE_URL on the API host to the target URL.")


if __name__ == "__main__":
    main()
