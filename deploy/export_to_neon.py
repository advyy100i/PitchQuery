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
  python deploy/export_to_neon.py --target "..." --dbt            # + the dbt layers

`--dbt` builds the three models that can exist on a copy without the raw event
JSONB — see build_dbt() for which two cannot, and why.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import unquote, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import psycopg  # noqa: E402

from core import db  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent

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
    ("ingest_watermark", """competition_id, season_id, last_match_id, last_run_at,
                            rows_loaded"""),
]

# Tables with no match_id, so the --matches filter cannot apply to them. The
# watermark is also the one table --matches makes untrue: it records how far the
# loader got locally, and a copy of the 150 most recent matches did not get
# there. Shipped whole or not at all — see main().
NO_MATCH_FILTER = {"ingest_watermark"}

# Only what the API actually queries. The GIN and HNSW indexes are for features
# that do not ship, and every index costs storage on a tier measured in MB.
INDEXES = [
    "CREATE INDEX IF NOT EXISTS events_match_poss ON events (match_id, possession)",
    "CREATE INDEX IF NOT EXISTS possessions_team ON possessions (team)",
    "CREATE INDEX IF NOT EXISTS possessions_pattern ON possessions (play_pattern)",
    "CREATE INDEX IF NOT EXISTS possessions_shot ON possessions (ended_in_shot)",
]


def build_dbt(target: str) -> int:
    """Build the dbt models the hosted copy can actually support.

    Three of the five can. stg_shots is a view over columns that ship,
    stg_freeze_frames unnests shots.freeze_frame which also ships, and
    mart_xg_features is built from those two — about 20 MB all told, two small
    tables and a view, against a tier measured in hundreds.

    Two of them cannot, and no configuration changes that. stg_events unpacks
    pass, shot, duel and dribble qualifiers out of `events.raw`, and
    mart_team_possessions counts crosses and through balls from stg_events. That
    JSONB is exactly what this script leaves behind: shipping it is the 3.3 GB
    that takes the deployed database from 424 MB to over 3.7 GB, against a
    500 MB free tier. /ops reports those two as unavailable there rather than as
    not built, because "run dbt" is not the fix and saying so wastes an
    afternoon.

    Which models are hostable is a tag on the models themselves, not a list
    here, so a new model declares it next to the SQL that decides it.
    """
    u = urlparse(target)
    env = {
        **os.environ,
        "PGHOST": u.hostname or "",
        "PGPORT": str(u.port or 5432),
        "PGUSER": unquote(u.username or ""),
        "PGPASSWORD": unquote(u.password or ""),
        "PGDATABASE": (u.path or "/").lstrip("/"),
        # Neon closes the connection without TLS. `prefer`, the libpq default
        # the profile falls back to, would silently accept a plaintext target.
        "PGSSLMODE": "require",
    }
    cmd = ["dbt", "build", "--select", "tag:hosted", "--profiles-dir", "."]
    print(f"\n  dbt: {' '.join(cmd)}  (against {u.hostname})")
    try:
        return subprocess.run(cmd, cwd=REPO_ROOT / "warehouse", env=env).returncode
    except FileNotFoundError:
        print("  dbt is not installed — pip install -r requirements-pipeline.txt")
        return 1


def match_filter(conn, limit):
    """The most recent N matches, or all of them."""
    if not limit:
        return None
    with conn.cursor() as cur:
        cur.execute("SELECT match_id FROM matches ORDER BY match_date DESC NULLS LAST "
                    "LIMIT %s", (limit,))
        return [r[0] for r in cur.fetchall()]


def where_for(table, ids):
    if ids is None or table in NO_MATCH_FILTER:
        return ""
    return " WHERE match_id = ANY(%s)"


def params_for(table, ids):
    """psycopg rejects parameters for a statement that has no placeholder, so
    the filter and its argument have to be decided together."""
    return (ids,) if where_for(table, ids) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, help="destination Postgres URL")
    ap.add_argument("--matches", type=int, default=None,
                    help="ship only the N most recent matches (default: all)")
    ap.add_argument("--dry-run", action="store_true", help="report sizes, copy nothing")
    ap.add_argument("--only", metavar="T1,T2",
                    help="copy only these tables. The schema, the indexes and "
                         "--dbt still run. For topping up a database that is "
                         "already correct — a full run TRUNCATEs every table, "
                         "and re-copying 1.6M event rows to add two watermark "
                         "rows means a live demo serves an empty corpus for the "
                         "several minutes it takes")
    ap.add_argument("--dbt", action="store_true",
                    help="also build the dbt models the hosted copy can support "
                         "(stg_shots, stg_freeze_frames, mart_xg_features)")
    args = ap.parse_args()

    src = db.connect()
    ids = match_filter(src, args.matches)
    scope = f"{len(ids)} matches" if ids else "all matches"

    tables = TABLES
    if args.only:
        want = {t.strip() for t in args.only.split(",") if t.strip()}
        unknown = want - {t[0] for t in TABLES}
        if unknown:
            sys.exit(f"--only: no such table: {', '.join(sorted(unknown))}")
        tables = [t for t in TABLES if t[0] in want]
    if ids:
        # The watermark says the loader reached match X. Ship it beside a subset
        # that stops short of X and the Ingest panel reports rows that are not
        # in the database it is describing — which is worse than the panel
        # saying it has nothing to report.
        tables = [t for t in tables if t[0] not in NO_MATCH_FILTER]
        print("note: --matches is set, so ingest_watermark is not shipped — it "
              "would describe a fuller load than this copy holds")

    print(f"source: {db.database_url().rsplit('@', 1)[-1]}  ({scope})")
    total = 0
    for table, cols in tables:
        with src.cursor() as cur:
            cur.execute(f"SELECT count(*) FROM {table}{where_for(table, ids)}",
                        params_for(table, ids))
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

        for table, cols in tables:
            t0 = time.time()
            with dst.cursor() as out:
                # Truncate so the script is re-runnable without duplicate keys.
                out.execute(f"TRUNCATE {table} CASCADE")
                flat = " ".join(cols.split())
                query = f"COPY (SELECT {flat} FROM {table}{where_for(table, ids)}) TO STDOUT (FORMAT BINARY)"
                with src.cursor() as inp, \
                        out.copy(f"COPY {table} ({flat}) FROM STDIN (FORMAT BINARY)") as writer:
                    with inp.copy(query, params_for(table, ids)) as reader:
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
    if args.dbt:
        if build_dbt(args.target) != 0:
            print("  dbt build failed — the bronze tables are copied either way")
        else:
            with psycopg.connect(args.target) as dst, dst.cursor() as cur:
                cur.execute("SELECT pg_size_pretty(pg_database_size(current_database()))")
                print(f"  with the dbt layers: {cur.fetchone()[0]}")
    else:
        print("Add --dbt to build stg_shots, stg_freeze_frames and "
              "mart_xg_features there too.")

    print("Set DATABASE_URL on the API host to the target URL.")


if __name__ == "__main__":
    main()
