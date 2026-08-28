"""Phase 1: load cached event JSON into Postgres.

Idempotent. Every insert is an upsert keyed on the StatsBomb id, so re-running
over the same cache changes nothing but the clock. Reads only from
data/raw — run 02_fetch.py first.

Every batch is checked against the Pandera contracts in pipeline/contracts.py
before it is written (Phase 4), and the ingest watermark is advanced inside the
same transaction as the insert (Phase 2), so a load that dies halfway leaves the
mark on the last match that actually committed.

Run:
  docker compose up -d db
  python ingest/03_load_events.py --init            # create schema, then load all cached matches
  python ingest/03_load_events.py --limit 5         # smoke test
  python ingest/03_load_events.py --match 3869118   # one match
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from psycopg.types.json import Jsonb  # noqa: E402

from core import db  # noqa: E402
from core.config import RAW_DIR, REPO_ROOT  # noqa: E402
from core.features import shot_row  # noqa: E402
from core.zones import token as grammar_token  # noqa: E402
from pipeline import watermark  # noqa: E402
from pipeline.contracts import EventSchema, ShotSchema, check, frame  # noqa: E402

BATCH = 5000

# The INSERTs below are positional, so the contracts have to be told the column
# order once. Keeping the two lists next to their statements is what stops a
# column being added to one and not the other.
EVENT_COLS = ["event_id", "match_id", "idx", "period", "minute", "second", "type",
              "play_pattern", "possession", "possession_team", "team", "player",
              "position", "x", "y", "end_x", "end_y", "under_pressure", "duration",
              "token", "raw"]

SHOT_COLS = ["event_id", "match_id", "competition_id", "season_id", "team", "player",
             "x", "y", "distance", "angle", "body_part", "technique", "shot_type",
             "first_time", "under_pressure", "play_pattern", "is_goal", "statsbomb_xg",
             "freeze_frame", "n_def_in_cone", "dist_nearest_def", "gk_dist_to_goal",
             "gk_off_line"]

MATCH_SQL = """
INSERT INTO matches (match_id, competition_id, season_id, competition, season,
                     match_date, home_team, away_team, home_score, away_score, has_360)
VALUES (%(match_id)s, %(competition_id)s, %(season_id)s, %(competition)s, %(season)s,
        %(match_date)s, %(home_team)s, %(away_team)s, %(home_score)s, %(away_score)s, %(has_360)s)
ON CONFLICT (match_id) DO UPDATE SET
  competition_id = EXCLUDED.competition_id, season_id = EXCLUDED.season_id,
  competition = EXCLUDED.competition, season = EXCLUDED.season,
  match_date = EXCLUDED.match_date, home_team = EXCLUDED.home_team,
  away_team = EXCLUDED.away_team, home_score = EXCLUDED.home_score,
  away_score = EXCLUDED.away_score, has_360 = EXCLUDED.has_360
"""

EVENT_SQL = """
INSERT INTO events (event_id, match_id, idx, period, minute, second, type,
                    play_pattern, possession, possession_team, team, player,
                    position, x, y, end_x, end_y, under_pressure, duration, token, raw)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (event_id) DO UPDATE SET
  idx = EXCLUDED.idx, period = EXCLUDED.period, minute = EXCLUDED.minute,
  second = EXCLUDED.second, type = EXCLUDED.type, play_pattern = EXCLUDED.play_pattern,
  possession = EXCLUDED.possession, possession_team = EXCLUDED.possession_team,
  team = EXCLUDED.team, player = EXCLUDED.player, position = EXCLUDED.position,
  x = EXCLUDED.x, y = EXCLUDED.y, end_x = EXCLUDED.end_x, end_y = EXCLUDED.end_y,
  under_pressure = EXCLUDED.under_pressure, duration = EXCLUDED.duration,
  token = EXCLUDED.token, raw = EXCLUDED.raw
"""

SHOT_SQL = """
INSERT INTO shots (event_id, match_id, competition_id, season_id, team, player,
                   x, y, distance, angle, body_part, technique, shot_type,
                   first_time, under_pressure, play_pattern, is_goal,
                   statsbomb_xg, freeze_frame, n_def_in_cone, dist_nearest_def,
                   gk_dist_to_goal, gk_off_line)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
        %s, %s, %s, %s, %s)
ON CONFLICT (event_id) DO UPDATE SET
  distance = EXCLUDED.distance, angle = EXCLUDED.angle,
  body_part = EXCLUDED.body_part, technique = EXCLUDED.technique,
  shot_type = EXCLUDED.shot_type, first_time = EXCLUDED.first_time,
  under_pressure = EXCLUDED.under_pressure, play_pattern = EXCLUDED.play_pattern,
  is_goal = EXCLUDED.is_goal, statsbomb_xg = EXCLUDED.statsbomb_xg,
  freeze_frame = EXCLUDED.freeze_frame, n_def_in_cone = EXCLUDED.n_def_in_cone,
  dist_nearest_def = EXCLUDED.dist_nearest_def,
  gk_dist_to_goal = EXCLUDED.gk_dist_to_goal, gk_off_line = EXCLUDED.gk_off_line
"""


def match_index() -> dict:
    """match_id -> row for the `matches` table, from every cached matches file."""
    out = {}
    for path in sorted((RAW_DIR / "matches").glob("*/*.json")):
        competition_id = int(path.parent.name)
        season_id = int(path.stem)
        for m in json.loads(path.read_text(encoding="utf-8")):
            out[m["match_id"]] = {
                "match_id": m["match_id"],
                "competition_id": competition_id,
                "season_id": season_id,
                "competition": (m.get("competition") or {}).get("competition_name"),
                "season": (m.get("season") or {}).get("season_name"),
                "match_date": m.get("match_date"),
                "home_team": (m.get("home_team") or {}).get("home_team_name"),
                "away_team": (m.get("away_team") or {}).get("away_team_name"),
                "home_score": m.get("home_score"),
                "away_score": m.get("away_score"),
                "has_360": m.get("match_status_360") == "available",
            }
    return out


def end_location(ev: dict, etype: str):
    """Pass/carry/shot end coordinates. Shot end_location is 3D — drop z."""
    key = {"Pass": "pass", "Carry": "carry", "Shot": "shot"}.get(etype)
    if not key:
        return None, None
    loc = (ev.get(key) or {}).get("end_location") or []
    if len(loc) < 2:
        return None, None
    return float(loc[0]), float(loc[1])


def event_tuple(ev: dict, match_id: int) -> tuple:
    etype = (ev.get("type") or {}).get("name")
    loc = ev.get("location") or []
    x = float(loc[0]) if len(loc) >= 2 else None
    y = float(loc[1]) if len(loc) >= 2 else None
    end_x, end_y = end_location(ev, etype)
    return (
        ev["id"], match_id, ev.get("index"), ev.get("period"),
        ev.get("minute"), ev.get("second"), etype,
        (ev.get("play_pattern") or {}).get("name"),
        ev.get("possession"),
        (ev.get("possession_team") or {}).get("name"),
        (ev.get("team") or {}).get("name"),
        (ev.get("player") or {}).get("name"),
        (ev.get("position") or {}).get("name"),
        x, y, end_x, end_y,
        bool(ev.get("under_pressure", False)),
        ev.get("duration"),
        grammar_token(ev),
        Jsonb(ev),
    )


def load_match(cur, path: Path, meta: dict, *, validate: bool = True) -> tuple:
    """Returns (n_events, n_shots)."""
    match_id = int(path.stem)
    events = json.loads(path.read_text(encoding="utf-8"))

    cur.execute(MATCH_SQL, meta)

    rows, shots = [], []
    for ev in events:
        rows.append(event_tuple(ev, match_id))
        sr = shot_row(ev)
        if sr:
            shots.append((
                sr["event_id"], match_id, meta["competition_id"], meta["season_id"],
                sr["team"], sr["player"], sr["x"], sr["y"], sr["distance"], sr["angle"],
                sr["body_part"], sr["technique"], sr["shot_type"], sr["first_time"],
                sr["under_pressure"], sr["play_pattern"], sr["is_goal"],
                sr["statsbomb_xg"],
                Jsonb(sr["freeze_frame"]) if sr["freeze_frame"] is not None else None,
                sr["n_def_in_cone"], sr["dist_nearest_def"],
                sr["gk_dist_to_goal"], sr["gk_off_line"],
            ))

    # Validate before the write, not after. A contract checked on the way out of
    # the database tells you the corpus is already wrong; checked here, the
    # transaction rolls back and the watermark does not move.
    if validate:
        check(EventSchema, frame(rows, EVENT_COLS), where=f"match {match_id}")
        if shots:
            check(ShotSchema, frame(shots, SHOT_COLS), where=f"match {match_id}")

    for i in range(0, len(rows), BATCH):
        cur.executemany(EVENT_SQL, rows[i:i + BATCH])
    if shots:
        cur.executemany(SHOT_SQL, shots)
    return len(rows), len(shots)


def main(match_ids: list = None, *, init: bool = False, limit: int = None,
         comps: list = None, validate: bool = True, advance: bool = True) -> dict:
    """Load cached event JSON into Postgres.

    `match_ids` is what ingest/02_fetch.py returns — pass it and only those
    matches are loaded. Left as None, every cached file is loaded, which is the
    original behaviour and what a bare `python ingest/03_load_events.py` does.

    `advance` controls the Phase 2 watermark. It is only meaningful when matches
    arrive in ascending id order, which is why the CLI turns it off for an
    explicit `--match` load: a high-water mark set from an out-of-order load
    would claim everything below it is present when it is not.

    Returns {"rows": events loaded, "matches", "shots", "match_ids"}.
    """
    conn = db.connect()
    if init:
        for name in ("001_schema.sql", "002_indexes.sql", "003_watermark.sql"):
            db.apply_sql_file(conn, REPO_ROOT / "sql" / name)
        print(f"schema applied to {db.database_url().rsplit('@', 1)[-1]}")
    watermark.ensure(conn)

    index = match_index()
    paths = sorted((RAW_DIR / "events").glob("*.json"), key=lambda p: int(p.stem))
    if match_ids is not None:
        wanted = set(match_ids)
        paths = [p for p in paths if int(p.stem) in wanted]
    if comps:
        pairs = {tuple(int(v) for v in spec.split(":")) for spec in comps}
        paths = [p for p in paths
                 if (m := index.get(int(p.stem))) is not None
                 and (m["competition_id"], m["season_id"]) in pairs]
    if limit:
        paths = paths[: limit]

    if not paths:
        print(f"nothing to load from {RAW_DIR / 'events'} "
              f"— everything requested is already past the watermark, or "
              f"ingest/02_fetch.py has not run")
        conn.close()
        return {"rows": 0, "matches": 0, "shots": 0, "match_ids": []}

    t0 = time.time()
    n_ev = n_sh = n_ok = 0
    loaded, skipped = [], []
    with conn.cursor() as cur:
        for i, path in enumerate(paths, 1):
            mid = int(path.stem)
            meta = index.get(mid)
            if meta is None:
                skipped.append(mid)
                continue
            e, s = load_match(cur, path, meta, validate=validate)
            n_ev += e
            n_sh += s
            n_ok += 1
            loaded.append(mid)
            if advance:
                # Same cursor, therefore the same transaction as the inserts
                # above. The commit below either lands both or neither.
                watermark.advance(cur, meta["competition_id"], meta["season_id"], mid, e)
            conn.commit()
            if i % 25 == 0 or i == len(paths):
                rate = i / max(time.time() - t0, 1e-9)
                print(f"  {i}/{len(paths)} matches  {n_ev:,} events  {n_sh:,} shots  "
                      f"({rate:.1f} match/s)")

    if skipped:
        print(f"skipped {len(skipped)} matches with no entry in raw/matches: {skipped[:5]}")

    with conn.cursor() as cur:
        cur.execute("SELECT (SELECT count(*) FROM matches), (SELECT count(*) FROM events), "
                    "(SELECT count(*) FROM shots), (SELECT count(*) FROM shots WHERE freeze_frame IS NOT NULL)")
        m, e, s, ff = cur.fetchone()
    print()
    print(f"loaded {n_ok} matches in {time.time() - t0:.0f}s")
    cover = f" ({ff / s * 100:.1f}% with freeze frames)" if s else ""
    print(f"db now holds: {m:,} matches, {e:,} events, {s:,} shots{cover}")
    conn.close()
    return {"rows": n_ev, "matches": n_ok, "shots": n_sh, "match_ids": loaded}


@db.cli
def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", action="store_true",
                    help="apply sql/001_schema.sql, 002_indexes.sql and 003_watermark.sql first")
    ap.add_argument("--limit", type=int, default=None, help="load at most N matches")
    ap.add_argument("--match", type=int, action="append", help="load only these match ids")
    ap.add_argument("--no-validate", action="store_true",
                    help="skip the Pandera contracts (they cost a few seconds over the corpus)")
    args = ap.parse_args()
    main(args.match, init=args.init, limit=args.limit,
         validate=not args.no_validate,
         # An explicit match list is not necessarily in order and is usually a
         # one-off repair, so it must not claim ground it has not covered.
         advance=args.match is None)


if __name__ == "__main__":
    cli()
