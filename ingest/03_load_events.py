"""Phase 1: load cached event JSON into Postgres.

Idempotent. Every insert is an upsert keyed on the StatsBomb id, so re-running
over the same cache changes nothing but the clock. Reads only from
data/raw — run 02_fetch.py first.

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

BATCH = 5000

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


def load_match(cur, path: Path, meta: dict) -> tuple:
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

    for i in range(0, len(rows), BATCH):
        cur.executemany(EVENT_SQL, rows[i:i + BATCH])
    if shots:
        cur.executemany(SHOT_SQL, shots)
    return len(rows), len(shots)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", action="store_true", help="apply sql/001_schema.sql and 002_indexes.sql first")
    ap.add_argument("--limit", type=int, default=None, help="load at most N matches")
    ap.add_argument("--match", type=int, action="append", help="load only these match ids")
    args = ap.parse_args()

    conn = db.connect()
    if args.init:
        for name in ("001_schema.sql", "002_indexes.sql"):
            db.apply_sql_file(conn, REPO_ROOT / "sql" / name)
        print(f"schema applied to {db.database_url().rsplit('@', 1)[-1]}")

    index = match_index()
    paths = sorted((RAW_DIR / "events").glob("*.json"), key=lambda p: int(p.stem))
    if args.match:
        wanted = set(args.match)
        paths = [p for p in paths if int(p.stem) in wanted]
    if args.limit:
        paths = paths[: args.limit]

    if not paths:
        print(f"no cached event files in {RAW_DIR / 'events'} — run ingest/02_fetch.py first")
        return

    t0 = time.time()
    n_ev = n_sh = n_ok = 0
    skipped = []
    with conn.cursor() as cur:
        for i, path in enumerate(paths, 1):
            mid = int(path.stem)
            meta = index.get(mid)
            if meta is None:
                skipped.append(mid)
                continue
            e, s = load_match(cur, path, meta)
            n_ev += e
            n_sh += s
            n_ok += 1
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
    print(f"\nloaded {n_ok} matches in {time.time() - t0:.0f}s")
    cover = f" ({ff / s * 100:.1f}% with freeze frames)" if s else ""
    print(f"db now holds: {m:,} matches, {e:,} events, {s:,} shots{cover}")
    conn.close()


if __name__ == "__main__":
    main()
