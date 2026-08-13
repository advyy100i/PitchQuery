"""Phase 2: turn events into possessions with a token string each (plan §5).

Reads the cached event JSON from disk (much cheaper than pulling 1.5M JSONB
blobs back out of Postgres) and match metadata from the `matches` table, so run
03_load_events.py first.

Two rules from the plan that this enforces:
  1. Only the attacking team's own events go in the string. An event where
     team != possession_team is the defence intervening, and interleaving those
     destroys n-gram similarity.
  2. Possessions with fewer than 3 tokens are dropped — throw-in noise that
     would otherwise flood every result set.

Run:
  python ingest/04_build_possessions.py --limit 5 --show 3   # eyeball first
  python ingest/04_build_possessions.py
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import db  # noqa: E402
from core.config import RAW_DIR  # noqa: E402
from core.zones import MIN_EVENTS, token, tsv_string  # noqa: E402

INSERT_SQL = """
INSERT INTO possessions (possession_uid, match_id, possession, team, opponent,
    competition, season, play_pattern, start_idx, end_idx, n_events, duration_s,
    start_zone, end_zone, zone_path, token_string, token_tsv, ended_in_shot,
    xg_sum, ended_in_goal)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
        to_tsvector('simple', %s), %s, %s, %s)
ON CONFLICT (possession_uid) DO UPDATE SET
  team = EXCLUDED.team, opponent = EXCLUDED.opponent,
  competition = EXCLUDED.competition, season = EXCLUDED.season,
  play_pattern = EXCLUDED.play_pattern, start_idx = EXCLUDED.start_idx,
  end_idx = EXCLUDED.end_idx, n_events = EXCLUDED.n_events,
  duration_s = EXCLUDED.duration_s, start_zone = EXCLUDED.start_zone,
  end_zone = EXCLUDED.end_zone, zone_path = EXCLUDED.zone_path,
  token_string = EXCLUDED.token_string, token_tsv = EXCLUDED.token_tsv,
  ended_in_shot = EXCLUDED.ended_in_shot, xg_sum = EXCLUDED.xg_sum,
  ended_in_goal = EXCLUDED.ended_in_goal,
  embedding = NULL          -- the string changed, so the old vector is stale
"""


def match_meta(conn) -> dict:
    with conn.cursor() as cur:
        cur.execute("SELECT match_id, competition, season, home_team, away_team FROM matches")
        return {r[0]: r[1:] for r in cur.fetchall()}


def secs(ev: dict) -> float:
    return (ev.get("minute") or 0) * 60 + (ev.get("second") or 0)


def build_match(match_id: int, events: list, meta: tuple) -> list:
    """One row per possession that survives the >= MIN_EVENTS filter."""
    competition, season, home, away = meta

    groups: dict = {}
    for ev in sorted(events, key=lambda e: e.get("index", 0)):   # index, never timestamp
        pid = ev.get("possession")
        if pid is None:
            continue
        groups.setdefault(pid, []).append(ev)

    rows = []
    for pid, evs in groups.items():
        pteam = (evs[0].get("possession_team") or {}).get("name")
        if not pteam:
            continue
        own = [e for e in evs if (e.get("team") or {}).get("name") == pteam]

        toks, zones = [], []
        for e in own:
            t = token(e)
            if t:
                toks.append(t)
                zones.append(t.split("@", 1)[1].rstrip("+>^"))
        if len(toks) < MIN_EVENTS:
            continue

        shots = [e for e in own if (e.get("type") or {}).get("name") == "Shot"]
        xg = sum((e.get("shot") or {}).get("statsbomb_xg") or 0.0 for e in shots)
        goal = any(((e.get("shot") or {}).get("outcome") or {}).get("name") == "Goal"
                   for e in shots)

        rows.append((
            f"{match_id}:{pid}", match_id, pid,
            pteam, away if pteam == home else home,
            competition, season,
            (evs[0].get("play_pattern") or {}).get("name"),
            evs[0].get("index"), evs[-1].get("index"), len(toks),
            max(secs(evs[-1]) - secs(evs[0]), 0.0),
            zones[0], zones[-1], " ".join(zones), " ".join(toks),
            tsv_string(toks),
            bool(shots), float(xg), goal,
        ))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="process at most N matches")
    ap.add_argument("--show", type=int, default=0,
                    help="print this many goal-ending possessions for the eyeball test")
    ap.add_argument("--dry-run", action="store_true", help="build but do not write")
    args = ap.parse_args()

    conn = db.connect()
    meta = match_meta(conn)
    if not meta:
        print("no matches in the database — run ingest/03_load_events.py first")
        return

    paths = [p for p in sorted((RAW_DIR / "events").glob("*.json"), key=lambda p: int(p.stem))
             if int(p.stem) in meta]
    if args.limit:
        paths = paths[: args.limit]

    t0 = time.time()
    total = shown = 0
    with conn.cursor() as cur:
        for i, path in enumerate(paths, 1):
            mid = int(path.stem)
            events = json.loads(path.read_text(encoding="utf-8"))
            rows = build_match(mid, events, meta[mid])
            total += len(rows)

            if shown < args.show:
                for r in rows:
                    if r[19] and shown < args.show:      # ended_in_goal
                        print(f"\n{r[0]}  {r[3]} vs {r[4]}  ({r[7]}, {r[11]:.0f}s, xg {r[18]:.2f})")
                        print(f"  {r[15]}")
                        shown += 1

            if not args.dry_run:
                cur.executemany(INSERT_SQL, rows)
                conn.commit()
            if i % 50 == 0 or i == len(paths):
                print(f"  {i}/{len(paths)} matches  {total:,} possessions  "
                      f"({i / max(time.time() - t0, 1e-9):.1f} match/s)")

    if args.dry_run:
        print(f"\ndry run — built {total:,} possessions from {len(paths)} matches, wrote nothing")
        return

    with conn.cursor() as cur:
        cur.execute("SELECT count(*), avg(n_events), avg(duration_s), "
                    "avg(ended_in_shot::int), avg(ended_in_goal::int) FROM possessions")
        n, avg_ev, avg_dur, p_shot, p_goal = cur.fetchone()
    print(f"\nbuilt {total:,} possessions in {time.time() - t0:.0f}s")
    print(f"db now holds {n:,} possessions — mean {avg_ev:.1f} tokens, {avg_dur:.1f}s, "
          f"{p_shot * 100:.1f}% end in a shot, {p_goal * 100:.2f}% in a goal")
    conn.close()


if __name__ == "__main__":
    main()
