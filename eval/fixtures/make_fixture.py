"""Build eval/fixtures/corpus.sql.gz — the corpus CI measures against.

CI has no database and it is never going to have 1.6M events. So the metric gate
runs against a committed sample, and the honest thing is to say so plainly: the
numbers in a PR comment are fixture numbers, not corpus numbers. That is a
strength rather than a compromise — the gate is deterministic, a run costs
seconds, and a regression shows up as a change in the same measurement rather
than as noise from a different slice of football.

How the sample is chosen matters as much as its size, and both had to be
measured rather than guessed — see the note on PER_QUERY below for what the
first two attempts got wrong. A uniform random sample would leave most of the 30
eval queries with nothing relevant to find and every P@5 would sit at zero, so
the relevant rows are chosen by the RUBRIC — `eval/judge.py`, the same predicate the
scorer uses — and never by what the current ranker returns. Sampling on ranker
output would bake today's ranking into the fixture and the gate would then be
measuring whether a change reproduces today's ranking, which is not the same
question as whether it is better.

Shots come from the two held-out competitions in full, because eval/score_xg.py
scores the shipped artefact out of sample and a fixture that sampled those would
make the log-loss depend on which shots happened to be picked.

Run:
  python eval/fixtures/make_fixture.py                 # rebuild the dump
  python eval/fixtures/make_fixture.py --target 20000
  python eval/fixtures/make_fixture.py --load "postgresql://..."   # load it
"""
import argparse
import gzip
import random
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import yaml  # noqa: E402

from core import db  # noqa: E402
from core.config import REPO_ROOT  # noqa: E402
from eval.judge import judge  # noqa: E402
from core.retrieval import Filters  # noqa: E402

HERE = Path(__file__).resolve().parent
FIXTURE = HERE / "corpus.sql.gz"
SCHEMA_FILES = ("001_schema.sql", "002_indexes.sql", "003_watermark.sql")

# The xG artefact holds these out, and eval/score_xg.py only scores them. Listed
# here rather than read from the artefact so that rebuilding the fixture does
# not silently change scope when a model is retrained with a different split.
HELDOUT = [(43, 106), (72, 107)]

# Per query, and the size of the pool they sit in. Both were tuned by measuring,
# because both initial guesses produced a gate that defended nothing.
#
#   45 relevant per query in 2,000 rows left 62% of the fixture relevant to
#   something. Deliberately coarsening the ranker there IMPROVED P@5 by four to
#   eight points: with that many relevant rows around, matching more loosely
#   still lands on one. The measurement had inverted.
#
#   30 in 10,000 fixed the density but not the difficulty. Stripping the zone
#   out of every token — which costs 0.096 of P@5 on the real corpus — moved the
#   fixture number by 0.000. Ten thousand distractors are not enough to punish a
#   ranker that has stopped discriminating.
#
#   30 in 40,000 reproduces it: the same change costs 0.104 there against 0.096
#   on the full corpus, and the gate blocks. That is the property a fixture needs
#   — not a small corpus, a corpus small enough to commit that still ranks like
#   the real one. 40k possessions is 7.2 MB gzipped, inside the plan's budget,
#   because a possession row is ~125 compressed bytes and the freeze frames in
#   `shots` dominate the file either way.
PER_QUERY = 30
DEFAULT_TARGET = 40_000

# Columns dumped, per table. `embedding` is excluded on purpose: CI runs
# sparse-only, a 384-float vector per row would dominate the file, and a NULL
# embedding is exactly what the dense ranker checks for before skipping a row.
COLUMNS = {
    "matches": ["match_id", "competition_id", "season_id", "competition", "season",
                "match_date", "home_team", "away_team", "home_score", "away_score",
                "has_360"],
    "events": ["event_id", "match_id", "idx", "period", "minute", "second", "type",
               "play_pattern", "possession", "possession_team", "team", "player",
               "position", "x", "y", "end_x", "end_y", "under_pressure", "duration",
               "token", "raw"],
    "shots": ["event_id", "match_id", "competition_id", "season_id", "team", "player",
              "x", "y", "distance", "angle", "body_part", "technique", "shot_type",
              "first_time", "under_pressure", "play_pattern", "is_goal",
              "statsbomb_xg", "freeze_frame", "n_def_in_cone", "dist_nearest_def",
              "gk_dist_to_goal", "gk_off_line"],
    "possessions": ["possession_uid", "match_id", "possession", "team", "opponent",
                    "competition", "season", "play_pattern", "start_idx", "end_idx",
                    "n_events", "duration_s", "start_zone", "end_zone", "zone_path",
                    "token_string", "token_tsv", "ended_in_shot", "xg_sum",
                    "ended_in_goal"],
}

JUDGE_COLS = ["possession_uid", "token_string", "zone_path", "n_events",
              "ended_in_shot", "ended_in_goal", "xg_sum", "play_pattern"]


def relevant_uids(conn, qid: str, filters: Filters, cap: int, rng) -> list:
    """Possessions the rubric calls relevant, sampled down to `cap`.

    Sampled, not top-N by anything: there is no score here to take a top of, and
    that is the point — the fixture must not know what the ranker thinks.
    """
    sql, params = filters.where()
    with conn.cursor() as cur:
        cur.execute(f"SELECT {', '.join(JUDGE_COLS)} FROM possessions WHERE {sql}",
                    params)
        cols = [d.name for d in cur.description]
        hits = [r[0] for r in cur.fetchall() if judge(qid, dict(zip(cols, r)))]
    rng.shuffle(hits)
    return hits[:cap]


def choose_possessions(conn, target: int, seed: int, per_query: int) -> list:
    """The uid list, and a printed account of where each part came from."""
    rng = random.Random(seed)
    queries = yaml.safe_load(
        (HERE.parent / "queries.yaml").read_text(encoding="utf-8"))["queries"]

    chosen, hits = set(), {}
    for q in queries:
        uids = relevant_uids(conn, q["id"], Filters(**(q.get("filters") or {})),
                             per_query, rng)
        hits[q["id"]] = len(uids)
        chosen.update(uids)
    thin = [qid for qid, n in hits.items() if n < 5]
    print(f"rubric-relevant: {len(chosen):,} possessions across {len(queries)} queries")
    if thin:
        print(f"  ! only a handful of relevant rows for {thin} — those queries will "
              f"be insensitive in CI")

    # The blind audit in eval/audit_labels.yaml names specific possessions, and
    # score_retrieval reports human agreement from them. Drop them and CI would
    # print "not yet audited" on every run.
    audit_path = HERE.parent / "audit_labels.yaml"
    if audit_path.exists():
        labels = yaml.safe_load(audit_path.read_text(encoding="utf-8")) or {}
        audited = {k.split("|", 1)[1] for k in labels}
        print(f"audited: {len(audited)} possessions kept so the agreement figure survives")
        chosen.update(audited)

    # Distractors. Without them every possession in the fixture is relevant to
    # something and precision cannot fall, which would make the gate one-sided.
    with conn.cursor() as cur:
        cur.execute("SELECT possession_uid FROM possessions ORDER BY possession_uid")
        everything = [r[0] for r in cur.fetchall()]
    rest = [u for u in everything if u not in chosen]
    rng.shuffle(rest)
    fill = max(0, target - len(chosen))
    chosen.update(rest[:fill])
    print(f"distractors: {min(fill, len(rest)):,} random possessions added")
    return sorted(chosen)


def copy_out(conn, table: str, where_sql: str, params: list, out) -> int:
    """Append one COPY block for `table` and return the row count."""
    cols = COLUMNS[table]
    out.write(f"COPY {table} ({', '.join(cols)}) FROM stdin;\n".encode("utf-8"))
    n = 0
    with conn.cursor() as cur:
        query = (f"COPY (SELECT {', '.join(cols)} FROM {table} WHERE {where_sql}) "
                 f"TO STDOUT")
        with cur.copy(query, params) as copy:
            for block in copy:
                n += bytes(block).count(b"\n")
                out.write(bytes(block))
    out.write(b"\\.\n\n")
    return n


def build(target: int, seed: int, per_query: int) -> Path:
    conn = db.connect()
    uids = choose_possessions(conn, target, seed, per_query)

    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT match_id FROM possessions "
                    "WHERE possession_uid = ANY(%s)", (uids,))
        match_ids = {r[0] for r in cur.fetchall()}
        # Every shot from the held-out competitions, because eval/score_xg.py
        # measures exactly those and a sample would make its log-loss depend on
        # the sample rather than on the model.
        cur.execute("SELECT event_id, match_id FROM shots WHERE (competition_id, season_id) "
                    "IN (SELECT * FROM unnest(%s::int[], %s::int[]))",
                    ([c for c, _ in HELDOUT], [s for _, s in HELDOUT]))
        shot_rows = cur.fetchall()
    shot_ids = [r[0] for r in shot_rows]
    match_ids.update(r[1] for r in shot_rows)

    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(FIXTURE, "wb", compresslevel=9) as out:
        out.write(HEADER.encode("utf-8"))
        # Order matters: events references matches, shots references events.
        n_m = copy_out(conn, "matches", "match_id = ANY(%s)", [sorted(match_ids)], out)
        # Only the shot events. The FK on shots needs them; nothing in the eval
        # path reads any other event, and 1.6M rows is not a fixture.
        n_e = copy_out(conn, "events", "event_id = ANY(%s)", [shot_ids], out)
        n_s = copy_out(conn, "shots", "event_id = ANY(%s)", [shot_ids], out)
        n_p = copy_out(conn, "possessions", "possession_uid = ANY(%s)", [uids], out)
        out.write(FOOTER.encode("utf-8"))
    conn.close()

    size = FIXTURE.stat().st_size / 1e6
    print(f"\nwrote {FIXTURE} ({size:.1f} MB)")
    print(f"  {n_m:,} matches  {n_e:,} events  {n_s:,} shots  {n_p:,} possessions")
    if size > 12:
        print("  ! larger than the 5-10 MB the plan budgets — lower --target, or "
              "trim the held-out shot set")
    return FIXTURE


HEADER = """\
-- PitchQuery CI fixture. Generated by eval/fixtures/make_fixture.py — do not
-- hand-edit. Load it into a database that already has the schema applied:
--
--   psql "$DATABASE_URL" -f sql/001_schema.sql
--   gunzip -c eval/fixtures/corpus.sql.gz | psql -v ON_ERROR_STOP=1 "$DATABASE_URL"
--
-- `events` holds only the events that shots point at, because the FK needs them
-- and nothing in the eval path reads any other event. `possessions.embedding`
-- is absent: CI runs sparse-only.
BEGIN;
"""

FOOTER = """\
COMMIT;
ANALYZE;
"""


def load(url: str) -> None:
    """Apply the schema then the dump, the same two steps CI runs."""
    for name in SCHEMA_FILES:
        subprocess.run(["psql", "-v", "ON_ERROR_STOP=1", url, "-q",
                        "-f", str(REPO_ROOT / "sql" / name)], check=True)
    data = gzip.decompress(FIXTURE.read_bytes())
    subprocess.run(["psql", "-v", "ON_ERROR_STOP=1", url, "-q"],
                   input=data, check=True)
    print(f"loaded {FIXTURE.name} into {url.rsplit('@', 1)[-1]}")


def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=DEFAULT_TARGET,
                    help="possessions to aim for (rubric hits are kept regardless)")
    ap.add_argument("--per-query", type=int, default=PER_QUERY,
                    help="rubric-relevant possessions kept per eval query")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--load", metavar="DATABASE_URL",
                    help="load the existing fixture instead of rebuilding it")
    args = ap.parse_args()
    if args.load:
        load(args.load)
    else:
        build(args.target, args.seed, args.per_query)


if __name__ == "__main__":
    db.cli(cli)()
