"""Phase 1: the whole ingest as one Prefect flow.

The scripts in `ingest/` are imported and called, not shelled out to. That is
the difference between a pipeline and a shell script with a UI: `subprocess.run`
can only pass an exit code, so every step would have to re-derive what the
previous one already knew — which matches were fetched, which were loaded. Here
`fetch` hands the loader a list of match ids and the loader hands the possession
builder the ones that actually landed.

Order, and why it is this order (plan Phase 3):

    fetch  ->  load raw  ->  dbt staging  ->  possessions  ->  dbt marts  ->  index
               (Python)      (SQL)           (Python)          (SQL)         (Python)

dbt owns the typed, tested SQL layers. Possession tokenising is Python and stays
Python — it is sequence logic over ordered events, not set logic over rows, and
rewriting it as SQL would be a worse version of the same thing.

Run:
  prefect server start                 # terminal 1, UI on :4200
  python -m pipeline.flows             # terminal 2
  python -m pipeline.flows --comp 43:106 --no-dbt
"""
import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from prefect import flow, get_run_logger, task  # noqa: E402

from core import db  # noqa: E402
from pipeline import watermark  # noqa: E402

WAREHOUSE = REPO_ROOT / "warehouse"

# The competitions this corpus is built from. Overridable per run — the flow
# takes `comps` as a parameter precisely so one competition and ten are the
# same command.
DEFAULT_COMPS = [
    "43:106",    # FIFA World Cup 2022
    "72:107",    # FIFA Women's World Cup 2023
    "55:43",     # UEFA Euro 2020
    "55:282",    # UEFA Euro 2024
    "53:106",    # UEFA Women's Euro 2022
    "53:315",    # UEFA Women's Euro 2025
    "11:90",     # La Liga 2020/2021
    "9:281",     # 1. Bundesliga 2023/2024
    "7:235",     # Ligue 1 2022/2023
    "7:108",     # Ligue 1 2021/2022
]

# Every task that touches Postgres or the network gets the same retry policy.
# core/db.py raises `Unavailable` rather than SystemExit for this to work: a
# BaseException would abort the flow run instead of being retried, and "the
# database blinked" is the case retries exist for.
RETRY = dict(retries=3, retry_delay_seconds=30)


def script(name: str):
    """Import `ingest/<name>.py`.

    The files are numbered so that reading the directory tells you what order to
    run them in, which makes them illegal Python identifiers. Loading them by
    path is the cost of that, and it is a small one.
    """
    path = REPO_ROOT / "ingest" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"ingest.{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# --- tasks --------------------------------------------------------------------

@task(name="fetch", **RETRY)
def fetch(comps: list, *, limit: int = None, with_360: bool = False,
          incremental: bool = True) -> dict:
    """Download event files, skipping anything already past the watermark."""
    log = get_run_logger()
    since = {}
    if incremental:
        conn = db.connect()
        try:
            since = watermark.since(conn, comps)
            log.info(watermark.report(conn))
        finally:
            conn.close()
    out = script("02_fetch").main(comps, limit=limit, with_360=with_360, since=since)
    log.info("fetch: %s new files, %s matches in scope, %s missing upstream",
             out["rows"], len(out["match_ids"]), out["missing"])
    return out


@task(name="load-events", **RETRY)
def load(fetched: dict, *, init: bool = False, validate: bool = True) -> dict:
    """Upsert the fetched matches, validating each batch and moving the mark."""
    log = get_run_logger()
    out = script("03_load_events").main(fetched["match_ids"], init=init, validate=validate)
    log.info("load: %s events over %s matches (%s shots)",
             out["rows"], out["matches"], out["shots"])
    return out


@task(name="dbt", retries=2, retry_delay_seconds=30)
def dbt_build(select: str, upstream: dict = None) -> dict:
    """`dbt build --select <select>` in warehouse/ — models and tests together.

    `upstream` is unused inside the body and is not decoration: it is what tells
    Prefect this SQL layer runs after the Python step that fills the tables it
    reads.
    """
    log = get_run_logger()
    # `--profiles-dir .` because warehouse/profiles.yml is committed next to the
    # project rather than left in ~/.dbt, where dbt looks by default and where
    # nothing about this repo could put it.
    cmd = [sys.executable, "-m", "dbt.cli.main", "build",
           "--profiles-dir", ".", "--select", select]
    proc = subprocess.run(cmd, cwd=WAREHOUSE, capture_output=True, text=True)
    for line in (proc.stdout or "").strip().splitlines()[-25:]:
        log.info("dbt| %s", line)
    if proc.returncode != 0:
        log.error("dbt| %s", (proc.stderr or "").strip()[-2000:])
        raise RuntimeError(f"dbt build --select {select} failed "
                           f"(exit {proc.returncode}) — see the log above")
    return {"select": select, "ok": True}


@task(name="build-possessions", **RETRY)
def possessions(loaded: dict, *, validate: bool = True) -> dict:
    """Group events into possessions and write one token string each."""
    log = get_run_logger()
    out = script("04_build_possessions").main(loaded["match_ids"], validate=validate)
    log.info("possessions: %s rows over %s matches", out["rows"], out["matches"])
    return out


@task(name="index", retries=2, retry_delay_seconds=30)
def index(built: dict, *, sparse_only: bool = True) -> dict:
    """Refit TF-IDF (and optionally MiniLM) over the whole corpus.

    Always whole-corpus: idf is a property of the corpus, so one new competition
    changes the weight of every token in it. Which is exactly why it is skipped
    when nothing was rebuilt — refitting an unchanged corpus produces a
    byte-identical matrix, and an incremental pipeline whose last step is
    unconditional is not incremental.
    """
    log = get_run_logger()
    if not built["rows"]:
        log.info("index: corpus unchanged, keeping the existing matrix")
        conn = db.connect()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT count(*) FROM possessions")
                return {"rows": cur.fetchone()[0], "skipped": True}
        finally:
            conn.close()
    out = script("05_embed").main(sparse_only=sparse_only)
    log.info("index: %s possessions, %s terms", out["rows"], out.get("terms", "-"))
    return out


# --- flow ---------------------------------------------------------------------

@flow(name="pitchquery-ingest", log_prints=True)
def ingest(comps: Optional[list] = None, *, init: bool = False,
           limit: Optional[int] = None, incremental: bool = True,
           sparse_only: bool = True, run_dbt: bool = True,
           validate: bool = True) -> dict:
    """One command for the whole ingest.

    `comps` is a flow parameter so a rerun can be one competition rather than
    all of them, which is the thing the watermark exists to make cheap.
    """
    comps = comps or DEFAULT_COMPS
    fetched = fetch(comps, limit=limit, incremental=incremental)
    loaded = load(fetched, init=init, validate=validate)

    silver = dbt_build("staging", loaded) if run_dbt else None
    built = possessions(loaded, validate=validate,
                        wait_for=[silver] if silver is not None else None)

    gold = dbt_build("marts", built) if run_dbt else None
    indexed = index(built, sparse_only=sparse_only,
                    wait_for=[gold] if gold is not None else None)

    return {"fetched": fetched["rows"], "events": loaded["rows"],
            "possessions": built["rows"], "corpus": indexed["rows"]}


# --- the nightly flow ---------------------------------------------------------

@task(name="grade-candidates", retries=2, retry_delay_seconds=30)
def collect_candidates() -> dict:
    """Turn yesterday's bad rankings into an eval backlog (Phase 8)."""
    log = get_run_logger()
    from pipeline import telemetry

    out = telemetry.main(write=True)
    log.info("telemetry: %s searches whose opened result sat at rank >= %s, "
             "%s unrecognised words",
             out["candidates"], telemetry.DEEP_CLICK_RANK, out["vocabulary"])
    return out


@task(name="drift", retries=2, retry_delay_seconds=30)
def drift(split: str = "heldout") -> dict:
    """Re-measure feature drift and write a dated report (Phase 9)."""
    log = get_run_logger()
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "monitoring" / "drift_report.py"),
         "--split", split],
        cwd=REPO_ROOT, capture_output=True, text=True)
    for line in (proc.stdout or "").strip().splitlines()[-12:]:
        log.info("drift| %s", line)
    if proc.returncode != 0:
        raise RuntimeError(f"drift report failed: {(proc.stderr or '')[-800:]}")
    return {"split": split, "ok": True}


@flow(name="pitchquery-nightly", log_prints=True)
def nightly(splits: Optional[list] = None) -> dict:
    """Everything that should happen once a day and nothing that must.

    Separate from the ingest flow on purpose. This one reads what the last day
    produced — the query log, the feature distributions — and neither task can
    change the corpus, so a failure here is a missing report rather than a
    pipeline that has to be rerun.
    """
    found = collect_candidates()
    for split in (splits or ["heldout", "gender"]):
        drift(split)
    return found


def cli():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--nightly", action="store_true",
                    help="run the reporting flow instead of the ingest")
    ap.add_argument("--comp", action="append", metavar="COMP_ID:SEASON_ID",
                    help="repeatable; defaults to the ten competitions in DEFAULT_COMPS")
    ap.add_argument("--init", action="store_true", help="apply the schema files first")
    ap.add_argument("--limit", type=int, default=None, help="max matches per competition")
    ap.add_argument("--full", action="store_true",
                    help="ignore the watermark and reconsider every match")
    ap.add_argument("--dense", action="store_true", help="also rebuild MiniLM embeddings")
    ap.add_argument("--no-dbt", action="store_true", help="skip the two dbt layers")
    ap.add_argument("--no-validate", action="store_true", help="skip the Pandera contracts")
    args = ap.parse_args()
    if args.nightly:
        nightly()
        return
    ingest(args.comp, init=args.init, limit=args.limit, incremental=not args.full,
           sparse_only=not args.dense, run_dbt=not args.no_dbt,
           validate=not args.no_validate)


if __name__ == "__main__":
    cli()
