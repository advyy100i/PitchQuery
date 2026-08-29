"""Phase 12: the operational view, served as data rather than rendered as a page.

Five sections, in the order you would actually check them: did the pipeline run,
is the data there, which model is champion, has the data moved, and what are
people searching for.

This was a Streamlit app. It is an endpoint now, read by `web/app/pipeline` —
the same Next.js app, the same types, the same design system and the same
deploy as the product. A second framework serving a second UI from a second host
is a second thing to keep alive, and it looked nothing like the thing it was
reporting on.

Every section degrades on its own and says WHY, which matters more here than
anywhere else in this API: the hosted deployment genuinely cannot answer three
of the five. `ingest_watermark` and the dbt schemas never reach Neon —
deploy/export_to_neon.py copies four tables and no others — and MLflow is a
SQLite file on a laptop. "Not here, and here is where it lives" is the true
answer. A blank panel reads as broken.

Cached for 60 seconds. Fifteen counts and four aggregates is not something to
run per page load against a free tier, and nothing reported here moves faster.
"""
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import psycopg  # noqa: E402

from core.config import DOCS_DIR, REPO_ROOT  # noqa: E402

TTL_S = 60
_cache: dict = {}

# Bronze is written by Python, silver and gold by dbt. Ordered as the data flows
# so that a gold table lagging bronze is visible as a step that did not happen.
LAYERS = [
    ("bronze", "matches", "public.matches"),
    ("bronze", "events", "public.events"),
    ("bronze", "shots", "public.shots"),
    ("bronze", "possessions", "public.possessions"),
    ("silver", "stg_events", "analytics.stg_events"),
    ("silver", "stg_shots", "analytics.stg_shots"),
    ("silver", "stg_freeze_frames", "analytics.stg_freeze_frames"),
    ("gold", "mart_xg_features", "analytics.mart_xg_features"),
    ("gold", "mart_team_possessions", "analytics.mart_team_possessions"),
]

# A ceiling on what one row count is allowed to cost. `events` is several
# million rows and count(*) walks every one of them; a status page is not worth
# a ten-second request. Past the timeout the planner's own estimate is reported
# AS an estimate — a different number, labelled as one, rather than a precise
# figure that is quietly late.
COUNT_TIMEOUT_MS = 3000

BASELINE_XG = REPO_ROOT / "eval" / "baselines" / "xg.json"


def _rows(conn, sql: str, params=None) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute(sql, params or [])
        cols = [d.name for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]


def _iso(v):
    return v.isoformat() if hasattr(v, "isoformat") else v


# --- 1. pipeline runs ---------------------------------------------------------

def pipeline(conn) -> dict:
    """The watermark, which is the record of what committed.

    Not of what was attempted: pipeline/watermark.py advances it inside the same
    transaction as the inserts it describes, so a Prefect run that died halfway
    leaves this pointing at the last match that actually landed.
    """
    try:
        runs = _rows(conn, """
            SELECT competition_id, season_id, last_match_id, last_run_at, rows_loaded
            FROM ingest_watermark ORDER BY last_run_at DESC NULLS LAST
        """)
    except psycopg.Error as exc:
        return {"runs": [], "error": str(exc).strip(),
                "hint": "ingest_watermark is not in this database. It is written "
                        "by the loader and does not ship to the hosted copy — "
                        "deploy/export_to_neon.py sends matches, events, shots "
                        "and possessions only."}
    for r in runs:
        r["last_run_at"] = _iso(r["last_run_at"])
        r["rows_loaded"] = int(r["rows_loaded"] or 0)
    return {"runs": runs, "error": None,
            "hint": None if runs else "No ingest recorded yet. "
                                      "Run `python -m pipeline.flows`."}


# --- 2. rows per layer --------------------------------------------------------

def _count(conn, table: str) -> dict:
    """Exact if it is cheap, estimated if it is not, absent if it is not there.

    `to_regclass` rather than catching the exception: a missing dbt model is a
    normal state on a fresh checkout and on every hosted deployment, and asking
    first costs one cheap lookup instead of putting an error through a path
    meant for failures.
    """
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass(%s)", (table,))
        if cur.fetchone()[0] is None:
            return {"rows": None, "estimated": False, "state": "missing"}
    try:
        with conn.cursor() as cur:
            cur.execute(f"SET statement_timeout = {COUNT_TIMEOUT_MS}")
            cur.execute(f"SELECT count(*) FROM {table}")
            return {"rows": int(cur.fetchone()[0]), "estimated": False, "state": "ok"}
    except psycopg.errors.QueryCanceled:
        with conn.cursor() as cur:
            cur.execute("SELECT reltuples::bigint FROM pg_class WHERE oid = to_regclass(%s)",
                        (table,))
            n = cur.fetchone()
        return {"rows": int(n[0]) if n and n[0] is not None else None,
                "estimated": True, "state": "ok"}
    except psycopg.Error as exc:
        return {"rows": None, "estimated": False, "state": str(exc).strip()}
    finally:
        with conn.cursor() as cur:
            cur.execute("RESET statement_timeout")


def layers(conn) -> dict:
    tables = []
    for layer, name, table in LAYERS:
        tables.append({"layer": layer, "table": name, "qualified": table,
                       **_count(conn, table)})
    missing = [t["table"] for t in tables if t["state"] == "missing"]
    return {"tables": tables,
            "hint": (f"Not built here: {', '.join(missing)}. The dbt layers are "
                     f"local — `cd warehouse && dbt build --profiles-dir .`")
                    if missing else None}


# --- 3. the champion model ----------------------------------------------------

def champion() -> dict:
    """MLflow if it is reachable, the committed baseline if it is not.

    The fallback is not a lesser version of the same thing and is not presented
    as one: `eval/baselines/xg.json` is what the shipped artefact scored on the
    held-out competitions, written by eval/report.py and committed. It is the
    number in docs/benchmark.md. What it cannot tell you is which registry
    version is serving, so that field is absent rather than guessed.
    """
    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        from models import tracking

        mlflow.set_tracking_uri(tracking.tracking_uri())
        client = MlflowClient()
        version = client.get_model_version_by_alias(tracking.MODEL_NAME, "champion")
        run = client.get_run(version.run_id)
        return {"source": "mlflow", "version": str(version.version),
                "metrics": run.data.metrics, "params": run.data.params,
                "commit": run.data.tags.get("git_commit"), "error": None}
    except Exception as exc:                       # not installed, or no registry
        try:
            b = json.loads(BASELINE_XG.read_text(encoding="utf-8"))
        except Exception as read_exc:
            return {"source": None, "error": f"{exc}", "read_error": str(read_exc),
                    "metrics": {}, "params": {}}
        return {
            "source": "baseline",
            "version": None,
            "metrics": {"log_loss": b.get("logloss"), "brier": b.get("brier"),
                        "ece": b.get("ece"), "auc": b.get("auc"),
                        "statsbomb_log_loss": b.get("statsbomb_logloss"),
                        "log_loss_gap": b.get("logloss_gap_to_statsbomb"),
                        "observed_over_expected": b.get("observed_over_expected")},
            "params": {"n_shots": b.get("n_shots"),
                       "test_comps": ", ".join(b.get("test_comps") or []),
                       "scope": b.get("scope"),
                       "measured_at": b.get("measured_at")},
            "commit": None,
            "error": None,
            "note": (f"MLflow is not reachable here ({type(exc).__name__}), so this "
                     f"is eval/baselines/xg.json — what the shipped artefact scored "
                     f"on the held-out competitions, committed to the repo. The "
                     f"registry is a local SQLite store; start it with "
                     f"`mlflow server --backend-store-uri sqlite:///mlflow.db`."),
        }


# --- 4. drift -----------------------------------------------------------------

def drift() -> dict:
    """The committed JSON reports, newest first.

    Effect size, not a drift verdict. Over thousands of shots a statistical test
    calls nearly every column drifted; Cohen's d says how far apart the
    distributions actually are, which is the question.
    """
    files = sorted((DOCS_DIR / "drift").glob("*.json"), reverse=True)
    reports = []
    for f in files:
        try:
            r = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        reports.append({
            "name": f.stem,
            "split": r.get("split"), "date": r.get("date"),
            "reference": r.get("reference"), "current": r.get("current"),
            "n_reference": r.get("n_reference"), "n_current": r.get("n_current"),
            "shifts": [{"feature": s["feature"],
                        "reference_mean": s["reference_mean"],
                        "current_mean": s["current_mean"],
                        "cohens_d": s["cohens_d"]} for s in r.get("shifts", [])],
        })
    return {"reports": reports,
            "hint": None if reports else "No drift report yet. "
                                         "Run `python monitoring/drift_report.py`."}


# --- 5. the query log ---------------------------------------------------------

DAILY_SQL = """
WITH s AS (
  SELECT date_trunc('day', ts)::date AS day,
         count(*) AS searches,
         percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms) AS p95_ms
  FROM search_log WHERE ts > now() - interval '30 days' GROUP BY 1
), c AS (
  SELECT date_trunc('day', s2.ts)::date AS day, count(*) AS clicks
  FROM click_log c2 JOIN search_log s2 ON s2.id = c2.search_id
  WHERE s2.ts > now() - interval '30 days' GROUP BY 1
)
SELECT s.day, s.searches, coalesce(c.clicks, 0) AS clicks, s.p95_ms
FROM s LEFT JOIN c USING (day) ORDER BY s.day
"""
# Two aggregates joined on the day, rather than one grouped join. Counting
# searches across `search_log LEFT JOIN click_log` counts a search once per
# click it received, and takes the p95 over the same duplicated rows — so a
# single popular search inflates the volume line and drags the latency line
# toward whatever that one query cost.


def queries(conn) -> dict:
    try:
        daily = _rows(conn, DAILY_SQL)
    except psycopg.Error as exc:
        return {"daily": [], "unparsed": [], "deep_clicks": [], "totals": {},
                "error": str(exc).strip(),
                "hint": "search_log and click_log are created on API startup. "
                        "PITCHQUERY_SEARCH_LOG=0 switches the writes off."}
    for d in daily:
        d["day"] = _iso(d["day"])
        d["searches"] = int(d["searches"])
        d["clicks"] = int(d["clicks"])
        d["p95_ms"] = float(d["p95_ms"]) if d["p95_ms"] is not None else None

    unparsed = _rows(conn, """
        SELECT word, count(*) AS n
        FROM search_log, unnest(unparsed_words) AS word
        GROUP BY 1 ORDER BY 2 DESC, 1 LIMIT 20
    """)
    for u in unparsed:
        u["n"] = int(u["n"])

    # Rank 5 or below is the definition of a ranking that was wrong: the answer
    # was there and the ranker put it where nobody looks.
    deep = _rows(conn, """
        SELECT s.query_text, c.rank, c.possession_uid, s.ranker, c.ts
        FROM click_log c JOIN search_log s ON s.id = c.search_id
        WHERE c.rank >= 5 ORDER BY c.ts DESC LIMIT 20
    """)
    for d in deep:
        d["ts"] = _iso(d["ts"])

    totals = _rows(conn, """
        SELECT (SELECT count(*) FROM search_log)                       AS searches,
               (SELECT count(*) FROM click_log)                        AS clicks,
               (SELECT count(*) FROM click_log WHERE rank >= 5)        AS deep_clicks,
               (SELECT count(DISTINCT word) FROM search_log,
                       unnest(unparsed_words) AS word)                 AS unknown_words
    """)[0]
    return {"daily": daily, "unparsed": unparsed, "deep_clicks": deep,
            "totals": {k: int(v or 0) for k, v in totals.items()},
            "error": None,
            "hint": None if daily else "No searches in the last 30 days. Run the "
                                       "API and search for something."}


# --- the whole page -----------------------------------------------------------

def snapshot(get_conn) -> dict:
    """Everything the page needs, in one round trip.

    One request and not five, because the hosted API sleeps after 15 minutes and
    takes ~50 s to wake: five parallel calls would each pay that, and the page
    would come back in pieces over a minute.

    `get_conn` is a callable rather than a connection so this module uses the
    same reconnect-on-a-dropped-socket path as every other endpoint, without
    importing api.main and closing the loop.
    """
    hit = _cache.get("snapshot")
    if hit and time.monotonic() - hit[0] < TTL_S:
        return hit[1]
    conn = get_conn()
    doc = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cache_ttl_s": TTL_S,
        "pipeline": pipeline(conn),
        "layers": layers(conn),
        "champion": champion(),
        "drift": drift(),
        "queries": queries(conn),
    }
    _cache["snapshot"] = (time.monotonic(), doc)
    return doc
