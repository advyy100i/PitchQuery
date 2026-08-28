"""Phase 12: one page showing whether the whole thing is working.

Five sections, in the order you would actually check them: did the pipeline run,
is the data there, which model is champion, has the data moved, and what are
people searching for.

Every section degrades on its own. The dashboard is deployable to Streamlit
Community Cloud pointed at Neon, where MLflow is not running and the drift files
may not have been committed yet — a panel that cannot answer says so and the
other four still render. A dashboard that goes blank because one dependency is
absent gets closed and never opened again.

Run:
  streamlit run dashboard/app.py

Deploy free on Streamlit Community Cloud (public repo required) and set
DATABASE_URL in the app's secrets.
"""
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

st.set_page_config(page_title="PitchQuery — pipeline", layout="wide",
                   page_icon="⚽")

# Streamlit Cloud has no .env; it has st.secrets. Copy it into the environment
# before core/config.py is imported, so every module downstream reads one thing.
#
# In a try, because `st.secrets` does not return empty when there is no secrets
# file — it raises. Locally there never is one (the connection comes from .env),
# so the unguarded version crashed the page on every developer machine while
# working perfectly on the deployment it was written for.
try:
    if "DATABASE_URL" in st.secrets and not os.getenv("DATABASE_URL"):
        os.environ["DATABASE_URL"] = st.secrets["DATABASE_URL"]
except Exception:
    pass

from core import db  # noqa: E402
from core.config import DOCS_DIR  # noqa: E402


@st.cache_resource
def connection():
    return db.connect(autocommit=True)


def query(sql: str, params=None) -> pd.DataFrame:
    with connection().cursor() as cur:
        cur.execute(sql, params or [])
        cols = [d.name for d in cur.description]
        return pd.DataFrame(cur.fetchall(), columns=cols)


def section(title: str, caption: str = None):
    st.subheader(title)
    if caption:
        st.caption(caption)


st.title("PitchQuery")
st.caption(
    "Tactical possession search over StatsBomb open data. This page is the "
    "operational view: the pipeline, the warehouse, the model registry, the "
    "drift reports and the query log. Data source: StatsBomb."
)

try:
    connection()
except Exception as exc:
    st.error(f"Cannot reach the database: {exc}")
    st.stop()

# --- 1. pipeline runs ---------------------------------------------------------

section("Pipeline runs",
        "From `ingest_watermark`, which the loader advances inside the same "
        "transaction as its inserts. It is the record of what actually "
        "committed, not of what was attempted — a Prefect run that failed "
        "halfway leaves this pointing at the last match that landed.")
try:
    runs = query("""
        SELECT competition_id, season_id, last_match_id, last_run_at, rows_loaded
        FROM ingest_watermark ORDER BY last_run_at DESC
    """)
    if runs.empty:
        st.info("No ingest recorded yet. Run `python -m pipeline.flows`.")
    else:
        left, right = st.columns([1, 2])
        left.metric("Competitions loaded", len(runs))
        left.metric("Rows loaded (cumulative)", f"{int(runs['rows_loaded'].sum()):,}")
        left.metric("Last run", str(runs["last_run_at"].max())[:16])
        right.dataframe(runs, use_container_width=True, hide_index=True)
except Exception as exc:
    st.warning(f"watermark unavailable: {exc}")

# --- 2. row counts per layer --------------------------------------------------

section("Rows per layer",
        "Bronze is written by Python, silver and gold by dbt. A gold table that "
        "has not kept up with bronze means `dbt build` has not run since the "
        "last ingest.")

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


@st.cache_data(ttl=60)
def layer_counts() -> pd.DataFrame:
    rows = []
    for layer, name, table in LAYERS:
        try:
            n = int(query(f"SELECT count(*) AS n FROM {table}")["n"].iloc[0])
        except Exception:
            # A missing dbt model is a normal state on a fresh checkout, not an
            # error worth a red box across the page.
            connection().rollback() if not connection().autocommit else None
            n = None
        rows.append({"layer": layer, "table": name, "rows": n})
    return pd.DataFrame(rows)


counts = layer_counts()
cols = st.columns(3)
for col, layer in zip(cols, ("bronze", "silver", "gold")):
    part = counts[counts["layer"] == layer]
    total = part["rows"].dropna().sum()
    col.metric(f"{layer} rows", f"{int(total):,}" if total else "—")
    col.dataframe(part[["table", "rows"]], use_container_width=True,
                  hide_index=True)
missing = counts[counts["rows"].isna()]["table"].tolist()
if missing:
    st.caption(f"Not built: {', '.join(missing)} — run "
               f"`cd warehouse && dbt build --profiles-dir .`")

# --- 3. the champion model ----------------------------------------------------

section("Champion xG model",
        "From the MLflow registry. The `champion` alias only moves when a run's "
        "held-out log-loss actually beats the incumbent — the rule is code in "
        "`models/tracking.py`, not a habit.")


@st.cache_data(ttl=60)
def champion() -> dict:
    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        from models import tracking

        mlflow.set_tracking_uri(tracking.tracking_uri())
        client = MlflowClient()
        version = client.get_model_version_by_alias(tracking.MODEL_NAME, "champion")
        run = client.get_run(version.run_id)
        return {"version": version.version, "metrics": run.data.metrics,
                "params": run.data.params, "tags": run.data.tags}
    except Exception as exc:
        return {"error": str(exc)}


champ = champion()
if "error" in champ:
    st.info(f"MLflow registry not reachable ({champ['error']}). It is a local "
            f"SQLite store — start it with `mlflow server "
            f"--backend-store-uri sqlite:///mlflow.db`, or train a model.")
    # Fall back to the committed benchmark, which is in the repo either way.
    bench = DOCS_DIR / "benchmark.md"
    if bench.exists():
        st.caption("Falling back to the committed `docs/benchmark.md`:")
        st.markdown(bench.read_text(encoding="utf-8").split("## By competition")[0])
else:
    m = champ["metrics"]
    c = st.columns(5)
    c[0].metric("Version", f"v{champ['version']}")
    c[1].metric("Log-loss", f"{m.get('log_loss', float('nan')):.4f}")
    c[2].metric("Brier", f"{m.get('brier', float('nan')):.4f}")
    c[3].metric("ECE", f"{m.get('ece', float('nan')):.4f}")
    gap = m.get("gap_closed")
    c[4].metric("Gap closed to StatsBomb",
                f"{gap * 100:.0f}%" if gap is not None else "—")
    st.caption(
        f"Trained at commit `{champ['tags'].get('git_commit', '?')}` on "
        f"{champ['params'].get('n_train_shots', '?')} shots, held out "
        f"{champ['params'].get('test_comps', '?')}. Calibration: "
        f"{champ['params'].get('calibration', '?')}.")

# --- 4. drift -----------------------------------------------------------------

section("Feature drift",
        "Reported as effect size, not as a drift verdict. Over thousands of "
        "shots a statistical test calls nearly every column drifted; Cohen's d "
        "says how far apart the distributions actually are.")

drift_files = sorted((DOCS_DIR / "drift").glob("*.json"), reverse=True)
if not drift_files:
    st.info("No drift report yet. Run `python monitoring/drift_report.py`.")
else:
    pick = st.selectbox("Report", drift_files, format_func=lambda p: p.stem)
    report = json.loads(pick.read_text(encoding="utf-8"))
    st.caption(f"{report['n_current']:,} shots from the {report['current']} "
               f"against {report['n_reference']:,} from the {report['reference']}.")
    shifts = pd.DataFrame(report["shifts"])
    left, right = st.columns([2, 3])
    left.dataframe(
        shifts[["feature", "reference_mean", "current_mean", "cohens_d"]],
        use_container_width=True, hide_index=True)
    right.bar_chart(shifts.set_index("feature")["cohens_d"], horizontal=True)
    top = shifts.iloc[0]
    right.caption(
        f"Largest shift: **{top['feature']}**, Cohen's d {top['cohens_d']:+.2f}. "
        f"Anything under about |0.2| is a difference you would struggle to see.")

# --- 5. query log -------------------------------------------------------------

section("Queries",
        "From `search_log` and `click_log`. The vocabulary table is the useful "
        "artefact: it is the list of words the parser does not know, written by "
        "the people using it.")

try:
    vol = query("""
        SELECT date_trunc('day', s.ts)::date AS day,
               count(*) AS searches,
               count(c.id) AS clicks,
               percentile_cont(0.95) WITHIN GROUP (ORDER BY s.latency_ms) AS p95_ms
        FROM search_log s LEFT JOIN click_log c ON c.search_id = s.id
        WHERE s.ts > now() - interval '30 days'
        GROUP BY 1 ORDER BY 1
    """)
    if vol.empty:
        st.info("No searches logged yet. Run the API and search for something.")
    else:
        a, b = st.columns(2)
        a.caption("Searches and clicks per day")
        a.line_chart(vol.set_index("day")[["searches", "clicks"]])
        b.caption("p95 latency, ms")
        b.line_chart(vol.set_index("day")["p95_ms"])

        c1, c2 = st.columns(2)
        c1.caption("Words the parser could not place")
        words = query("""
            SELECT word, count(*) AS n
            FROM search_log, unnest(unparsed_words) AS word
            GROUP BY 1 ORDER BY 2 DESC LIMIT 20
        """)
        c1.dataframe(words, use_container_width=True, hide_index=True)

        c2.caption("Results opened at rank 5 or below — rankings that were wrong")
        deep = query("""
            SELECT s.query_text, c.rank, c.possession_uid, s.ranker
            FROM click_log c JOIN search_log s ON s.id = c.search_id
            WHERE c.rank >= 5 ORDER BY c.rank DESC LIMIT 20
        """)
        if deep.empty:
            c2.caption("None yet. `python -m pipeline.telemetry --write` collects "
                       "these into eval/candidates.json for hand-grading.")
        else:
            c2.dataframe(deep, use_container_width=True, hide_index=True)
except Exception as exc:
    st.info(f"Query log not available ({exc}). It is created on API startup.")

st.divider()
st.caption(
    "Orchestration (Prefect), the warehouse (dbt), tracking (MLflow), "
    "monitoring (Prometheus/Grafana) and the match replay (Redpanda) all run "
    "locally via Docker profiles. Only the API and the web app are hosted, "
    "because the free tier gives 512 MB and the API already uses 252 MB."
)
