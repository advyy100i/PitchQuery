"""Phase 8 + 10: what the API records about itself.

Two separate things that both have to stay off the hot path.

`log_search` writes a row to `search_log`. It is called from a FastAPI
background task, which runs after the response has been handed to the client, so
a slow insert costs the user nothing. It also swallows its own exceptions: a
telemetry table that can take the search endpoint down with it is a liability,
not an observability feature. The write uses its own short-lived connection for
the same reason — sharing the request connection would mean a failed insert
poisons the transaction the response was built from.

The Prometheus counters are the three things the default HTTP metrics cannot
see. Request duration and status codes come free from the instrumentator; what
it cannot know is whether the parser understood the query, whether the search
found anything, or which ranker answered. Those are the numbers that say the
product is working rather than merely responding.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import db  # noqa: E402
from core.config import REPO_ROOT  # noqa: E402

TELEMETRY_DDL = REPO_ROOT / "sql" / "004_telemetry.sql"


def _flag(name: str, default: bool) -> bool:
    return os.getenv(name, "1" if default else "0").lower() not in ("0", "false", "no")


# On by default locally, and switchable off for a deployment that would rather
# not take writes on a free-tier database.
LOG_SEARCHES = _flag("PITCHQUERY_SEARCH_LOG", True)


def ensure_tables(conn) -> bool:
    """Create search_log/click_log if missing. Returns whether they are usable."""
    if not LOG_SEARCHES:
        return False
    try:
        with conn.cursor() as cur:
            cur.execute(TELEMETRY_DDL.read_text(encoding="utf-8"))
        if not conn.autocommit:
            conn.commit()
        return True
    except Exception as exc:                       # a read-only replica, say
        print(f"search logging off: {type(exc).__name__}: {exc}")
        return False


INSERT_SQL = """
INSERT INTO search_log (query_text, parsed_filters, sequence_hint, unparsed_words,
                        ranker, latency_ms, rerank_ms, n_results, n_candidates, top_uids)
VALUES (%s, %s::jsonb, %s, %s, %s, %s, %s, %s, %s, %s)
RETURNING id
"""


def log_search(payload: dict) -> None:
    """Insert one search. Never raises — see the module docstring."""
    if not LOG_SEARCHES:
        return
    try:
        import json

        conn = db.connect(autocommit=True)
        try:
            with conn.cursor() as cur:
                cur.execute(INSERT_SQL, (
                    payload.get("query_text"),
                    json.dumps(payload.get("parsed_filters") or {}, default=str),
                    payload.get("sequence_hint"),
                    payload.get("unparsed_words") or [],
                    payload.get("ranker"),
                    int(payload.get("latency_ms") or 0),
                    int(payload["rerank_ms"]) if payload.get("rerank_ms") else None,
                    int(payload.get("n_results") or 0),
                    int(payload.get("n_candidates") or 0),
                    payload.get("top_uids") or [],
                ))
        finally:
            conn.close()
    except Exception as exc:
        # Deliberately a print and not a raise. The response has already gone
        # out; there is nobody left to tell.
        print(f"search_log insert failed ({type(exc).__name__}: {exc})")


def reserve_search_id(conn, payload: dict):
    """Insert synchronously and return the new id, or None.

    /search needs the id in the response so the frontend can attach a click to
    it, and an id that arrives after the response is useless. So this one write
    is on the hot path — a single indexed insert, measured at well under a
    millisecond — while everything else about logging stays off it.
    """
    if not LOG_SEARCHES:
        return None
    try:
        import json

        with conn.cursor() as cur:
            cur.execute(INSERT_SQL, (
                payload.get("query_text"),
                json.dumps(payload.get("parsed_filters") or {}, default=str),
                payload.get("sequence_hint"),
                payload.get("unparsed_words") or [],
                payload.get("ranker"),
                int(payload.get("latency_ms") or 0),
                int(payload["rerank_ms"]) if payload.get("rerank_ms") else None,
                int(payload.get("n_results") or 0),
                int(payload.get("n_candidates") or 0),
                payload.get("top_uids") or [],
            ))
            return cur.fetchone()[0]
    except Exception as exc:
        print(f"search_log insert failed ({type(exc).__name__}: {exc})")
        return None


def log_click(conn, search_id: int, possession_uid: str, rank: int) -> bool:
    try:
        with conn.cursor() as cur:
            cur.execute("INSERT INTO click_log (search_id, possession_uid, rank) "
                        "VALUES (%s, %s, %s)", (search_id, possession_uid, rank))
        if not conn.autocommit:
            conn.commit()
        return True
    except Exception as exc:
        print(f"click_log insert failed ({type(exc).__name__}: {exc})")
        return False


# --- Phase 10: the counters the default HTTP metrics cannot produce -----------

class _NullCounter:
    def labels(self, **_kw):
        return self

    def inc(self, *_a):
        pass


PARSE_FAILURES = ZERO_RESULTS = RANKER_USED = _NullCounter()
RERANK_SECONDS = None
METRICS_AVAILABLE = False


def install(app) -> str:
    """Attach /metrics and define the three custom counters.

    Returns a status string for /health. Optional the same way the dense ranker
    and the xG model are: a missing observability package degrades the metrics
    endpoint, not the product.
    """
    global PARSE_FAILURES, ZERO_RESULTS, RANKER_USED, RERANK_SECONDS, METRICS_AVAILABLE
    if not _flag("PITCHQUERY_METRICS", True):
        return "disabled by PITCHQUERY_METRICS=0"
    try:
        from prometheus_client import Counter, Histogram
        from prometheus_fastapi_instrumentator import Instrumentator
    except ImportError:
        return "prometheus-fastapi-instrumentator is not installed"

    PARSE_FAILURES = Counter(
        "pitchquery_parse_failures_total",
        "Searches where core/planner.py left at least one word unplaced")
    ZERO_RESULTS = Counter(
        "pitchquery_zero_result_queries_total",
        "Searches that returned nothing")
    RANKER_USED = Counter(
        "pitchquery_ranker_used_total",
        "Searches answered, by which ranker ordered them", ["ranker"])
    RERANK_SECONDS = Histogram(
        "pitchquery_rerank_seconds",
        "Time the learned reranker spent reordering the pool",
        # Tuned to the measured range: the reranker takes ~2-10 ms over 100
        # candidates, so the default buckets would put every observation in the
        # first one and report nothing.
        buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25))

    Instrumentator(
        should_group_status_codes=False,
        excluded_handlers=["/metrics", "/health"],
    ).instrument(app).expose(app, include_in_schema=False)
    METRICS_AVAILABLE = True
    return "prometheus at /metrics"


def observe_search(*, unparsed: list, n_results: int, ranker: str,
                   rerank_ms: float = None) -> None:
    """One call site for the three counters, so they cannot drift apart."""
    if unparsed:
        PARSE_FAILURES.inc()
    if not n_results:
        ZERO_RESULTS.inc()
    RANKER_USED.labels(ranker=ranker).inc()
    if rerank_ms is not None and RERANK_SECONDS is not None:
        RERANK_SECONDS.observe(rerank_ms / 1000.0)
