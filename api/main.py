"""FastAPI layer over the retrieval index and the xG models.

No LLM here. Every endpoint takes structured arguments, which is what makes the
Phase 6 planner a thin translation layer rather than the product — it will call
exactly this /search with exactly these parameters.

The heavy objects (39 MB sparse index, MiniLM) load once at startup, not per
request: the first MiniLM call costs ~12 s of import and nobody should pay that
inside a query.

Run:
  uvicorn api.main:app --reload --port 8000
  open http://localhost:8000/docs
"""
import os
import pickle
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import psycopg  # noqa: E402
from fastapi import FastAPI, HTTPException, Query  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

from api.schemas import (EventPoint, FreezeFramePlayer, NoteClaim,  # noqa: E402
                         PlanResponse, PlanTerm, PossessionDetail,
                         PossessionSummary, SearchResponse, ShotComparison)
from core import db  # noqa: E402
from core.config import INDEX_DIR  # noqa: E402
from core.notes import scouting_note  # noqa: E402
from core.planner import Vocabulary, plan as plan_query  # noqa: E402
from core.retrieval import Filters, Retriever, hydrate  # noqa: E402

STATE: dict = {}

# --- deployment configuration -------------------------------------------------
# All optional: the defaults are the local development setup. A hosted instance
# sets these as environment variables.

def _flag(name: str, default: bool = True) -> bool:
    return os.getenv(name, "1" if default else "0").lower() not in ("0", "false", "no")


# Browsers block cross-origin calls, so a deployed frontend on Vercel has to be
# named explicitly. Comma-separated; the local dev server is always allowed.
CORS_ORIGINS = [o.strip() for o in os.getenv(
    "PITCHQUERY_CORS_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000").split(",") if o.strip()]

# Dense retrieval needs torch (~2.5 GB installed, several hundred MB resident),
# which does not fit a free hosting tier. Off there, on locally.
DENSE_ENABLED = _flag("PITCHQUERY_DENSE", True)

# Shape search caches 67k zone paths (~50 MB). Worth preloading on a machine
# with headroom; on a 512 MB box, let the first drawn query pay for it instead.
PRELOAD_SHAPES = _flag("PITCHQUERY_PRELOAD_SHAPES", True)

# The 15 zones of the grid, for validating a drawn shape.
VALID_ZONES = {f"{b}-{c}" for b in ("D", "M", "F")
               for c in ("L", "LI", "C", "RI", "R")}


@asynccontextmanager
async def lifespan(app: FastAPI):
    STATE["conn"] = db.connect(autocommit=True)
    # Falls back to fitting the index from the database when no pickle is
    # present, so a deployment needs no prebuilt artefact.
    STATE["retriever"] = Retriever(conn=STATE["conn"])
    if DENSE_ENABLED and STATE["retriever"].dense_available:
        STATE["retriever"].model      # pay the MiniLM import at boot, not per query
    # Team and competition names come from the data, so the parser recognises
    # exactly the entities that exist rather than a curated list that goes stale.
    STATE["vocab"] = Vocabulary.from_db(STATE["conn"])
    # Zone paths for shape search — built once so the first drawn query is as
    # fast as the hundredth.
    if PRELOAD_SHAPES:
        STATE["retriever"].load_shapes(STATE["conn"])
    try:
        with open(INDEX_DIR / "xg_models.pkl", "rb") as f:
            STATE["xg"] = pickle.load(f)
    except FileNotFoundError:
        STATE["xg"] = None            # /shot still works, just without my_xg
    yield
    STATE["conn"].close()


app = FastAPI(title="PitchQuery", version="0.1",
              description="Tactical possession search over StatsBomb open data.",
              lifespan=lifespan)

# The Next.js dev server runs on a different origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["GET"],
    allow_headers=["*"],
)


def dbc():
    """A live connection, reconnecting if the server dropped it.

    Managed Postgres suspends an idle compute (Neon does so after ~5 minutes on
    the free tier) and closes the socket. A single connection opened at startup
    therefore goes stale the first time the demo is left alone, and every
    request after that fails until the service is redeployed — the deployment
    would look fine for five minutes and be broken thereafter.

    One `SELECT 1` per request is cheap when the API and the database are in the
    same region, which is why render.yaml pins Ohio to match Neon's us-east-2.
    """
    c = STATE.get("conn")
    try:
        if c is None or c.closed:
            raise psycopg.OperationalError("no connection")
        with c.cursor() as cur:
            cur.execute("SELECT 1")
        return c
    except psycopg.Error:
        if c is not None:
            try:
                c.close()
            except Exception:
                pass
        STATE["conn"] = c = db.connect(autocommit=True)
        return c


def summary(row: dict) -> PossessionSummary:
    return PossessionSummary(
        possession_uid=row["possession_uid"], match_id=row["match_id"],
        team=row["team"], opponent=row["opponent"], competition=row["competition"],
        season=row["season"], play_pattern=row["play_pattern"],
        zone_path=row["zone_path"], token_string=row["token_string"],
        n_events=row["n_events"], duration_s=row["duration_s"] or 0.0,
        ended_in_shot=bool(row["ended_in_shot"]), ended_in_goal=bool(row["ended_in_goal"]),
        xg_sum=float(row["xg_sum"] or 0.0))


def note_for(rows: list, f: Filters, extra_given=()) -> list:
    """Scouting note over the top results, skipping whatever the query fixed."""
    given = {k for k, v in f.__dict__.items() if v not in (None, [], {})}
    given.update(extra_given)
    return [NoteClaim(**c.as_dict()) for c in scouting_note(rows[:8], given=given)]


def parse_freeze_frame(ff) -> list:
    out = []
    for p in ff or []:
        loc = p.get("location") or []
        if len(loc) < 2:
            continue
        out.append(FreezeFramePlayer(
            x=float(loc[0]), y=float(loc[1]),
            teammate=bool(p.get("teammate", False)),
            keeper=(p.get("position") or {}).get("name") == "Goalkeeper"))
    return out


@app.get("/health")
def health():
    conn = dbc()
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM possessions")
        n = cur.fetchone()[0]
    return {"ok": True, "possessions": n,
            "sparse_index": len(STATE["retriever"].uids),
            "xg_models": STATE["xg"] is not None,
            "planner": "rules",
            "teams_known": len(STATE["vocab"].teams),
            # Say which rankers are actually live rather than implying fusion.
            "rankers": "sparse+dense (fused)" if _dense_on() else "sparse only",
            "dense": _dense_on()}


def _dense_on() -> bool:
    return DENSE_ENABLED and STATE["retriever"].dense_available


def _build_plan(text: str) -> PlanResponse:
    t0 = time.perf_counter()
    p = plan_query(text, STATE["vocab"])
    took = (time.perf_counter() - t0) * 1000
    return PlanResponse(
        text=text,
        filters={k: v for k, v in p.filters.__dict__.items() if v not in (None, [], {})},
        sequence_hint=p.sequence_hint,
        terms=[PlanTerm(**t) for t in p.explain()],
        ignored=p.unmatched,
        parse_ms=round(took, 3),
    ), p


@app.get("/plan", response_model=PlanResponse)
def plan_only(q: str = Query(..., description="an English description of a passage of play")):
    """Translate English to a structured query without running it.

    Useful on its own: it is the endpoint that proves the translation is
    inspectable rather than a black box.
    """
    response, _ = _build_plan(q)
    return response


@app.get("/search", response_model=SearchResponse)
def search(
    q: Optional[str] = Query(None, description="plain English; parsed by core/planner.py"),
    sequence_hint: Optional[str] = Query(
        None, description="token string, e.g. 'CROSS@F-R> SHOT@F-C'"),
    team: Optional[str] = None,
    opponent: Optional[str] = None,
    competition: Optional[str] = None,
    season: Optional[str] = None,
    play_pattern: Optional[str] = None,
    start_band: Optional[str] = Query(None, pattern="^[DMF]$"),
    end_band: Optional[str] = Query(None, pattern="^[DMF]$"),
    end_zone: Optional[str] = None,
    must_include: Optional[list[str]] = Query(None),
    min_xg: Optional[float] = None,
    min_events: Optional[int] = None,
    ended_in_shot: Optional[bool] = None,
    ended_in_goal: Optional[bool] = None,
    limit: int = Query(20, ge=1, le=100),
    use_dense: bool = True,
):
    plan_response = None
    if q:
        plan_response, parsed = _build_plan(q)
        f = parsed.filters
        sequence_hint = sequence_hint or parsed.sequence_hint
        # Anything passed explicitly alongside `q` wins — the parser is a
        # starting point the caller can always correct.
        for name, value in (("team", team), ("opponent", opponent),
                            ("competition", competition), ("season", season),
                            ("play_pattern", play_pattern), ("start_band", start_band),
                            ("end_band", end_band), ("end_zone", end_zone),
                            ("min_xg", min_xg), ("min_events", min_events),
                            ("ended_in_shot", ended_in_shot),
                            ("ended_in_goal", ended_in_goal)):
            if value is not None:
                setattr(f, name, value)
        if must_include:
            f.must_include = must_include
    else:
        if not sequence_hint:
            raise HTTPException(400, "provide either q= (English) or sequence_hint= (tokens)")
        f = Filters(team=team, opponent=opponent, competition=competition, season=season,
                    play_pattern=play_pattern, start_band=start_band, end_band=end_band,
                    end_zone=end_zone, must_include=must_include or [], min_xg=min_xg,
                    min_events=min_events, ended_in_shot=ended_in_shot,
                    ended_in_goal=ended_in_goal)
    conn = dbc()
    t0 = time.perf_counter()
    out = STATE["retriever"].search(conn, sequence_hint=sequence_hint,
                                    filters=f, limit=limit,
                                    # The deployment flag is authoritative: with
                                    # PITCHQUERY_DENSE=0 a caller cannot force a
                                    # MiniLM load onto a box that can't afford it.
                                    use_dense=use_dense and _dense_on())
    rows = hydrate(conn, out["results"])
    return SearchResponse(
        results=[summary(r) for r in rows],
        n_candidates=out["n_candidates"],
        took_ms=round((time.perf_counter() - t0) * 1000, 1),
        sequence_hint=sequence_hint,
        filters={k: v for k, v in f.__dict__.items() if v not in (None, [], {})},
        ranker_uids={"sparse": out["sparse"][:limit], "dense": out["dense"][:limit]},
        plan=plan_response,
        note=note_for(rows, f))


@app.get("/shape", response_model=SearchResponse)
def shape(
    zones: str = Query(..., description="ordered zones, e.g. 'D-C,M-C,F-RI'"),
    team: Optional[str] = None,
    competition: Optional[str] = None,
    play_pattern: Optional[str] = None,
    ended_in_shot: Optional[bool] = None,
    ended_in_goal: Optional[bool] = None,
    min_xg: Optional[float] = None,
    limit: int = Query(20, ge=1, le=100),
):
    """Retrieve by drawn shape alone — no text, no vectors, no model.

    The query is a path clicked on the pitch, matched against the `zone_path`
    every possession already carries.
    """
    drawn = [z.strip().upper() for z in zones.split(",") if z.strip()]
    if not drawn:
        raise HTTPException(400, "give at least one zone, e.g. zones=D-C,M-C,F-C")
    bad = [z for z in drawn if z not in VALID_ZONES]
    if bad:
        raise HTTPException(400, f"unknown zone(s): {bad}. Valid: {sorted(VALID_ZONES)}")

    f = Filters(team=team, competition=competition, play_pattern=play_pattern,
                ended_in_shot=ended_in_shot, ended_in_goal=ended_in_goal, min_xg=min_xg)
    conn = dbc()
    t0 = time.perf_counter()
    out = STATE["retriever"].by_shape(conn, drawn, filters=f, limit=limit)
    rows = hydrate(conn, out["results"])
    return SearchResponse(
        results=[summary(r) for r in rows],
        n_candidates=out["n_matched"],
        took_ms=round((time.perf_counter() - t0) * 1000, 1),
        sequence_hint=" → ".join(drawn),
        filters={k: v for k, v in f.__dict__.items() if v not in (None, [], {})},
        ranker_uids={},
        # The drawn shape fixes where the move ends, so don't restate it.
        note=note_for(rows, f, extra_given={"end_zone"}))


@app.get("/similar/{uid:path}", response_model=SearchResponse)
def similar(uid: str, limit: int = Query(20, ge=1, le=100), use_dense: bool = True):
    """'More like this' — no query language and no LLM, just the seed's vectors."""
    conn = dbc()
    t0 = time.perf_counter()
    try:
        out = STATE["retriever"].similar(conn, uid, limit=limit,
                                         use_dense=use_dense and _dense_on())
    except KeyError:
        raise HTTPException(404, f"{uid} is not in the index")
    rows = hydrate(conn, out["results"])
    return SearchResponse(
        results=[summary(r) for r in rows],
        n_candidates=len(STATE["retriever"].uids),
        took_ms=round((time.perf_counter() - t0) * 1000, 1),
        sequence_hint=f"more like {uid}", filters={"seed": uid},
        ranker_uids={"sparse": out["sparse"][:limit], "dense": out["dense"][:limit]},
        note=note_for(rows, Filters()))


@app.get("/possession/{uid:path}", response_model=PossessionDetail)
def possession(uid: str):
    """Everything the animator needs: ordered events with coordinates."""
    conn = dbc()
    rows = hydrate(conn, [uid])
    if not rows:
        raise HTTPException(404, f"no possession {uid}")
    row = rows[0]

    with conn.cursor() as cur:
        cur.execute(
            "SELECT e.idx, e.period, e.minute, e.second, e.type, e.player, e.team, "
            "       e.x, e.y, e.end_x, e.end_y, e.duration, e.under_pressure, "
            "       e.possession_team, e.token, s.freeze_frame "
            "FROM events e LEFT JOIN shots s ON s.event_id = e.event_id "
            "WHERE e.match_id = %s AND e.possession = %s "
            "ORDER BY e.idx",        # index, never timestamp — it resets each period
            (row["match_id"], int(uid.split(":")[1])))
        cols = [d.name for d in cur.description]
        evs = [dict(zip(cols, r)) for r in cur.fetchall()]

    points, freeze = [], []
    for e in evs:
        attacking = e["team"] == e["possession_team"]
        # StatsBomb records EVERY event as if the acting team attacks left->right.
        # So a defender's interception is stored in the defending team's frame and
        # has to be mirrored to sit on the same pitch as the attack. Verified on
        # 3857266:137, where an interception at (27.6, 8.8) flips to (92.4, 71.2)
        # — a yard from the attacking receipt it interrupted. Plotted raw it would
        # appear in the opposite corner.
        x, y, ex, ey = e["x"], e["y"], e["end_x"], e["end_y"]
        if not attacking:
            x = None if x is None else 120.0 - x
            y = None if y is None else 80.0 - y
            ex = None if ex is None else 120.0 - ex
            ey = None if ey is None else 80.0 - ey
        points.append(EventPoint(
            idx=e["idx"], period=e["period"] or 1, minute=e["minute"] or 0,
            second=e["second"] or 0, type=e["type"], player=e["player"], team=e["team"],
            is_attacking=attacking, x=x, y=y, end_x=ex, end_y=ey,
            duration=e["duration"], under_pressure=bool(e["under_pressure"]),
            token=e["token"] if attacking else None))
        if e["type"] == "Shot" and e["freeze_frame"]:
            freeze = parse_freeze_frame(e["freeze_frame"])

    return PossessionDetail(summary=summary(row), events=points, freeze_frame=freeze)


@app.get("/shot/{event_id}", response_model=ShotComparison)
def shot(event_id: str):
    """My xG and StatsBomb's, side by side, for one shot."""
    with dbc().cursor() as cur:
        cur.execute(
            "SELECT event_id, match_id, team, player, x, y, distance, angle, body_part, "
            "       technique, shot_type, play_pattern, first_time, under_pressure, "
            "       is_goal, statsbomb_xg, freeze_frame, n_def_in_cone, dist_nearest_def, "
            "       gk_dist_to_goal, gk_off_line "
            "FROM shots WHERE event_id = %s", (event_id,))
        got = cur.fetchone()
        if not got:
            raise HTTPException(404, f"no shot {event_id}")
        cols = [d.name for d in cur.description]
    s = dict(zip(cols, got))

    my_xg = None
    if STATE["xg"] is not None and s["shot_type"] != "Penalty":
        import pandas as pd

        from models.train_xg import BOOLEAN, CATEGORICAL, CONTEXT_NUMERIC, NUMERIC

        blob = STATE["xg"]
        frame = pd.DataFrame([s])
        X = frame[NUMERIC + CONTEXT_NUMERIC].astype(float).copy()
        for c in BOOLEAN:
            X[c] = int(bool(frame[c].iloc[0]))
        for c in CATEGORICAL:
            X[c] = pd.Categorical(frame[c], categories=blob["categories"][c]).codes
            X[c] = X[c].astype("category")
        X = X[blob["features"]]
        my_xg = float(blob["mine"].predict_proba(X)[0, 1])

    return ShotComparison(
        event_id=str(s["event_id"]), match_id=s["match_id"], team=s["team"],
        player=s["player"], x=s["x"], y=s["y"], distance=s["distance"], angle=s["angle"],
        body_part=s["body_part"], shot_type=s["shot_type"], play_pattern=s["play_pattern"],
        is_goal=bool(s["is_goal"]), statsbomb_xg=s["statsbomb_xg"], my_xg=my_xg,
        n_def_in_cone=s["n_def_in_cone"], dist_nearest_def=s["dist_nearest_def"],
        gk_dist_to_goal=s["gk_dist_to_goal"], gk_off_line=s["gk_off_line"],
        freeze_frame=parse_freeze_frame(s["freeze_frame"]))


@app.get("/meta")
def meta():
    """Dropdown values for the UI. Cached in memory after the first call."""
    if "meta" in STATE:
        return STATE["meta"]
    with dbc().cursor() as cur:
        # Corpus counts come from the database, so a deployment that ships a
        # subset reports what it actually has instead of a number baked into
        # the frontend at some point in the past.
        cur.execute("SELECT (SELECT count(*) FROM possessions), "
                    "       (SELECT count(*) FROM matches)")
        n_poss, n_matches = cur.fetchone()
        cur.execute("SELECT DISTINCT competition FROM possessions ORDER BY 1")
        comps = [r[0] for r in cur.fetchall()]
        cur.execute("SELECT team, count(*) FROM possessions GROUP BY 1 ORDER BY 2 DESC")
        teams = [r[0] for r in cur.fetchall()]
        cur.execute("SELECT DISTINCT play_pattern FROM possessions ORDER BY 1")
        patterns = [r[0] for r in cur.fetchall()]
    STATE["meta"] = {"competitions": comps, "teams": teams, "play_patterns": patterns,
                     "possessions": n_poss, "matches": n_matches}
    return STATE["meta"]
