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
                         PossessionSummary, SearchResponse, ShotComparison,
                         ShotXG)
from core import db  # noqa: E402
from core.notes import scouting_note  # noqa: E402
from core.planner import Vocabulary, plan as plan_query  # noqa: E402
from core.retrieval import Filters, Retriever, hydrate  # noqa: E402
from core.xg import UNSUPPORTED_SHOT_TYPES, XGModel  # noqa: E402

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

# The xG model. Unlike the dense ranker this one DOES fit a free tier — the
# exported artefact is 599 KB and lightgbm adds ~20 MB resident, against torch's
# several hundred. The switch exists so a memory-starved deployment has
# something to turn off before it starts dropping retrieval features.
XG_ENABLED = _flag("PITCHQUERY_XG", True)

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
    STATE["xg"], STATE["xg_status"] = _load_xg()
    yield
    STATE["conn"].close()


def _load_xg():
    """The xG model, or None and the reason why.

    Deliberately non-fatal. The model is one panel of one view; retrieval is the
    product. An API that refuses to boot because a 599 KB file is missing would
    trade a degraded feature for a total outage, so the failure is reported on
    /health and the UI says the model is unavailable rather than showing a blank
    where a number should be.
    """
    if not XG_ENABLED:
        return None, "disabled by PITCHQUERY_XG=0"
    try:
        model = XGModel.load()
        return model, f"loaded (trained with lightgbm {model.trained_with.get('lightgbm', '?')})"
    except FileNotFoundError:
        return None, "models/xg_portable.json.gz not found — run models/train_xg.py"
    except ImportError:
        return None, "lightgbm is not installed"
    except Exception as exc:                       # a corrupt or future artefact
        return None, f"{type(exc).__name__}: {exc}"


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
        xg_sum=float(row["xg_sum"] or 0.0), my_xg_sum=row.get("my_xg_sum"))


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


# The columns the xG model reads, plus the ones worth showing beside it. Listed
# explicitly for the same reason models/train_xg.py does: a `SELECT *` here is
# how statsbomb_xg would eventually find its way into the feature dict.
SHOT_COLS = """s.event_id, s.match_id, s.competition_id, s.season_id, s.team,
               s.player, s.x, s.y, s.distance, s.angle, s.body_part,
               s.technique, s.shot_type, s.first_time, s.under_pressure,
               s.play_pattern, s.is_goal, s.statsbomb_xg, s.n_def_in_cone,
               s.dist_nearest_def, s.gk_dist_to_goal, s.gk_off_line"""

# The clock lives on the event, not the shot — the shots table carries features,
# not chronology. Joined on the primary key, so it costs nothing.
SHOT_FROM = "FROM shots s JOIN events e ON e.event_id = s.event_id"


def shot_rows(conn, event_ids: list[str]) -> dict:
    """Fetch shots by event id, keyed by id."""
    if not event_ids:
        return {}
    with conn.cursor() as cur:
        cur.execute(f"SELECT {SHOT_COLS}, e.minute {SHOT_FROM} "
                    "WHERE s.event_id = ANY(%s)", (event_ids,))
        cols = [d.name for d in cur.description]
        return {str(r[0]): dict(zip(cols, r)) for r in cur.fetchall()}


def score(s: dict) -> tuple:
    """(my_xg, reason it is null). The model never guesses silently."""
    model = STATE.get("xg")
    if model is None:
        return None, STATE.get("xg_status", "model not loaded")
    value = model.predict_one(s)
    if value is not None:
        return value, None
    if s.get("shot_type") == "Penalty":
        # Not a gap in the model — an exclusion it was trained under.
        return None, "penalties are excluded from training"
    return None, "no shot geometry recorded"


def held_out(s: dict) -> Optional[bool]:
    """Was this shot's competition kept out of training?

    Worth surfacing rather than assuming. Most shots a visitor clicks come from
    competitions the model trained on, where a good number proves nothing. The
    two tournaments it never saw are where the comparison with StatsBomb is
    actually a test, and the UI should be able to say which is which instead of
    presenting every prediction as though it were out of sample.
    """
    model = STATE.get("xg")
    if model is None or s.get("competition_id") is None:
        return None
    return f"{s['competition_id']}:{s['season_id']}" in model.test_comps


def attach_my_xg(conn, rows: list) -> list:
    """Add `my_xg_sum` to each hydrated possession row, in place.

    The result cards used to show `xg_sum`, which is StatsBomb's number summed
    over the possession's shots. Displaying someone else's model beside this
    project's retrieval was confusing, so the badge shows ours instead — and
    that has to be computed, since only StatsBomb's is a stored column.

    One query and one batched predict for the whole result page. Scoring shots
    one at a time inside the loop that builds twenty cards would mean twenty
    round trips for a number nobody clicked on.

    A possession whose every shot is one the model declines (a penalty) gets
    None rather than 0.0. Zero would read as "the model thinks this was nothing",
    which is the opposite of "the model was never trained to answer".
    """
    model = STATE.get("xg")
    if model is None or not rows:
        return rows

    pairs = [(r["match_id"], int(r["possession_uid"].split(":")[1])) for r in rows]
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT e.match_id, e.possession, {SHOT_COLS} {SHOT_FROM} "
            # unnest of two parallel arrays, rather than match_id = ANY(...) AND
            # possession = ANY(...) — that form matches the cross product, so a
            # page of 20 results would pull shots from up to 400 possessions.
            "WHERE (e.match_id, e.possession) IN "
            "      (SELECT * FROM unnest(%s::bigint[], %s::int[]))",
            ([m for m, _ in pairs], [p for _, p in pairs]))
        cols = [d.name for d in cur.description]
        shots = [dict(zip(cols, r)) for r in cur.fetchall()]

    if shots:
        for s, p in zip(shots, model.predict(shots)):
            # predict() scores everything handed to it; predict_one() is what
            # knows about penalties, so the exclusion is re-applied here.
            s["_p"] = None if s["shot_type"] in UNSUPPORTED_SHOT_TYPES else float(p)

    totals: dict = {}
    for s in shots:
        if s["_p"] is None:
            continue
        uid = f"{s['match_id']}:{s['possession']}"
        totals[uid] = totals.get(uid, 0.0) + s["_p"]

    for r in rows:
        r["my_xg_sum"] = totals.get(r["possession_uid"])
    return rows


def shot_xg(s: dict) -> ShotXG:
    my_xg, note = score(s)
    return ShotXG(
        event_id=str(s["event_id"]), player=s["player"], minute=s["minute"],
        distance=s["distance"], angle=s["angle"], body_part=s["body_part"],
        shot_type=s["shot_type"], is_goal=bool(s["is_goal"]),
        statsbomb_xg=s["statsbomb_xg"], my_xg=my_xg, my_xg_note=note,
        in_holdout=held_out(s),
        n_def_in_cone=s["n_def_in_cone"], dist_nearest_def=s["dist_nearest_def"],
        gk_dist_to_goal=s["gk_dist_to_goal"], gk_off_line=s["gk_off_line"])


@app.get("/health")
def health():
    conn = dbc()
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM possessions")
        n = cur.fetchone()[0]
    return {"ok": True, "possessions": n,
            "sparse_index": len(STATE["retriever"].uids),
            "xg_model": STATE["xg"] is not None,
            # Say why, not just no. A deployment reporting false with no reason
            # is a support question; this answers it in the same response.
            "xg_status": STATE.get("xg_status", "not initialised"),
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
    rows = attach_my_xg(conn, hydrate(conn, out["results"]))
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
    rows = attach_my_xg(conn, hydrate(conn, out["results"]))
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
    rows = attach_my_xg(conn, hydrate(conn, out["results"]))
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
    rows = attach_my_xg(conn, hydrate(conn, [uid]))
    if not rows:
        raise HTTPException(404, f"no possession {uid}")
    row = rows[0]

    with conn.cursor() as cur:
        cur.execute(
            "SELECT e.event_id, e.idx, e.period, e.minute, e.second, e.type, "
            "       e.player, e.team, "
            "       e.x, e.y, e.end_x, e.end_y, e.duration, e.under_pressure, "
            "       e.possession_team, e.token, s.freeze_frame "
            "FROM events e LEFT JOIN shots s ON s.event_id = e.event_id "
            "WHERE e.match_id = %s AND e.possession = %s "
            "ORDER BY e.idx",        # index, never timestamp — it resets each period
            (row["match_id"], int(uid.split(":")[1])))
        cols = [d.name for d in cur.description]
        evs = [dict(zip(cols, r)) for r in cur.fetchall()]

    points, freeze, last_shot_id = [], [], None
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
        if e["type"] == "Shot":
            # The last shot, matching the freeze frame that gets drawn. A
            # possession can contain several — a save and a rebound — and the
            # animation ends on the final one.
            last_shot_id = str(e["event_id"])
            if e["freeze_frame"]:
                freeze = parse_freeze_frame(e["freeze_frame"])

    shot = None
    if last_shot_id:
        got = shot_rows(conn, [last_shot_id]).get(last_shot_id)
        if got:
            shot = shot_xg(got)

    return PossessionDetail(summary=summary(row), events=points,
                            freeze_frame=freeze, shot=shot)


@app.get("/shot/{event_id}", response_model=ShotComparison)
def shot(event_id: str):
    """My xG and StatsBomb's, side by side, for one shot."""
    conn = dbc()
    with conn.cursor() as cur:
        cur.execute(f"SELECT {SHOT_COLS}, e.minute, s.freeze_frame {SHOT_FROM} "
                    "WHERE s.event_id = %s", (event_id,))
        got = cur.fetchone()
        if not got:
            raise HTTPException(404, f"no shot {event_id}")
        cols = [d.name for d in cur.description]
    s = dict(zip(cols, got))

    return ShotComparison(
        **shot_xg(s).model_dump(),
        match_id=s["match_id"], team=s["team"], x=s["x"], y=s["y"],
        play_pattern=s["play_pattern"],
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
