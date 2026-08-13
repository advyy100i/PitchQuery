"""Hybrid retrieval: SQL prefilter -> two rankers -> reciprocal rank fusion.

Filter first, rank second. Never the other way round: ranking the whole corpus
and then filtering means a query for one team's corners can come back with
another team's open play once the filter eats the top results.

The sparse ranker is expected to do most of the work. The token vocabulary is
controlled, not natural language, so TF-IDF over n-grams of tokens behaves like
phrase matching. The dense ranker earns its place only in fusion — Phase 3 of
the plan scores both separately so that claim is evidence rather than assertion.
"""
import pickle
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np

from core.config import EMBED_MODEL, INDEX_DIR

SPARSE_PATH = INDEX_DIR / "tfidf.pkl"

# Columns a caller is allowed to filter on. Whitelisted rather than interpolated
# so no caller — including the LLM planner in Phase 6 — can reach past them.
_EQ_FIELDS = ("team", "opponent", "competition", "season", "play_pattern",
              "start_zone", "end_zone")


@dataclass
class Filters:
    """Hard constraints applied in SQL before anything is ranked."""
    team: Optional[str] = None
    opponent: Optional[str] = None
    competition: Optional[str] = None
    season: Optional[str] = None
    play_pattern: Optional[str] = None
    start_zone: Optional[str] = None
    end_zone: Optional[str] = None
    start_band: Optional[str] = None          # 'D' | 'M' | 'F'
    end_band: Optional[str] = None
    ended_in_shot: Optional[bool] = None
    ended_in_goal: Optional[bool] = None
    min_xg: Optional[float] = None
    min_events: Optional[int] = None
    must_include: list = field(default_factory=list)   # token prefixes

    def where(self) -> tuple:
        """Returns (sql_fragment, params). Always safe to interpolate — every
        fragment is a constant, every value is a bound parameter."""
        clauses, params = [], []
        for f in _EQ_FIELDS:
            v = getattr(self, f)
            if v is not None:
                clauses.append(f"{f} = %s")
                params.append(v)
        if self.start_band:
            clauses.append("start_zone LIKE %s")
            params.append(f"{self.start_band}-%")
        if self.end_band:
            clauses.append("end_zone LIKE %s")
            params.append(f"{self.end_band}-%")
        if self.ended_in_shot is not None:
            clauses.append("ended_in_shot = %s")
            params.append(self.ended_in_shot)
        if self.ended_in_goal is not None:
            clauses.append("ended_in_goal = %s")
            params.append(self.ended_in_goal)
        if self.min_xg is not None:
            clauses.append("xg_sum >= %s")
            params.append(self.min_xg)
        if self.min_events is not None:
            clauses.append("n_events >= %s")
            params.append(self.min_events)
        for prefix in self.must_include or []:
            # Token prefixes, so 'CROSS' matches every zone and 'CROSS@F-R'
            # only the right wing. LIKE, not tsquery: the tsvector stores a
            # punctuation-folded rewrite and cannot express a prefix cleanly.
            clauses.append("token_string LIKE %s")
            params.append(f"%{prefix}%")
        return (" AND ".join(clauses) if clauses else "TRUE"), params

    def is_empty(self) -> bool:
        return self.where()[0] == "TRUE"


def rrf(*rankings: Iterable[str], k: int = 60) -> list:
    """Reciprocal rank fusion. Rank position only — the two rankers' scores are
    on incomparable scales, so fusing on rank is the honest way to combine them."""
    scores = defaultdict(float)
    for ranking in rankings:
        for rank, pid in enumerate(ranking):
            scores[pid] += 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda p: (-scores[p], p))


class Retriever:
    """Holds the sparse index in RAM. Build once, reuse for every query."""

    def __init__(self, path=SPARSE_PATH):
        with open(path, "rb") as f:
            blob = pickle.load(f)
        self.vectorizer = blob["vectorizer"]
        self.matrix = blob["matrix"]              # l2-normalised rows
        self.uids = blob["uids"]
        self.row_of = {uid: i for i, uid in enumerate(self.uids)}
        self._model = None

    # -- dense model is lazy: sparse-only work never pays the torch import ----
    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(EMBED_MODEL, device="cpu")
        return self._model

    def candidates(self, conn, filters: Filters, cap: int = 50_000):
        """uids passing the hard filters, or None when there are no filters."""
        if filters.is_empty():
            return None
        sql, params = filters.where()
        with conn.cursor() as cur:
            cur.execute(f"SELECT possession_uid FROM possessions WHERE {sql} LIMIT {cap}",
                        params)
            return {r[0] for r in cur.fetchall()}

    def sparse_rank(self, query: str, allowed=None, limit: int = 50) -> list:
        q = self.vectorizer.transform([query])
        if q.nnz == 0:                      # nothing in the query is in-vocabulary
            return []
        scores = (self.matrix @ q.T).toarray().ravel()

        if allowed is not None:
            mask = np.zeros(len(self.uids), dtype=bool)
            for uid in allowed:
                i = self.row_of.get(uid)
                if i is not None:
                    mask[i] = True
            scores = np.where(mask, scores, -1.0)

        n = min(limit, len(scores))
        top = np.argpartition(-scores, n - 1)[:n]
        top = top[np.argsort(-scores[top])]
        return [self.uids[i] for i in top if scores[i] > 0]

    def dense_rank(self, conn, query: str = None, *, vector=None,
                   filters: Filters = None, limit: int = 50,
                   exclude: str = None) -> list:
        if vector is None:
            vector = self.model.encode([query], normalize_embeddings=True)[0]
        literal = "[" + ",".join(f"{v:.6f}" for v in vector) + "]"

        sql, params = (filters or Filters()).where()
        extra = ""
        if exclude:
            extra = " AND possession_uid <> %s"
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT possession_uid FROM possessions "
                f"WHERE embedding IS NOT NULL AND {sql}{extra} "
                f"ORDER BY embedding <=> %s::vector LIMIT %s",
                params + ([exclude] if exclude else []) + [literal, limit])
            return [r[0] for r in cur.fetchall()]

    # -- shape search ---------------------------------------------------------

    def load_shapes(self, conn):
        """Cache every possession's REDUCED zone path in memory (~67k lists)."""
        if getattr(self, "_shapes", None) is None:
            with conn.cursor() as cur:
                cur.execute("SELECT possession_uid, zone_path, n_events "
                            "FROM possessions ORDER BY possession_uid")
                rows = cur.fetchall()
            self._shapes = [(uid, reduce_path(zp.split()), n) for uid, zp, n in rows]
        return self._shapes

    def by_shape(self, conn, shape: list, *, filters: Filters = None,
                 limit: int = 20) -> dict:
        """Rank possessions by how tightly they contain a drawn zone sequence.

        No text, no vectors, no model — the query is a path a person clicked on
        a pitch, matched against the `zone_path` every possession already
        carries.

        A possession qualifies if the drawn zones occur in order in its reduced
        path, and is then scored by COVERAGE — what fraction of the possession's
        whole journey the drawing accounts for:

            coverage = len(shape) / len(reduced_path)

        1.0 means the possession's entire trajectory is the shape that was
        drawn. Scoring by coverage rather than by how tightly the zones cluster
        is what stops a forty-touch possession winning because the three drawn
        zones happen to appear consecutively somewhere in the middle of it —
        which is incidental, not the shape of the move.

        It also makes the interaction self-consistent: the number of zones you
        draw sets the length of possession you get back. Three zones means a
        direct move; draw eight and you ask for a long build-up.

        Ties break on event count, so among possessions with the same journey
        the richer passage is shown first rather than a two-second fragment.
        """
        filters = filters or Filters()
        allowed = self.candidates(conn, filters)
        shapes = self.load_shapes(conn)
        shape = reduce_path(shape)

        scored = []
        for uid, path, n_events in shapes:
            if allowed is not None and uid not in allowed:
                continue
            if not is_subsequence(shape, path):
                continue
            scored.append((len(shape) / len(path), n_events, uid))

        scored.sort(reverse=True)
        return {
            "results": [uid for _t, _n, uid in scored[:limit]],
            "n_candidates": len(allowed) if allowed is not None else len(shapes),
            "n_matched": len(scored),
            "scores": {uid: round(t, 3) for t, _n, uid in scored[:limit]},
        }

    # -- the two public entry points -----------------------------------------

    def search(self, conn, *, sequence_hint: str, filters: Filters = None,
               limit: int = 20, pool: int = 50, use_dense: bool = True) -> dict:
        """`sequence_hint` is a token string — either written by hand or, in
        Phase 6, invented by the planner as an 'ideal possession'."""
        filters = filters or Filters()
        allowed = self.candidates(conn, filters)
        if allowed is not None and not allowed:
            return {"results": [], "sparse": [], "dense": [], "n_candidates": 0}

        sparse = self.sparse_rank(sequence_hint, allowed, limit=pool)
        dense = (self.dense_rank(conn, sequence_hint, filters=filters, limit=pool)
                 if use_dense else [])
        fused = rrf(sparse, dense) if dense else sparse
        return {
            "results": fused[:limit],
            "sparse": sparse[:limit],
            "dense": dense[:limit],
            "n_candidates": len(allowed) if allowed is not None else len(self.uids),
        }

    def similar(self, conn, uid: str, *, limit: int = 20, pool: int = 50,
                filters: Filters = None, use_dense: bool = True) -> dict:
        """'More like this'. Uses the seed's own vectors as the query, so it
        needs no LLM and no query language at all."""
        row = self.row_of.get(uid)
        if row is None:
            raise KeyError(f"{uid} is not in the sparse index")
        filters = filters or Filters()
        allowed = self.candidates(conn, filters)

        scores = (self.matrix @ self.matrix[row].T).toarray().ravel()
        scores[row] = -1.0                                  # never return the seed
        if allowed is not None:
            mask = np.zeros(len(self.uids), dtype=bool)
            for u in allowed:
                i = self.row_of.get(u)
                if i is not None:
                    mask[i] = True
            scores = np.where(mask, scores, -1.0)
        n = min(pool, len(scores))
        top = np.argpartition(-scores, n - 1)[:n]
        sparse = [self.uids[i] for i in top[np.argsort(-scores[top])] if scores[i] > 0]

        dense = []
        if use_dense:
            with conn.cursor() as cur:
                cur.execute("SELECT embedding FROM possessions WHERE possession_uid = %s",
                            (uid,))
                got = cur.fetchone()
            if got and got[0] is not None:
                vec = np.fromstring(got[0].strip("[]"), sep=",")
                dense = self.dense_rank(conn, vector=vec, filters=filters,
                                        limit=pool, exclude=uid)

        fused = rrf(sparse, dense) if dense else sparse
        return {"results": fused[:limit], "sparse": sparse[:limit], "dense": dense[:limit]}


def reduce_path(zones: list) -> list:
    """Collapse consecutive repeats: D-C D-C M-C M-C M-C F-C -> D-C M-C F-C.

    Dwelling in a zone for six touches and passing through it once are the same
    journey, and a drawn shape describes the journey. Reducing first is what
    makes a three-zone drawing comparable to a forty-event possession.
    """
    out = []
    for z in zones:
        if not out or out[-1] != z:
            out.append(z)
    return out


def is_subsequence(shape: list, path: list) -> bool:
    """Does `shape` occur in `path` in order, gaps allowed?"""
    i = 0
    for z in path:
        if i < len(shape) and z == shape[i]:
            i += 1
    return i == len(shape)


def hydrate(conn, uids: list) -> list:
    """Fetch display rows for a ranked uid list, preserving the ranking."""
    if not uids:
        return []
    with conn.cursor() as cur:
        cur.execute(
            "SELECT possession_uid, match_id, team, opponent, competition, season, "
            "       play_pattern, zone_path, token_string, n_events, duration_s, "
            "       ended_in_shot, ended_in_goal, xg_sum "
            "FROM possessions WHERE possession_uid = ANY(%s)", (uids,))
        cols = [d.name for d in cur.description]
        by_uid = {r[0]: dict(zip(cols, r)) for r in cur.fetchall()}
    return [by_uid[u] for u in uids if u in by_uid]
