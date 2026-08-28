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
import time
from collections import defaultdict
from pathlib import Path
from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np

from core.config import EMBED_MODEL, INDEX_DIR

SPARSE_PATH = INDEX_DIR / "tfidf.pkl"

# Distinguishes "not looked for yet" from "looked for and absent", so a
# missing ranker artefact is not re-probed on every single query.
_UNSET = object()

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


def make_vectorizer():
    """The TF-IDF configuration, defined once.

    Both `ingest/05_embed.py` and the runtime fallback below build the index
    from this, because a vectorizer fitted with different settings produces a
    different vocabulary — and a query vectorised one way against a matrix
    built the other way silently returns nonsense.

    tokenizer=str.split is not optional: the default token_pattern is
    r"(?u)\\b\\w\\w+\\b", which would shred 'CROSS@F-R>' into 'cross' and 'f'
    and discard every modifier — the exact signal the grammar encodes.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    return TfidfVectorizer(
        analyzer="word",
        tokenizer=str.split,
        token_pattern=None,
        preprocessor=None,
        lowercase=False,
        ngram_range=(1, 3),
        min_df=2,
        sublinear_tf=True,
        norm="l2",
    )


class Retriever:
    """Holds the sparse index in RAM. Build once, reuse for every query."""

    def __init__(self, path=SPARSE_PATH, *, conn=None):
        if Path(path).exists():
            with open(path, "rb") as f:
                blob = pickle.load(f)
            self.vectorizer = blob["vectorizer"]
            self.matrix = blob["matrix"]          # l2-normalised rows
            self.uids = blob["uids"]
        elif conn is not None:
            # No prebuilt artefact — fit from the database instead. This is what
            # makes a deployment self-contained: the host does not need a 39 MB
            # pickle shipped alongside it, and the index can never drift from
            # the data it was built out of. Fitting 67k short documents costs a
            # couple of seconds at boot.
            self.vectorizer, self.matrix, self.uids = self._fit_from_db(conn)
        else:
            raise FileNotFoundError(
                f"no sparse index at {path}. Either run `python ingest/05_embed.py "
                f"--sparse-only`, or construct with Retriever(conn=...) to build "
                f"it from the database at startup.")
        self.row_of = {uid: i for i, uid in enumerate(self.uids)}
        self._model = None
        self._dense_ok = None
        self._shapes = None
        self._ranker = _UNSET
        self.ranker_status = "not loaded"

    @staticmethod
    def _fit_from_db(conn):
        with conn.cursor() as cur:
            cur.execute("SELECT possession_uid, token_string FROM possessions "
                        "ORDER BY possession_uid")
            rows = cur.fetchall()
        if not rows:
            raise RuntimeError("the possessions table is empty — run the ingest first")
        uids = [r[0] for r in rows]
        vec = make_vectorizer()
        return vec, vec.fit_transform([r[1] for r in rows]), uids

    @property
    def dense_available(self) -> bool:
        """Whether the MiniLM ranker can run here.

        Deployments on small free tiers skip torch entirely (it is ~2.5 GB
        installed and needs more RAM than the box has), so the dense half of
        the fusion is absent there. Everything else — sparse, shape search, the
        planner — works unchanged, and the API reports which mode it is in
        rather than pretending.
        """
        if self._dense_ok is None:
            try:
                import sentence_transformers  # noqa: F401
                self._dense_ok = True
            except Exception:
                self._dense_ok = False
        return self._dense_ok

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

    def sparse_scored(self, query: str, allowed=None, limit: int = 50) -> list:
        """[(uid, cosine similarity)], best first.

        The learned ranker in Phase 7 needs the score and not just the position:
        a candidate that wins its rank by a hair and one that wins by a mile
        occupy the same rank, and the difference is exactly what a fixed RRF
        throws away.
        """
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
        return [(self.uids[i], float(scores[i])) for i in top if scores[i] > 0]

    def sparse_rank(self, query: str, allowed=None, limit: int = 50) -> list:
        return [uid for uid, _ in self.sparse_scored(query, allowed, limit)]

    def dense_scored(self, conn, query: str = None, *, vector=None,
                     filters: Filters = None, limit: int = 50,
                     exclude: str = None) -> list:
        """[(uid, cosine similarity)], best first.

        pgvector's `<=>` is cosine DISTANCE, so the similarity is 1 minus it.
        Returning the similarity rather than the distance keeps both rankers'
        scores pointing the same way, which is one fewer sign to get wrong in
        core/rank_features.py.
        """
        if vector is None:
            vector = self.model.encode([query], normalize_embeddings=True)[0]
        literal = "[" + ",".join(f"{v:.6f}" for v in vector) + "]"

        sql, params = (filters or Filters()).where()
        extra = ""
        if exclude:
            extra = " AND possession_uid <> %s"
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT possession_uid, 1 - (embedding <=> %s::vector) "
                f"FROM possessions "
                f"WHERE embedding IS NOT NULL AND {sql}{extra} "
                f"ORDER BY embedding <=> %s::vector LIMIT %s",
                [literal] + params + ([exclude] if exclude else []) + [literal, limit])
            return [(r[0], float(r[1])) for r in cur.fetchall()]

    def dense_rank(self, conn, query: str = None, *, vector=None,
                   filters: Filters = None, limit: int = 50,
                   exclude: str = None) -> list:
        return [uid for uid, _ in self.dense_scored(
            conn, query, vector=vector, filters=filters, limit=limit, exclude=exclude)]

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

    @property
    def ranker(self):
        """The learned reranker, or None with the reason on `ranker_status`.

        Non-fatal on purpose, like the xG model: reciprocal rank fusion is a
        perfectly good default and an API that refused to start because a 38 KB
        file was missing would trade a better ranking for an outage.
        """
        if self._ranker is _UNSET:
            try:
                from core.ranker import Ranker

                self._ranker = Ranker.load()
                self.ranker_status = self._ranker.summary
            except FileNotFoundError:
                self._ranker = None
                self.ranker_status = ("models/ranker.json.gz not found — "
                                      "run models/train_ranker.py --promote")
            except Exception as exc:
                self._ranker = None
                self.ranker_status = f"{type(exc).__name__}: {exc}"
        return self._ranker

    def search(self, conn, *, sequence_hint: str, filters: Filters = None,
               limit: int = 20, pool: int = 50, use_dense: bool = True,
               use_ranker: bool = False) -> dict:
        """`sequence_hint` is a token string — either written by hand or, in
        Phase 6, invented by the planner as an 'ideal possession'.

        With `use_ranker`, the two rankers still decide which candidates are in
        play and the learned model (Phase 7) decides their order. It reorders
        the pool and never widens it: recall is fusion's job, precision at the
        top is the ranker's.
        """
        filters = filters or Filters()
        # Fall back to sparse-only rather than raising where torch is absent.
        use_dense = use_dense and self.dense_available
        allowed = self.candidates(conn, filters)
        if allowed is not None and not allowed:
            return {"results": [], "sparse": [], "dense": [], "n_candidates": 0,
                    "ranker": "none", "rerank_ms": None}

        model = self.ranker if use_ranker else None
        # The learned ranker was trained on a pool of 100 and its rank features
        # are 1-based positions within one; scoring it over a pool of 50 would
        # feed it a distribution it never saw.
        if model is not None:
            pool = max(pool, model.pool)

        sparse = self.sparse_scored(sequence_hint, allowed, limit=pool)
        dense = (self.dense_scored(conn, sequence_hint, filters=filters, limit=pool)
                 if use_dense else [])
        sparse_uids = [u for u, _ in sparse]
        dense_uids = [u for u, _ in dense]
        fused = rrf(sparse_uids, dense_uids) if dense_uids else sparse_uids

        results, rerank_ms, which = fused[:limit], None, ("fused" if dense_uids
                                                          else "sparse")
        if model is not None and fused:
            t0 = time.perf_counter()
            rows = hydrate(conn, fused[:model.pool])
            reordered = model.rerank(
                rows, sequence_hint,
                {u: (s, i + 1.0) for i, (u, s) in enumerate(sparse)},
                {u: (s, i + 1.0) for i, (u, s) in enumerate(dense)})
            rerank_ms = round((time.perf_counter() - t0) * 1000, 2)
            results = [r["possession_uid"] for r in reordered][:limit]
            which = "learned"

        return {
            "results": results,
            "sparse": sparse_uids[:limit],
            "dense": dense_uids[:limit],
            "fused": fused[:limit],
            "n_candidates": len(allowed) if allowed is not None else len(self.uids),
            "ranker": which,
            # Logged as its own number rather than folded into the total, so a
            # regression in the reranker is visible instead of being absorbed
            # into "search got slower".
            "rerank_ms": rerank_ms,
        }

    def similar(self, conn, uid: str, *, limit: int = 20, pool: int = 50,
                filters: Filters = None, use_dense: bool = True) -> dict:
        """'More like this'. Uses the seed's own vectors as the query, so it
        needs no LLM and no query language at all."""
        row = self.row_of.get(uid)
        if row is None:
            raise KeyError(f"{uid} is not in the sparse index")
        use_dense = use_dense and self.dense_available
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
