"""Phase 7: the (query, candidate) features the learned ranker reads.

One module, imported by `models/train_ranker.py` and by `core/retrieval.py` at
serve time. That is this project's answer to training/serving skew, and it is
the whole answer — a feature store would solve the same problem with a registry,
a materialisation job and a second source of truth to keep in step. Fifty lines
that both sides import cannot disagree with themselves.

The two constants a feature can depend on — the corpus median token count, and
the feature order — are written into the model artefact at training time and
read back from it at serve time, so a corpus that grows cannot silently shift
what `n_tokens_ratio` means for a model trained before it did.

On the plan's feature list. Two of the seven it proposed are not here:

    filter match count      how many hard filters matched exactly
    competition id match    query mentioned a tournament

Both are constant within a query. Retrieval filters in SQL before it ranks, so
every candidate the ranker ever sees already satisfies every filter, and every
one is in the named competition. LambdaRank optimises the order *within* a
query group; a feature identical across a group carries no gradient and would
just be a column of ones in the importances, which reads as signal and is not.
The features that replace them — action coverage, end-zone match, and whether
the query's zones appear in order — vary across candidates, which is the
property that matters.
"""
from __future__ import annotations

# Rank features are 1-based; this is what an absent candidate gets. Large enough
# to sit well outside any pool the ranker reorders, finite so that a tree split
# on it is meaningful rather than a NaN branch.
MISSING_RANK = 1000.0

FEATURES = [
    "sparse_score",
    "sparse_rank",
    "dense_score",
    "dense_rank",
    "n_tokens",
    "n_tokens_ratio",
    "zone_coverage",
    "ordered_zone_match",
    "action_coverage",
    "end_zone_match",
    "duration_s",
]


def zone_of(token: str) -> str:
    """'CROSS@F-R>' -> 'F-R'."""
    return token.split("@", 1)[1].rstrip("+>^") if "@" in token else ""


def action_of(token: str) -> str:
    return token.split("@", 1)[0] if "@" in token else token


def reduce_path(zones: list) -> list:
    """Collapse consecutive repeats. Same rule shape search uses: dwelling in a
    zone and passing through it are the same journey."""
    out = []
    for z in zones:
        if not out or out[-1] != z:
            out.append(z)
    return out


def is_subsequence(needle: list, haystack: list) -> bool:
    i = 0
    for z in haystack:
        if i < len(needle) and z == needle[i]:
            i += 1
    return i == len(needle)


class QueryContext:
    """Everything about the query that does not change per candidate.

    Built once per search rather than per candidate: a page of 100 candidates
    would otherwise re-split the same hint string a hundred times.
    """

    def __init__(self, sequence_hint: str, median_tokens: float):
        tokens = (sequence_hint or "").split()
        self.zones = reduce_path([zone_of(t) for t in tokens if "@" in t])
        self.zone_set = set(self.zones)
        self.actions = {action_of(t) for t in tokens if "@" in t}
        self.end_zone = self.zones[-1] if self.zones else None
        # Guard against a corpus statistic of zero rather than letting it become
        # a division by zero at serve time, where there is nobody to see it.
        self.median_tokens = float(median_tokens) if median_tokens else 1.0


def row_features(row: dict, q: QueryContext, *, sparse_score: float,
                 sparse_rank: float, dense_score: float,
                 dense_rank: float) -> list:
    """One candidate's feature vector, in FEATURES order.

    `row` is a hydrated possession: the columns core.retrieval.hydrate returns.
    """
    path = reduce_path((row.get("zone_path") or "").split())
    path_set = set(path)
    actions = {action_of(t) for t in (row.get("token_string") or "").split()}

    n_tokens = float(row.get("n_events") or 0)
    covered = (len(q.zone_set & path_set) / len(q.zone_set)) if q.zone_set else 0.0
    act_cov = (len(q.actions & actions) / len(q.actions)) if q.actions else 0.0

    return [
        float(sparse_score),
        float(sparse_rank),
        float(dense_score),
        float(dense_rank),
        n_tokens,
        # The scale-free version. This is the feature that is actually supposed
        # to fix the short-possession bias: a fixed RRF has no way to express
        # "this passage is too short to be the move that was asked for", and
        # n_tokens alone would tie the model to one corpus size.
        n_tokens / q.median_tokens,
        covered,
        1.0 if (q.zones and is_subsequence(q.zones, path)) else 0.0,
        act_cov,
        1.0 if (q.end_zone and path and path[-1] == q.end_zone) else 0.0,
        float(row.get("duration_s") or 0.0),
    ]


def build_matrix(rows: list, q: QueryContext, sparse: dict, dense: dict) -> list:
    """Feature rows for a list of hydrated possessions.

    `sparse` and `dense` map uid -> (score, rank). A candidate that one ranker
    never returned gets a zero score and MISSING_RANK, which is the honest
    encoding: the ranker did not rank it, rather than ranked it last.
    """
    out = []
    for row in rows:
        uid = row["possession_uid"]
        s_score, s_rank = sparse.get(uid, (0.0, MISSING_RANK))
        d_score, d_rank = dense.get(uid, (0.0, MISSING_RANK))
        out.append(row_features(row, q, sparse_score=s_score, sparse_rank=s_rank,
                                dense_score=d_score, dense_rank=d_rank))
    return out
