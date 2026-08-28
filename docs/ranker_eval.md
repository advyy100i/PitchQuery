# Learned ranker

An `LGBMRanker` with `objective=lambdarank` reorders the top 100 candidates that sparse and dense retrieval return. It beats the fixed reciprocal rank fusion it replaces.

| | NDCG@10 |
|---|--:|
| reciprocal rank fusion | 0.4066 |
| learned ranker | **0.5401** |
| difference | +0.1335 ± 0.0898 |

Better on 23 queries, worse on 6, out of 30.

## How much to trust this

**30 training queries is a very small set for a pairwise ranker.** Evaluation is leave-one-query-out rather than a random split, because a random split over (query, candidate) pairs puts candidates from the same query on both sides of it and reports a number several times better than the truth. Even so: The paired difference clears its own 95% interval, so the direction is probably real.

The fix is more labelled queries, not a better model. `search_log` and `click_log` (Phase 8) exist to grow this set out of real use — a result clicked at rank 5 or below is a query the ranking got wrong, and those are collected for hand-grading rather than invented.

## Labels

Graded 0-3 by `eval.judge.grade`: 0 not relevant, 1 relevant, 2 relevant and produced a shot, 3 relevant and produced a goal. Because the label depends on the outcome columns, `ended_in_shot` and `ended_in_goal` are deliberately absent from the feature set — a model given them would learn the label instead of the ranking.

## Features

| feature | importance |
|---|--:|
| `duration_s` | 113 |
| `sparse_score` | 109 |
| `sparse_rank` | 97 |
| `n_tokens` | 92 |
| `action_coverage` | 64 |
| `dense_rank` | 57 |
| `end_zone_match` | 56 |
| `dense_score` | 54 |
| `zone_coverage` | 53 |
| `ordered_zone_match` | 13 |
| `n_tokens_ratio` | 12 |

Reranking 100 candidates costs 2.1 ms at p95, which is why the ranker reorders a pool rather than scoring the corpus.

Two features from the original plan are absent — `filter match count` and `competition id match`. Retrieval filters in SQL before it ranks, so both are constant across every candidate in a query and carry no gradient in a within-group objective. See `core/rank_features.py`.

Data source: StatsBomb.
