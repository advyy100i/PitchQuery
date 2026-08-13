# Retrieval evaluation

30 queries over a 66,817-possession corpus. Relevance is decided by the programmatic rubrics in `eval/judge.py`, one predicate per query, translated from the written rubric in `eval/queries.yaml` before any results were seen.

**Human agreement: 62/71 = 87%** on a blind 71-item audit (`eval/audit.py`) — the auditor was never shown what the predicate decided. Of the 9 disagreements, 6 were the rubric refusing something the human accepted and 3 the reverse. The rubrics are net **stricter** than the human, so precision here is a floor rather than an inflated figure.

Data source: StatsBomb.

## Headline

Excluding the 5 queries whose SQL filter alone already satisfies the rubric for >=90% of rows it returns (q04, q05, q08, q24, q30) — ranking cannot affect those, so including them only inflates the mean. **25 discriminating queries:**

| ranker | P@5 | P@10 | MRR | p50 latency | p95 latency |
|---|--:|--:|--:|--:|--:|
| sparse | **0.544** | 0.540 | 0.671 | 15 ms | 48 ms |
| dense | **0.584** | 0.516 | 0.638 | 14 ms | 21 ms |
| fused | **0.608** | 0.600 | 0.751 | 36 ms | 123 ms |

All 30 queries, for completeness:

| ranker | P@5 | P@10 | MRR |
|---|--:|--:|--:|
| sparse | 0.613 | 0.613 | 0.709 |
| dense | 0.620 | 0.563 | 0.665 |
| fused | 0.667 | 0.663 | 0.776 |

## Per query (P@5)

| id | query | relevant in corpus | sparse | dense | fused |
|---|---|--:|--:|--:|--:|
| q01 | right-wing cross into the box that ends in a shot | 1,370 | 1.0 | 0.8 | 1.0 |
| q02 | left-wing cross into the box that ends in a shot | 1,261 | 1.0 | 0.6 | 1.0 |
| q03 | switch of play from one flank to the other | 12,573 | 1.0 | 1.0 | 1.0 |
| q04 * | fast counter-attack ending in a shot | 438 | 1.0 | 1.0 | 1.0 |
| q05 * | corner headed goalward | 1,644 | 1.0 | 1.0 | 1.0 |
| q06 | through ball played in behind the defence | 1,913 | 0.2 | 0.4 | 0.2 |
| q07 | build-up from the goalkeeper that reaches the final third | 2,534 | 0.2 | 0.2 | 0.2 |
| q08 * | long possession that ends in a shot | 3,380 | 0.8 | 0.0 | 0.8 |
| q09 | high turnover in the final third leading immediately to a shot | 454 | 0.8 | 1.0 | 0.8 |
| q10 | dribble into the box | 2,606 | 0.0 | 1.0 | 0.4 |
| q11 | Barcelona working the ball into the left half-space | 570 | 0.8 | 0.0 | 0.6 |
| q12 | Paris Saint-Germain counter-attack | 65 | 0.8 | 1.0 | 1.0 |
| q13 | Bayer Leverkusen pressing high and scoring | 6 | 0.0 | 0.2 | 0.4 |
| q14 | shot from outside the box after a lay-off | 29 | 0.4 | 0.0 | 0.2 |
| q15 | overlapping run down the right ending in a cutback | 2,873 | 0.0 | 0.0 | 0.0 |
| q16 | throw-in routine that creates a shot | 945 | 0.8 | 1.0 | 0.8 |
| q17 | free kick played into the box | 3,393 | 1.0 | 1.0 | 1.0 |
| q18 | possession recycled backwards then switched and attacked again | 6,180 | 0.0 | 0.0 | 0.2 |
| q19 | central combination play through the middle third | 7,590 | 0.4 | 0.8 | 0.8 |
| q20 | goal scored from inside the six-yard box | 778 | 1.0 | 1.0 | 1.0 |
| q21 | pressure-resistant build-up under heavy pressing | 10,599 | 0.2 | 0.8 | 0.8 |
| q22 | direct long ball from defence into the final third | 2,243 | 1.0 | 0.6 | 0.8 |
| q23 | England attacking the left channel in the final third | 264 | 0.6 | 0.6 | 0.6 |
| q24 * | Spain patient possession ending in a shot | 145 | 1.0 | 1.0 | 1.0 |
| q25 | shot from a rebound after a blocked attempt | 796 | 1.0 | 1.0 | 1.0 |
| q26 | attack down the right that ends with a shot from the right half-space | 1,133 | 0.2 | 0.8 | 0.4 |
| q27 | interception in midfield turned into an immediate attack | 291 | 0.6 | 0.4 | 0.6 |
| q28 | cutback from the byline | 2,766 | 0.4 | 0.4 | 0.2 |
| q29 | quick one-two through the inside-left channel | 11,782 | 0.2 | 0.0 | 0.2 |
| q30 * | high-xG chance created from open play | 276 | 1.0 | 1.0 | 1.0 |

`*` = filter-dominated, excluded from the headline mean.
