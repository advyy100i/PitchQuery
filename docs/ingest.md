# Ingest — what is actually in the database

Phases 1 and 2 of the build plan. Data source: StatsBomb.

## Corpus

| | |
|---|--:|
| matches | 431 |
| events | 1,603,663 |
| shots | 11,185 |
| shots excluding penalties | 10,858 |
| possessions (>= 3 tokens) | 66,817 |
| raw JSON on disk | 1,449 MB |

Full load takes ~160 s. Re-running `03_load_events.py` over the same cache
reproduces the counts exactly — every write is an upsert keyed on the StatsBomb
`id`, so the loader is safely re-runnable.

### By competition/season

Eleven 360-flagged competition/seasons from `docs/selection.md`, plus two
three-match remnants of the Phase 0 probe sample (Premier League and La Liga
2015/16). The remnants are kept because they cost nothing, but they are too
small to serve as an xG split group — use the eleven full ones.

| comp | season | matches | possessions | shots (no pens) |
|---|---|--:|--:|--:|
| Women's World Cup | 2023 | 64 | 10,977 | 1,608 |
| FIFA World Cup | 2022 | 64 | 9,773 | 1,430 |
| UEFA Euro | 2024 | 51 | 7,145 | 1,304 |
| UEFA Euro | 2020 | 51 | 7,816 | 1,234 |
| La Liga | 2020/2021 | 35 | 5,404 | 827 |
| 1. Bundesliga | 2023/2024 | 34 | 4,987 | 908 |
| Ligue 1 | 2022/2023 | 32 | 4,635 | 840 |
| UEFA Women's Euro | 2025 | 31 | 5,060 | 862 |
| UEFA Women's Euro | 2022 | 31 | 5,100 | 871 |
| Ligue 1 | 2021/2022 | 26 | 4,012 | 672 |
| Major League Soccer | 2023 | 6 | 883 | 146 |
| Premier League | 2015/2016 | 3 | 505 | 81 |
| La Liga | 2015/2016 | 3 | 520 | 75 |

## Freeze-frame coverage — better than the plan assumed

| shot type | shots | avg StatsBomb xG | goals | with freeze frame |
|---|--:|--:|--:|--:|
| Open Play | 10,493 | 0.100 | 1,077 | **100.0%** |
| Free Kick | 358 | 0.047 | 14 | **100.0%** |
| Penalty | 327 | 0.784 | 220 | 13.5% |
| Corner | 7 | 0.000 | 1 | 100.0% |

Headline coverage reads 97.5%, and the entire shortfall is penalties — which
§8 rule 2 drops anyway. **Every non-penalty shot in the corpus has a freeze
frame**, so `n_def_in_cone`, `dist_nearest_def`, `gk_dist_to_goal` and
`gk_off_line` are available on 100% of the modelling set rather than only on the
360 matches. The context-aware xG model does not have to fall back to
distance+angle for any row.

## Possession quality

Mean possession: 19.0 tokens over 23.6 s. 14.0% end in a shot, 1.57% in a goal.

The split by `play_pattern` is the real check that grouping and tokenisation are
sound — these are football-shaped numbers, not artefacts:

| play_pattern | possessions | % ending in a shot | mean xG |
|---|--:|--:|--:|
| From Corner | 3,265 | 50.5% | 0.055 |
| From Counter | 1,239 | 36.2% | 0.053 |
| From Free Kick | 9,185 | 15.7% | 0.016 |
| Other | 240 | 15.0% | 0.105 |
| Regular Play | 27,123 | 12.3% | 0.014 |
| From Throw In | 15,646 | 11.1% | 0.011 |
| From Keeper | 2,077 | 8.4% | 0.010 |
| From Goal Kick | 5,927 | 7.2% | 0.008 |
| From Kick Off | 2,115 | 5.2% | 0.006 |

Corners and counters are the dangerous restarts; goal kicks and kick-offs the
least. That ordering falling out of the data unprompted is the evidence that
`(match_id, possession)` grouping and the attacking-team filter are correct.

## Three grammar bugs the eyeball test caught

Plan §5 says to read twenty goal-ending token strings out loud before trusting
the grammar. Doing that on the first build surfaced three defects, all fixed in
`core/zones.py` before the full run:

1. **Backward passes were marked progressive.** The `+` rule was implemented as
   "the ball crosses into a different band", but the plan says *higher* band. A
   pass from `F-L` back to `M-LI` was being tagged `PASS@F-L+`. Now compares
   band indices directionally.

2. **Corners were classified `SWITCH`.** A corner travels far across y and
   barely at all along x — precisely the switch-of-play test. `SETP` was listed
   in the plan's token vocabulary but no derivation rule ever produced it, so
   every set piece fell through. Set-piece restarts (`pass.type` in Corner,
   Free Kick, Throw-in, Goal Kick, Kick Off) now emit `SETP` first, and
   `play_pattern` carries the finer distinction as a hard filter.

3. **Shots carried `+` and `>`.** A shot's `end_location` is where the ball
   finished, so those modifiers encoded whether it flew goalward — outcome
   leaking into the token, and worse, splintering the single most important
   token into variants that no longer matched each other. Shots now take `^`
   only.

Before and after on the same corner:

```
SWITCH@F-R> RECV@F-C SHOT@F-C>^ RECOV@F-C^ SHOT@F-C>^     # before
SETP@F-R>   RECV@F-C SHOT@F-C^  RECOV@F-C^ SHOT@F-C^      # after
```

The second reads as what it is: corner into the box from the right, shot,
rebound, shot again.

## Known open questions for Phase 3 tuning

- **`RECV` may be dead weight.** A completed pass implies a receipt, so
  `PASS@X RECV@Y CARRY@Y` spends three tokens on one action and inflates strings
  by roughly a third. Keeping it for now precisely so the before/after P@5 of
  removing it is a measurable result rather than a guess.
- `token_tsv` stores a punctuation-folded rewrite (`CROSS@F-R>` →
  `cross_f_r_box`) because Postgres' text-search parser splits on `@`, `-` and
  `+`. The Python TF-IDF ranker reads the original `token_string`.

## Reproducing

```powershell
docker compose up -d db
python ingest/02_fetch.py --comp 43:106 --comp 72:107 --comp 55:282 --comp 55:43 `
  --comp 11:90 --comp 9:281 --comp 7:235 --comp 53:315 --comp 53:106 `
  --comp 7:108 --comp 44:107
python ingest/03_load_events.py --init
python ingest/04_build_possessions.py
python ingest/_shotmap.py            # -> docs/shotmap_check.png
```
