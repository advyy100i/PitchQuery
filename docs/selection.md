# Which competitions PitchQuery ingests, and why

The open data has **3,961 matches across 80 competition/seasons, 426 of them
with 360 freeze-frame data** (`docs/data_inventory.md`). We do not need all of
it, and pulling all of it costs several GB of disk and hours of ingest for no
extra credibility.

## Stage A — ingest now (426 matches, ~1.5 GB raw JSON)

Every competition/season flagged for 360. These carry shot freeze frames, which
are what the context-aware xG model in §8 is built on.

| comp_id | season_id | competition | season | matches |
|--:|--:|---|---|--:|
| 43 | 106 | FIFA World Cup | 2022 | 64 |
| 72 | 107 | Women's World Cup | 2023 | 64 |
| 55 | 282 | UEFA Euro | 2024 | 51 |
| 55 | 43 | UEFA Euro | 2020 | 51 |
| 11 | 90 | La Liga | 2020/2021 | 35 |
| 9 | 281 | 1. Bundesliga | 2023/2024 | 34 |
| 7 | 235 | Ligue 1 | 2022/2023 | 32 |
| 53 | 315 | UEFA Women's Euro | 2025 | 31 |
| 53 | 106 | UEFA Women's Euro | 2022 | 31 |
| 7 | 108 | Ligue 1 | 2021/2022 | 26 |
| 44 | 107 | Major League Soccer | 2023 | 6 |

Expected scale: ~1.5M events, ~80k possessions, ~11k shots. Eleven distinct
competition/seasons means the xG split (§8 rule 1, split by competition, never
by shot) has plenty of held-out groups to choose from.

## Stage B — add only if retrieval needs volume (760 matches, ~3 GB more)

| comp_id | season_id | competition | season | matches |
|--:|--:|---|---|--:|
| 2 | 27 | Premier League | 2015/2016 | 380 |
| 11 | 27 | La Liga | 2015/2016 | 380 |

Two complete domestic seasons, no 360, but the probe run found **100% shot
freeze-frame coverage** in this sample so they are still usable for xG. Adding
them takes the corpus to ~1,190 matches and ~230k possessions, which is the
"~200k possessions" figure the resume bullet claims. Do this after Phase 3
proves retrieval works — it is a bigger index, not a better one, and it costs
ingest time.

## Deliberately skipped

- Everything else non-360 and small (historical one-off finals, 1970s matches).
  They add teams and eras that make the labelled eval set harder to write
  without adding tactical variety.
- `three-sixty/` files (10–40 MB each, ~10 GB for Stage A). Only fetched with
  `--with-360` for the optional Phase 7 tracking module. Shot freeze frames come
  inside the ordinary event files, so xG does **not** need them.
