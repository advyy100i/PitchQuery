# PitchQuery — the complete plain-English guide

Everything in this project, explained simply, with the *why* behind every choice.
Read top to bottom once, then use the "Questions they will ask" section at the end
as a drill.

---

## Contents

1. [The 60-second pitch](#1-the-60-second-pitch)
2. [The problem, in plain words](#2-the-problem-in-plain-words)
3. [Jargon cheat sheet](#3-jargon-cheat-sheet)
4. [The whole system in one flow](#4-the-whole-system-in-one-flow)
5. [The data: where it comes from](#5-the-data-where-it-comes-from)
6. [The ingest pipeline, step by step](#6-the-ingest-pipeline-step-by-step)
7. [The token grammar — the central idea](#7-the-token-grammar--the-central-idea)
8. [The database](#8-the-database)
9. [Search: how results are found and ranked](#9-search-how-results-are-found-and-ranked)
10. [Shape search — drawing instead of typing](#10-shape-search--drawing-instead-of-typing)
11. [The English parser (the "planner")](#11-the-english-parser-the-planner)
12. [Scouting notes with citations that cannot lie](#12-scouting-notes-with-citations-that-cannot-lie)
13. [The xG model](#13-the-xg-model)
14. [Shipping the model as arithmetic, not a pickle](#14-shipping-the-model-as-arithmetic-not-a-pickle)
15. [How everything was measured](#15-how-everything-was-measured)
16. [The API](#16-the-api)
17. [The frontend](#17-the-frontend)
18. [Deployment](#18-deployment)
19. [Tests](#19-tests)
20. [Known problems and how to talk about them](#20-known-problems-and-how-to-talk-about-them)
21. [Numbers to memorise](#21-numbers-to-memorise)
22. [Questions they will ask, and how to answer](#22-questions-they-will-ask-and-how-to-answer)
23. [If they ask "what next?"](#23-if-they-ask-what-next)

---

## 1. The 60-second pitch

> PitchQuery is a search engine for football moves. There are 66,817 passages of
> play in it, taken from 431 real matches. You either describe a passage in
> ordinary English — "a cross from the right that ends in a shot" — or you draw
> its path on a picture of a pitch by clicking zones. It gives you back the best
> matching passages, ranked, and plays each one as a little animation. It also
> writes a short summary of the results where every sentence links to the exact
> clips it was calculated from, and it shows my own expected-goals model's number
> on any move that ends with a shot.
>
> The interesting part isn't the football. It's that I turned each passage of
> play into a single short line of text using a small invented language, which
> made ordinary text-search techniques work on football. And that everything I
> claim about it has a measured number behind it, including the parts that turned
> out worse than I expected.

If they only let you say three sentences, say the bolded ideas:

- **A possession is compressed into one line of a made-up language, and that line is what gets searched.**
- **Hard filters run in the database first, ranking happens second.**
- **Every claim in the README has an evaluation script behind it, including the failures.**

---

## 2. The problem, in plain words

Football "event data" is a giant spreadsheet. Each row is one thing that
happened: a pass at these coordinates at this second, a carry, a shot. A single
match is roughly 3,700 rows. The whole project has **1.6 million** of them.

An analyst does not think in rows. They think in **passages**: "playing out from
the back under pressure", "a counter-attack that ends in a shot", "a cutback from
the byline". There is no way to type that into a spreadsheet and get answers.

So the project makes **the passage the thing you search for**, not the row.

A "possession" is football's natural unit for this: the stretch of time one team
has the ball. StatsBomb already numbers these in the data (`possession: 47`), so
we group by `(match_id, possession)` and get 66,817 passages.

---

## 3. Jargon cheat sheet

You will need these words. Each one is defined in one sentence you can say out
loud.

| Term | Plain meaning |
|---|---|
| **Event data** | One row per thing that happened in a match (pass, carry, shot), with coordinates and a timestamp. |
| **Possession** | A stretch of play where one team has the ball. Our unit of search. |
| **Token** | One short code for one event, e.g. `CROSS@F-R>`. |
| **Token string** | All the tokens of one possession joined with spaces. This is the "document" we search. |
| **Corpus** | The whole collection of documents — here, all 66,817 token strings. |
| **Retrieval** | Finding and ordering the documents that best match a query. Search. |
| **Filter** | A hard yes/no condition, e.g. `team = 'Barcelona'`. Either a row passes or it is thrown away. |
| **Ranking** | Putting the surviving rows in order of how well they match. |
| **TF-IDF** | A classic way of scoring text overlap: rare words that appear in both query and document count for a lot, common words count for little. |
| **n-gram** | A run of N words in a row. "cross then receive then shot" is a 3-gram. Lets matching care about *order*, not just presence. |
| **Sparse ranker** | The TF-IDF one. Called sparse because most words are absent from most documents, so the matrix is mostly zeros. |
| **Embedding / vector** | A list of numbers that represents a piece of text, made by a small neural model. Similar meanings land near each other. |
| **Dense ranker** | The embedding one. Called dense because every one of the 384 numbers has a value. |
| **MiniLM** | The small pre-trained model that produces those 384 numbers. Runs on CPU, no API key. |
| **pgvector** | A Postgres extension that lets the database store vectors and find nearest ones. |
| **Cosine similarity** | How close two vectors point in the same direction. The usual "how similar" score. |
| **RRF (reciprocal rank fusion)** | A way to merge two ranked lists using only positions, not scores. |
| **P@5** | Precision at 5: of the top 5 results, what fraction were relevant. 0.6 means 3 out of 5. |
| **MRR** | Mean reciprocal rank: 1 divided by the position of the first correct result, averaged. 1.0 = first result always right, 0.5 = second, 0.33 = third. |
| **xG (expected goals)** | The probability a given shot becomes a goal, from 0 to 1. A tap-in might be 0.7, a 30-yard effort 0.02. |
| **Log-loss** | A score for probability predictions where confident-and-wrong is punished hard. Lower is better. |
| **Calibration** | Whether the numbers mean what they say: of all shots you called 0.10, roughly 10% should actually go in. |
| **Freeze frame** | StatsBomb gives the positions of every visible player at the moment of a shot. Real positions, not guessed. |
| **Holdout** | Data deliberately kept out of training so you can test honestly. |
| **Pickle** | Python's way of saving an object to a file. Convenient, but fragile across library versions. |

---

## 4. The whole system in one flow

```
StatsBomb open data (public JSON on GitHub, no key needed)
        │
        │  downloaded once, cached to disk (~1.4 GB)
        ▼
INGEST (offline scripts, run once)
   fetch → load into Postgres → group into possessions → build search indexes
        │
        ▼
POSTGRES 16 + pgvector
   431 matches · 1.6M events · 11,185 shots · 66,817 possessions
        │
        ├──────────────► RETRIEVAL: filter in SQL → rank in Python → fuse
        │                   (three ways in: English, tokens, drawn shape)
        │
        └──────────────► MODELLING: xG model trained offline, exported small
        │
        ▼
FASTAPI  (/search /shape /plan /similar /possession /shot /meta /health)
        │
        ▼
NEXT.JS frontend — animated SVG pitch, shape picker, scouting note, xG panel
```

Live: frontend on **Vercel**, API on **Render**, database on **Neon**.

---

## 5. The data: where it comes from

**StatsBomb Open Data** — a public GitHub repository of real match event data,
free to use with credit. No API key, no cost, no scraping.

What was taken: **431 matches** across 13 competition/seasons, chosen to be
varied — men's and women's, tournaments and leagues:

- FIFA World Cup 2022 (64 matches)
- Women's World Cup 2023 (64)
- UEFA Euro 2024 (51) and Euro 2020 (51)
- Women's Euro 2025 (31) and Women's Euro 2022 (31)
- La Liga 2020/21 (35), Bundesliga 2023/24 (34)
- Ligue 1 2022/23 (32) and 2021/22 (26)
- MLS 2023 (6), plus 3 Premier League and 3 La Liga 2015/16 matches left over
  from early testing

**Why a mix and not just one league:** if the whole corpus were one league, both
the search and the xG model would learn that league's style and nothing else.
Mixing men's and women's football and tournaments with leagues means the numbers
aren't a fit to a single style.

### Assumptions were checked, not trusted

Before building anything, `ingest/01_probe_assumptions.py` verified six things
against 12 sample matches, and the results are written down in `docs/probes.md`:

| Assumption | What was found |
|---|---|
| The pitch is 120 × 80 with the origin at the top-left | Confirmed over 43,014 events |
| The team acting always attacks left→right, in both halves | Confirmed — 100% of 289 shots had x > 60 |
| Every event has a possession number and a possession team | Confirmed, zero missing |
| Shots have an xG value and usually a freeze frame | Confirmed, 100% of both in the sample |
| `index` is the true ordering, not the timestamp | **Important find:** the timestamp resets to 00:00 each half, so sorting by time scrambles the match. Everything sorts by `index`. |

**Why this matters in an interview:** it shows the habit of testing the boring
assumptions. The timestamp one is a real bug that would have silently corrupted
every possession in the second half.

---

## 6. The ingest pipeline, step by step

Five numbered scripts. They run once, offline, and they are all safe to re-run.

### `00_inventory.py` — what is available
Lists all competitions and seasons StatsBomb offers, so the selection is made
from facts rather than memory.

### `01_probe_assumptions.py` — verify the assumptions
The table above. Writes `docs/probes.md`.

### `02_fetch.py` — download
Pulls the JSON files over HTTPS and caches them on disk. Skips anything already
downloaded, so an interrupted run resumes instead of starting over.

**Why cache to disk:** the pipeline can be re-run offline forever afterwards, and
StatsBomb's servers get hit once.

### `03_load_events.py` — load into Postgres
Flattens each event into a row: coordinates, type, minute, player, team,
possession number. Three things happen here that matter:

1. **The whole raw JSON is kept too**, in a `JSONB` column. Why: if a new feature
   is needed later, it can be computed without re-downloading 1.4 GB.
2. **Shot geometry is computed now, not at query time** — distance to goal, the
   angle of the visible goal mouth, and four freeze-frame features.
3. **Each event's grammar token is computed and stored now.** Why: serving never
   has to parse JSON, which keeps 1.3 GB of JSONB off the fast path and out of
   the deployed database entirely.

Every write is an **upsert** (insert, or update if the key already exists), keyed
on StatsBomb's own event id. So re-running the loader over the same cache changes
nothing. Full load: ~160 seconds.

### `04_build_possessions.py` — group into passages
Groups events by `(match_id, possession)` and builds the token string. Two rules:

1. **Only the attacking team's own events go in the string.** If the defending
   team's interception were mixed in, the sequence would read as nonsense and
   n-gram matching would break.
2. **Possessions with fewer than 3 tokens are dropped.** These are throw-in noise
   — ball goes out, ball comes back — and they would flood every result set.

Result: 66,817 possessions, averaging 19 tokens over 23.6 seconds. 14.0% end in a
shot, 1.57% in a goal.

**The sanity check that proved the grouping was right:** breaking the corpus down
by how each possession started produced football-shaped numbers without being
asked to —

| Start | % ending in a shot |
|---|--:|
| From Corner | 50.5% |
| From Counter | 36.2% |
| From Free Kick | 15.7% |
| Regular Play | 12.3% |
| From Goal Kick | 7.2% |
| From Kick Off | 5.2% |

Corners and counters are the dangerous restarts, kick-offs the least. Nobody
coded that ordering; it fell out. That is the evidence the grouping and the
attacking-team filter are correct.

### `05_embed.py` — build the two search indexes
- **Sparse:** fit TF-IDF over all 66,817 token strings, save it.
- **Dense:** run each token string through MiniLM, store the 384-number vector in
  Postgres via pgvector, then build an HNSW index over them.

**Why build the HNSW index after loading the vectors, not before:** filling an
existing index row by row is far slower than indexing an already-full table.

---

## 7. The token grammar — the central idea

**This is the part to lead with. It is the actual invention.**

Every event becomes one code of the form:

```
ACTION @ ZONE + modifiers
```

### The zones: a 3 × 5 grid = 15 zones

The pitch is 120 long and 80 wide. Cut it into 3 bands along the length (40 m
each) and 5 channels across the width (16 m each).

```
   x=0          x=40         x=80        x=120
y=0  +------------+------------+------------+
     |    D-L     |    M-L     |    F-L     |  left touchline
     +------------+------------+------------+
     |    D-LI    |    M-LI    |    F-LI    |  left half-space
     +------------+------------+------------+
     |    D-C     |    M-C     |    F-C     |  <-- attacking this goal
     +------------+------------+------------+
     |    D-RI    |    M-RI    |    F-RI    |  right half-space
     +------------+------------+------------+
     |    D-R     |    M-R     |    F-R     |  right touchline
y=80 +------------+------------+------------+
```

- **D / M / F** = defensive third, middle third, final third.
- **L / LI / C / RI / R** = left wing, left half-space, centre, right half-space,
  right wing.

**Why 15 and not 100?** This is a compression trade-off. Too fine and nothing
matches — two crosses from three yards apart would land in different zones and
look unrelated. Too coarse and everything matches. 15 lines up with how coaches
actually talk (thirds, half-spaces, wings), so a query in football language maps
cleanly onto it.

### The actions: 14 of them

`PASS · CROSS · THROUGH · SWITCH · CARRY · DRIB · SHOT · RECV · RECOV · DUEL ·
CLR · INT · LOSS · SETP`

They are derived in a fixed order — the first rule that fires wins. For a pass:

1. Was it a set-piece restart (corner, free kick, throw-in, goal kick, kick-off)?
   → `SETP`
2. Was it flagged as a cross? → `CROSS`
3. Was the technique "through ball"? → `THROUGH`
4. Did it travel 30 m+ sideways with under 20 m forward? → `SWITCH`
5. Otherwise → `PASS`

**Events that are deliberately thrown away:** `Pressure`, substitutions, tactical
shifts, camera events, half start/end. Why: `Pressure` alone is 8% of all events,
it is always the *defending* team, and it carries no information about what the
attack did. Dropping it keeps the n-grams meaningful.

### The three modifiers

| Symbol | Meaning | Rule |
|---|---|---|
| `+` | progressive | ends at least 12 m further forward, **or** crosses into a higher band |
| `>` | into the box | ends inside the 18-yard box (x ≥ 102, y between 18 and 62) |
| `^` | under pressure | StatsBomb's own `under_pressure` flag |

**Why `+` compares band *indices* and not just "changed band":** a long ball
backwards is not progress. Direction matters, so the rule checks the band index
went *up*.

### What a possession looks like

```
SETP@F-R>  RECV@F-C  SHOT@F-C^
```
A corner from the right into the box, received centrally, headed at goal under
pressure.

```
RECOV@M-C  CARRY@M-C+  PASS@F-LI+  RECV@F-C  SHOT@F-C
```
Ball won in midfield, driven forward, played into the left half-space, laid off,
shot.

### Three grammar bugs the eyeball test caught

Before trusting the grammar, twenty goal-ending strings were read out loud. Three
defects surfaced immediately:

1. **Backward passes were being marked progressive.** The rule had been written
   as "crosses into a different band" instead of "a higher band". A pass from
   `F-L` back to `M-LI` was tagged `+`. Fixed to compare band indices
   directionally.

2. **Corners were being classified as `SWITCH`.** A corner travels far sideways
   and barely at all forward — which is exactly the switch-of-play test. `SETP`
   existed in the vocabulary but no rule ever produced it, so every set piece
   fell through the cracks. Now set-piece restarts are checked first.

3. **Shots carried `+` and `>`.** A shot's end location is where the ball
   finished, so those modifiers were encoding *whether it flew towards goal* —
   the outcome leaking into the token. Worse, it splintered `SHOT@F-C` into
   variants that no longer matched each other, which damages the single most
   important token in the whole grammar. Shots now take `^` only.

Same corner, before and after:

```
SWITCH@F-R> RECV@F-C SHOT@F-C>^ RECOV@F-C^ SHOT@F-C>^     # before
SETP@F-R>   RECV@F-C SHOT@F-C^  RECOV@F-C^ SHOT@F-C^      # after
```

The second reads as what it is: corner from the right, shot, rebound, shot again.

**This story is gold in an interview.** It shows manual verification catching
three real bugs that no test would have found, because the output was
*plausible* in every case.

---

## 8. The database

PostgreSQL 16 with the pgvector extension, running in Docker locally.

Four tables:

| Table | Rows | What it holds |
|---|--:|---|
| `matches` | 431 | competition, season, date, teams, score |
| `events` | 1,603,663 | one row per event + the precomputed token + the raw JSON |
| `shots` | 11,185 | shot geometry and freeze-frame features, plus StatsBomb's xG for comparison |
| `possessions` | 66,817 | the searchable unit: token string, zone path, outcome flags, 384-dim embedding |

The `possessions` row is the important one:

```sql
possession_uid  TEXT PRIMARY KEY,   -- '3869118:47'
team, opponent, competition, season, play_pattern,
n_events, duration_s,
start_zone, end_zone,
zone_path       TEXT,   -- 'D-C D-L M-L M-C F-RI F-R F-C'
token_string    TEXT,   -- 'RECOV@D-C PASS@D-L+ ... SHOT@F-C'
token_tsv       TSVECTOR,
ended_in_shot, ended_in_goal, xg_sum,
embedding       VECTOR(384)
```

**Why the `zone_path` is stored separately from the `token_string`:** shape search
only cares about the journey, not the actions. Keeping the path as its own column
means shape search is a straight string comparison with no parsing.

**Why `statsbomb_xg` sits in `shots` but is labelled "comparison only":** it is
the yardstick the model is measured against and must never become an input. The
column comment says so, and every SELECT in the training code lists columns
explicitly so a careless `SELECT *` can't leak it in.

**The `token_tsv` oddity worth knowing:** Postgres's text search splits words on
`@`, `-` and `+`, which would shred `CROSS@F-R>` into useless fragments. So a
folded rewrite is stored instead — `cross_f_r_box_prs`. The Python ranker reads
the original string; only the Postgres full-text index uses the folded one.

---

## 9. Search: how results are found and ranked

### The one rule: filter first, rank second

```
1. SQL WHERE clause  →  the set of possession ids that qualify   (hard)
2. Rank only those                                               (soft)
```

**Why never the other way round:** if you rank the whole corpus and then filter,
a search for Barcelona's corners returns the top 50 corners overall, and then the
Barcelona filter deletes most of them, leaving three results that happen to be
Barcelona's rather than *their best* corners. Filtering first means the ranking
is always working on the right pool.

The filters are on a **whitelist** — team, opponent, competition, season, play
pattern, start/end zone, start/end band, ended in shot, ended in goal, minimum
xG, minimum events, must-include tokens. Every value is a bound parameter, never
string-interpolated, so there is no SQL injection surface even though the query
comes from user text.

### Ranker 1: sparse (TF-IDF over n-grams)

The library is scikit-learn's `TfidfVectorizer`, configured deliberately:

| Setting | Value | Why |
|---|---|---|
| `tokenizer=str.split` | split on spaces | **Critical.** The default pattern would shred `CROSS@F-R>` into "cross" and "f" and discard every modifier — the exact signal the grammar encodes. |
| `lowercase=False` | keep case | The tokens are uppercase by design. |
| `ngram_range=(1,3)` | 1-, 2- and 3-token runs | Makes *order* count. `CROSS@F-R> RECV@F-C SHOT@F-C` matching as a unit is phrase matching, not bag-of-words. |
| `min_df=2` | ignore tokens seen once | A one-off is noise. |
| `sublinear_tf=True` | dampen repeats | Ten carries in a row shouldn't score ten times as high as one. |
| `norm="l2"` | unit-length vectors | Standard — and also the cause of the known bug in §20. |

At query time, the query string is vectorised the same way and multiplied against
the whole matrix — one sparse matrix multiply, then take the top N. This is
milliseconds for 67,000 documents held in RAM.

**Why sparse was expected to win:** the token vocabulary is small and controlled
(14 actions × 15 zones × 8 modifier combinations). That is not natural language,
it is a code. Exact overlap of codes is exactly what TF-IDF is good at.

### Ranker 2: dense (MiniLM + pgvector)

Each token string is turned into 384 numbers by `all-MiniLM-L6-v2`, a small
sentence-embedding model that runs on CPU with no API key. Postgres finds the
nearest vectors with the `<=>` cosine-distance operator, accelerated by an HNSW
index.

### The result that contradicted the premise

The plan said sparse would beat dense, and dense would only earn its place in
fusion. Half right:

| ranker | P@5 | P@10 | MRR |
|---|--:|--:|--:|
| sparse (TF-IDF) | 0.544 | **0.540** | **0.671** |
| dense (MiniLM) | **0.584** | 0.516 | 0.638 |
| **fused (RRF)** | **0.608** | **0.600** | **0.751** |

Read this out loud as: *"Sparse puts its single best hit higher up — that's the
MRR. Dense is better across the top 5. They're good at different things."*

**Why fusing works so well:** the two rankers agree on only about **1 of their
top 10** results for a typical query. Two lists that overlap that little are
seeing different things, so combining them adds real information rather than
just averaging noise.

**Where dense fails, specifically:** asked for a right-wing cross it returned
`CROSS@F-L>` — a *left*-wing cross. MiniLM sees `F-L` and `F-R` as strings one
character apart and has no idea that one of them means left. That is a great
concrete answer to "what are the weaknesses of embeddings here".

### Fusion: RRF (reciprocal rank fusion)

```python
score(doc) = Σ over lists  1 / (60 + position_in_that_list)
```

**Why fuse on position and not on score:** TF-IDF cosine and vector cosine are on
different scales that mean different things. Adding them, or normalising them
into a common range, invents a comparison that isn't there. Position is the one
thing both lists genuinely agree on the meaning of. The constant 60 is the
standard value from the original RRF paper; it stops rank 1 from dominating
everything below it.

### "More like this"

`/similar/{uid}` uses the seed possession's own vectors as the query — its row
of the TF-IDF matrix and its stored embedding. No query language, no parsing, no
model call. It is the cheapest useful feature in the project.

---

## 10. Shape search — drawing instead of typing

Click zones on a pitch: `D-C → M-C → F-RI → F-C`. Get back moves that took that
journey. **No text, no vectors, no model at all.** ~30 ms across all 66,817.

How it works:

1. Take each possession's `zone_path` and **collapse consecutive repeats**:
   `D-C D-C M-C M-C M-C F-C` becomes `D-C M-C F-C`.
   **Why:** dwelling in a zone for six touches and passing through it once are
   the same *journey*, and a drawn shape describes a journey. Collapsing is what
   makes a three-zone drawing comparable to a forty-event possession.

2. Keep the possessions where the drawn zones appear **in order** (gaps allowed).

3. Rank by **coverage**:
   ```
   coverage = number of zones you drew / length of the possession's reduced path
   ```
   1.0 means the possession's entire trajectory *is* the shape you drew.

4. Ties break on event count, so the richer passage shows above a two-second
   fragment with the same journey.

**Why coverage and not clustering — this is the good story.** The first attempt
ranked by how tightly the drawn zones sat together in the possession. It was
wrong: it surfaced 100-touch possessions where three zones happened to line up
somewhere in the middle. That is incidental, not the shape of the move.

Coverage also makes the interaction self-consistent: **the number of zones you
draw sets the length of possession you get back.** Draw three zones and you're
asking for a direct move; draw eight and you're asking for a long build-up.

---

## 11. The English parser (the "planner")

The original plan called for an LLM here. It is a rule-based parser instead.

### What it does

Input: `"Barcelona working the ball into the left half-space"`

Output, in about **1 millisecond**:
- **Filters:** `{team: 'Barcelona'}`
- **Sequence hint:** `RECV@F-LI CARRY@F-LI CARRY@F-LI`
- **Explanation:** "Barcelona" → team filter; "left half-space" → LI channel hint
- **Ignored words:** *working the ball into the*

### The design rule that makes it work

> **Hard filters come only from unambiguous phrases.**
> **Channels and actions go into the hint, which only affects ranking.**

Why: a wrong filter returns an **empty result set** — total failure. A wrong hint
costs you a few places in the ordering. So team names, competitions, play
patterns, outcomes and thresholds become filters; anything fuzzier steers the
ranking only.

Concretely: "left half-space" nudges the hint towards `F-LI`. It does **not** add
`end_zone = 'F-LI'`, which would throw away every near-miss.

### How the matching works

- Vocabulary tables map canonical values to synonym sets — `"RI"` matches "right
  half-space", "right halfspace", "inside-right", "right channel"...
- **Longest phrase wins at each position.** "left half-space" beats "left". "goal
  kick" beats "goal". Without this, "goal kick" would set an `ended_in_goal`
  filter and return nothing.
- The last word of each phrase gets an optional inflection, so "finishing"
  matches "finish" and "crosses" matches "cross". Without it the parser is
  brittle in exactly the way rule systems are accused of being — "finishing from
  the inside right" produced no shot at all.
- **Team and competition names come from the database**, not a hardcoded list, so
  it recognises exactly the entities that exist and can't go stale.

### The hard case, and how it's handled

"In the final third" can mean where the move **started** or where it **ended**:

- *"high turnover **in the final third** leading to a shot"* — where it began
- *"England attacking the left channel **in the final third**"* — where it ended

The tie-breaker is the nearest preceding action. If it's a ball-winning verb
(recovery, interception, duel, clearance), the band describes where possession
*started*. Otherwise it describes where the ball *arrived*.

### Why the hint gets padded out

A hint of one token retrieves badly. The corpus averages 19 tokens per
possession, so a lone `INT@M-C` has almost nothing to overlap with. The parser
therefore expands phrases into several beats, and inserts connective `RECV` and
`CARRY` tokens — because in the real data a completed pass is always followed by
a receipt, and a hint without them looks nothing like a real possession.

There is also an honest negative result recorded in the code: padding with an
*arriving pass* instead was tried, measured **worse** (P@5 0.680 → 0.592), and
reverted rather than kept on intuition.

### Why rules instead of an LLM — the four reasons

1. **Free.** No API key, no cost, no rate limit, the whole system is
   self-contained.
2. **Deterministic.** The same sentence always produces the same query. So a
   retrieval regression is always the *retrieval's* fault, never the planner
   behaving differently today.
3. **Inspectable.** Every filter is traceable to the exact phrase that produced
   it, and the UI shows this — including **which words it failed to understand**.
   That turns "here's your translated query" into an honest feature rather than
   decoration.
4. **Testable.** Its quality is a measured number against 30 hand-written
   queries, not a vibe.

### The honest cost

**It only knows the vocabulary written into it.** There is no alias table, so
"PSG" is not a team. That failure is kept in the evaluation set on purpose rather
than quietly dropped — `docs/planner_eval.md` lists it under "known limits,
included deliberately".

### The numbers

- **Parse cost: 1.13 ms average, 12.89 ms worst case.**
- **Filter agreement with hand-written queries: 24 of 30 exact.**
- Retrieval quality: parsed queries score P@5 **0.680** against hand-written
  0.608 — but that number is **not comparable**, and the doc says so.

**Why not comparable, and this is the part that shows judgement:** where the two
sides produce *different* filters, they aren't ranking the same pool, and the
rubric can't always tell which choice was right. On q07 the parser reads "from
the goalkeeper" as `From Keeper` where the hand-written query chose `From Goal
Kick` — and that query's rubric only tests whether the ball reaches the final
third, so it's blind to the distinction. The parser scores 1.0 against 0.2 on
something the judge cannot see.

**So the figure actually quoted** is restricted to the 19 queries where both
sides produced *identical* filters, meaning only the ranking differs:
**hand 0.632, parsed 0.611.** Slightly worse, honestly reported.

### Does it generalise, or is it fitted to the 30 sentences?

The rules were written while looking at failures on those 30 sentences, so that
headline is an **in-sample** number — which is a real problem and is stated as
one. So 23 **paraphrases** were written, restating the same intents in
deliberately different words, judged by the original rubrics:

**23 paraphrases: P@5 0.748, versus 0.730 for the original wording.** It holds
up. That is the answer to "isn't a rule parser just overfitted to your test set?"

---

## 12. Scouting notes with citations that cannot lie

Above the results, the app writes 3–5 sentences about what you got back:

> *8 of the 12 come from Spain.*
> *5 begin from a corner.*
> *3 of the 12 are scored, from 0.31 xG.*
> *They are short and direct: a median of 6 touches over 9 seconds.*

Each sentence is **clickable** — it links to the exact clips it describes.

### The guarantee

The original plan was: ask an LLM to write sentences, require it to cite
possession ids, then verify the citations and drop the sentences that fail.

This does the same job **by construction instead of by inspection**:

```python
goals = [r for r in rows if r["ended_in_goal"]]
Claim(f"{len(goals)} of the {n} are scored",
      uids=[r["possession_uid"] for r in goals])
```

The claim and its evidence are produced by the **same expression**. There is no
verification step because there is nothing that can disagree.

**Say it like this:** *"The checker exists to catch a generator that can lie. This
generator cannot lie, so it's a stronger guarantee than an LLM plus a checker,
not a weaker one."*

### The bug this caught while it was being written

A sentence counting **goals** was citing **every shot**. The number was right,
the evidence was wrong, and it looked completely fine on screen. Now each
sentence has exactly one subject, and `tests/test_notes.py` enforces it — one
test asserts that a claim about goals cites only goals, another that a claim
about one team cites only that team's rows.

### The other nice detail: no echoing the query

If you filtered on "ends in a shot", being told "8 of 8 end in a shot" is not an
observation, it is an echo. So the note is passed the set of fields the query
already fixed, and it either skips them or narrows to the part that still carries
information — the *goals* within the shots, or the *average xG* rather than the
count.

### The honest trade

It only says what these functions know how to say. It will never notice something
surprising in the data the way a model might. That's the price of the guarantee.

---

## 13. The xG model

### What xG is

The probability that a shot becomes a goal. A tap-in might be 0.7; a speculative
30-yarder 0.02. It is the standard way football analytics measures chance
quality.

### The three models compared

| | What it is |
|---|---|
| **baseline** | Logistic regression on distance and angle only. The floor. |
| **mine** | LightGBM (gradient-boosted trees) on geometry + situation + freeze-frame context. |
| **StatsBomb** | Their shipped production model. The ceiling. Not trained here. |

### The features (12 of them)

**Geometry (2):**
- `distance` — metres from the shot to the centre of the goal
- `angle` — how wide the goal mouth appears from where the shooter stands, in
  radians, computed with the cosine rule on the triangle (shot, near post, far
  post). Wider visible goal = better chance.

**Freeze-frame context (4)** — from the real player positions at the moment of
the shot:
- `n_def_in_cone` — how many defenders are inside the triangle between the ball
  and the two posts. That is what physically blocks a shot. Computed with a
  barycentric sign test (a standard point-in-triangle check).
- `dist_nearest_def` — how close the nearest defender is (keeper excluded)
- `gk_dist_to_goal` — how far the keeper is from the goal centre
- `gk_off_line` — how far the keeper has come off his line

**Situation (6):** `body_part`, `technique`, `shot_type`, `play_pattern`,
`first_time`, `under_pressure`.

**A genuinely lucky finding:** freeze-frame coverage was expected to be limited
to the 360-flagged matches. It turns out **every single non-penalty shot in the
corpus has a freeze frame** — 100% of open play, 100% of free kicks. The headline
coverage reads 97.5% and the entire shortfall is penalties, which are dropped
anyway. So the context features are available on 100% of the modelling set and
the model never has to fall back to distance+angle.

### The four rules, enforced in code

**Rule 1 — split by competition, never by shot.**
Whole competition/seasons are held out: the **2022 World Cup** and the **2023
Women's World Cup**, 3,038 shots. Trained on 7,820 shots across 11 other
competition/seasons.

*Why this matters more than anything else here:* shots inside one match share a
game state, a pitch, a team and often the same shooter. If you split randomly by
shot, shot #3 and shot #4 from the same attack end up on opposite sides of the
split, the model has effectively seen the test data, and every metric is
inflated. There's an `assert` in the training script that fails if any group
appears on both sides.

**Rule 2 — penalties and own goals are dropped.**
A penalty is fixed geometry with a ~78% conversion rate. Including them inflates
AUC enormously for no skill at all — the model just learns "penalty = 0.78".

**Rule 3 — `statsbomb_xg` is never a feature.**
It's the thing being compared against. The training SELECT lists every column
explicitly precisely so that a later `SELECT *` cannot leak it in.

**Rule 4 — nothing is fitted on the test competitions**, including the category
encodings. A body part appearing only in the held-out data stays "unknown" rather
than silently becoming a new column and shifting everything else along.

### Training, and the calibration step

LightGBM: 300 trees, learning rate 0.05, 15 leaves, `min_child_samples=60`,
subsampling 0.9, L2 regularisation 5.0.

Raw gradient boosting on 7,800 shots at a 10% base rate comes out **sharp but
over-confident**: it ranks chances well (good AUC) while getting the *level*
wrong. Log-loss charges for exactly that.

So the probabilities are **calibrated** — and the choice of calibrator is made by
5-fold grouped cross-validation **on the training competitions only**, comparing
three candidates (none, isotonic, sigmoid). Sigmoid won. The held-out
tournaments were never involved in that decision.

**Missing values are not imputed.** LightGBM routes NaN natively, and filling in
a mean would invent defenders who weren't on the pitch.

### The results

| model | log-loss ↓ | Brier ↓ | ROC-AUC ↑ | expected goals | observed | O/E | gap closed |
|---|--:|--:|--:|--:|--:|--:|--:|
| distance+angle | 0.2811 | 0.0786 | 0.7330 | 322.4 | 288 | 0.893 | 0% |
| **mine (+context)** | **0.2581** | 0.0727 | 0.7933 | 297.8 | 288 | 0.967 | **76%** |
| StatsBomb | 0.2507 | 0.0693 | 0.8035 | 282.8 | 288 | 1.018 | 100% |

**The "gap closed" figure, explained simply:**

```
gap closed = (baseline_loss − my_loss) / (baseline_loss − statsbomb_loss)
```

*"Of the distance the simple baseline would have to travel to reach StatsBomb's
production model, mine covers 76% of it."*

**Always phrase it that way. Never say "beats StatsBomb."** The script even
checks whether StatsBomb actually beats the baseline on this split before
reporting the ratio at all — if it didn't, the ratio would be meaningless and the
script says so instead of printing a number.

**Why log-loss is the headline and not AUC:** AUC only cares about *ordering*. A
model can rank every shot perfectly and still be systematically wrong about how
many goals to expect. For xG the level matters — that's what the O/E column and
the calibration plot show. My model's O/E is 0.967, meaning it expected 297.8
goals where 288 happened; the baseline expected 322.4.

---

## 14. Shipping the model as arithmetic, not a pickle

**This is the strongest engineering story in the project. Have it ready.**

### The problem

Training produces a Python pickle: a `CalibratedClassifierCV` wrapping three
`LGBMClassifier` objects, fitted on a pandas DataFrame with categorical dtypes.

Deploying that pickle means:
- pinning scikit-learn **and** pandas on the server to versions close enough to
  whatever the laptop had, forever;
- nothing warns you when they drift — you get an import error, or worse, silently
  different behaviour;
- dragging pandas onto the box: ~70 MB resident, to score twelve numbers.

### The solution

Re-serialise the model into **what it actually is**:

```
p(goal) = mean over the 3 members of  1 / (1 + exp(a·margin + b))
```

Three gradient-boosted ensembles in **LightGBM's own documented text format**,
plus **two floats** per member for the sigmoid, plus a lookup table for the
categories.

Result: **599 KB gzipped, committed to the repo, scored with numpy.** Serving it
costs **7 MB** of memory. No scikit-learn, no pandas, no version lock — LightGBM's
text format is a versioned interchange format that newer releases read, unlike a
pickle which is a snapshot of Python objects.

### The trap this walked into

The translation is hand-written, so it must be verified rather than trusted.
`tests/test_xg_portable.py` scores **all 10,858 non-penalty shots both ways and
requires exact equality** — not a tolerance, because it should be the same
arithmetic on the same floats.

That test earns its place, and here is why:

> scikit-learn calibrates whatever `decision_function` returns. For LightGBM's
> classifier, that is the **raw margin** — not a probability.

Calibrating the probability instead — the obvious reading, and the first thing
that was implemented — produces values wedged **between 0.43 and 0.62**. Every
single one of them looks like a plausible xG on a page. The service returns 200.
Nothing crashes. And the model is wrong on every shot.

**A smoke test passes. An equality test over 10,858 shots does not.**

There is also a guard in the exporter: if the cross-validation ever picks
*isotonic* calibration instead of sigmoid, that's a step function per member
which this exporter cannot represent — so it **fails loudly** rather than
exporting something that scores differently from what was evaluated.

And `train_xg.py` writes both artefacts in the **same run**, chained together, so
the deployed model cannot drift from the one the benchmark page reports.

### How it's shown in the UI

- Only **my** number is shown on a clip, not StatsBomb's side by side.
  **Why:** a single shot cannot settle which model is better. A 0.15 chance going
  in is not evidence against 0.15. That question needs thousands of shots and a
  log-loss, and it lives in the benchmark doc. Two numbers next to one clip turns
  every clip into a scoreboard about the wrong thing.
- The bar runs **0 to 1**, because that is what a probability is. Rescaling to
  make small chances look bigger would be flattering the number rather than
  reporting it.
- A **badge** says whether that competition was held out of training or not.
  **Why:** most shots a visitor clicks come from competitions the model learned
  from, where a good number proves nothing. The two tournaments it never saw are
  where the comparison is actually a test, and the UI should say which is which
  instead of presenting every prediction as though it were out of sample.
- For a **penalty** it shows *"penalties are excluded from training"* rather than
  a number. The model declines instead of extrapolating.

---

## 15. How everything was measured

### The evaluation set

30 queries in `eval/queries.yaml`. Each one carries three things:

- `filters` — the hard SQL constraints
- `sequence_hint` — a hand-written "ideal possession" in the token grammar
- `rubric` — written English describing what should count as relevant

**The rubric was written before any results were seen.** That is the whole point.
Judging against a rubric you wrote after looking at the output is how you talk
yourself into a good P@5.

### Rubrics as code

`eval/judge.py` translates each written rubric into a **predicate** — a small
function that takes a possession and returns relevant / not relevant.

Example, for "right-wing cross into the box that ends in a shot":

```python
def q01(p):
    """Cross from F-R/F-RI into the box, shot afterwards."""
    return cross_then_shot(p, "CROSS@F-R")
```

**Why do this instead of hand-judging?**
- Reproducible — the label doesn't drift as a human gets tired at query 22.
- Applies to **all 66,817 possessions**, not just a pooled sample. So there are
  no unjudged results, and recall against the full corpus is computable.

**What it costs, stated openly:** judgement-y phrases like *"reads as controlled
circulation rather than a scramble"* cannot be expressed as a predicate. Where a
proxy is used, the docstring says so — that one becomes "at most two LOSS or DUEL
tokens".

### The blind human audit

Because "programmatic relevance rubric" is an unverified claim on its own,
`eval/audit.py` runs a blind check. It samples possessions **from what the system
actually returns** (auditing random corpus rows would mostly surface obvious
non-matches and flatter the number), shows the rubric and the possessions, and
**never shows what the predicate decided** — so you can't anchor to it.

Result: **62 of 71 = 87% agreement.**

And crucially, the *direction* of the disagreements is reported, because it
matters more than the rate:

- 6 were the rubric refusing something the human accepted
- 3 were the reverse

**The rubrics are net stricter than the human, so the precision figure is a floor
rather than an inflated number.** The script computes this direction
automatically and writes the appropriate sentence — including the opposite
warning if the bias ever flips.

### The metrics, in plain words

- **P@5** — of the top 5 results, what fraction were relevant. 0.608 means about
  3 of 5.
- **P@10** — same for the top 10.
- **MRR** — 1 divided by the position of the first relevant result, averaged over
  queries. 0.751 means the first good hit is typically in position 1–2.
- **p50 / p95 latency** — the median and the slow-tail response time. Fused: 36
  ms typical, 123 ms at the 95th percentile.

### The honest exclusion

5 of the 30 queries are **filter-dominated**: the SQL filter alone already
satisfies the rubric for ≥90% of the rows it returns. Ranking cannot affect those
— any five survivors score about 1.0.

So the headline is computed over the **25 discriminating queries**, and both
numbers are published:

| | P@5 (fused) |
|---|--:|
| 25 discriminating queries — **the headline** | **0.608** |
| all 30 queries | 0.667 |

**This is worth pointing out unprompted.** Choosing to publish the *lower* number
as the headline, and explaining why, is exactly the signal an interviewer is
looking for. And which queries are filter-dominated was **measured** with a
`--stats` flag, not guessed.

---

## 16. The API

FastAPI. Eight endpoints, all GET, all taking structured arguments.

| Endpoint | What it does |
|---|---|
| `/search` | The main one. Takes `q=` (English) or `sequence_hint=` (tokens) plus optional filters. |
| `/shape` | Ranked results for a drawn zone path. |
| `/plan` | Translates English to a structured query **without running it** — proves the translation is inspectable rather than a black box. |
| `/similar/{uid}` | More like this. |
| `/possession/{uid}` | Everything the animator needs: ordered events with coordinates. |
| `/shot/{event_id}` | My xG and StatsBomb's for one shot. |
| `/meta` | Dropdown values and corpus counts, cached after the first call. |
| `/health` | Row counts, whether the xG model loaded, **and which rankers are actually live**. |

### Things in here worth knowing

**Heavy objects load once at startup, not per request.** The TF-IDF index and
MiniLM are built/loaded in the FastAPI `lifespan` hook. The first MiniLM call
costs ~12 seconds of import and nobody should pay that inside a query.

**No prebuilt artefact is uploaded to the server.** If the TF-IDF pickle isn't
present, the index is fitted from the database at startup (~2 s for 67k short
documents). **Why:** it makes the deployment self-contained, and the index can
never drift from the data it indexes.

**Explicit parameters override the parser.** If you pass `q=` *and* `team=`, the
explicit team wins. The parser is a starting point the caller can always correct.

**The connection is checked on every request.** Managed Postgres (Neon's free
tier) suspends idle compute after ~5 minutes and closes the socket. A single
connection opened at startup goes stale the first time the demo is left alone,
and every request after that fails until redeploy — *"the deployment would look
fine for five minutes and be broken thereafter."* So each request runs a `SELECT
1` and reconnects if needed. That check is cheap **because** the API and database
are pinned to the same region.

**Defensive events are mirrored onto the same pitch.** StatsBomb records *every*
event as if the acting team attacks left→right. So a defender's interception is
stored in the *defending* team's frame. Plotted raw, it appears in the opposite
corner of the pitch from the attack it interrupted. The API flips those:
`x → 120−x, y → 80−y`. Verified on a specific case (possession `3857266:137`),
where an interception at (27.6, 8.8) flips to (92.4, 71.2) — a yard from the
attacking receipt it broke up.

**The xG model failing to load is non-fatal.** It returns `None` plus the reason,
`/health` reports it, and the UI says the model is unavailable. **Why:** the model
is one panel of one view; retrieval is the product. Refusing to boot because a
599 KB file is missing trades a degraded feature for a total outage.

**`/health` says which rankers are live** — `"sparse only"` or `"sparse+dense
(fused)"` — rather than implying fusion when the hosted box can't run it.

---

## 17. The frontend

Next.js (App Router) + TypeScript. One page, six components. No chart library, no
UI framework, no CSS framework.

| Component | What it does |
|---|---|
| `QueryBar` | English box + structured filter controls; shows the parsed plan and the ignored words |
| `ShapePicker` | The 15-zone grid you click to draw a path |
| `ResultCard` | One result: teams, competition, pattern, token string, "more like this" |
| `ScoutingNote` | The computed sentences; clicking one jumps to its clips |
| `Pitch` | Animated SVG pitch |
| `ShotPanel` | The xG panel |

### The animator

An SVG pitch drawn to StatsBomb's own 120 × 80 coordinates, so the viewBox *is*
the pitch and no transform is needed. The ball is tweened from each event's
location to its end location using `requestAnimationFrame`, with each step's
duration taken from the real event duration.

Three details worth mentioning:

- **`prefers-reduced-motion` is respected** — never auto-plays and jumps rather
  than tweens for users who ask for that.
- **A real bug is documented in the code:** the animation callback closed over a
  stale `i`, letting the index run past the last step and render "5/4". Fixed
  with a ref as the single source of truth for the loop.
- **No fake player dots.** Positions are drawn *only* where a real freeze frame
  exists — the moment of a shot. Everywhere else the ball moves alone, and the UI
  says why: *"No positions are shown at other moments, because the data does not
  contain them."* Inventing player positions would make the animation look better
  and mean nothing.

---

## 18. Deployment

```
   browser
      │ https
      ▼
  ┌──────────────┐  CORS  ┌────────────────────┐  SSL  ┌──────────────┐
  │ VERCEL       │───────▶│ RENDER             │──────▶│ NEON         │
  │ Next.js      │        │ FastAPI, free tier │       │ Postgres     │
  │ Hobby, free  │        │ 512 MB RAM, Ohio   │       │ free, 0.5 GB │
  └──────────────┘        └────────────────────┘       │ us-east-2    │
                                                        │ 424 MB used  │
                                                        └──────────────┘
```

### Four deliberate choices

**1. Render is pinned to Ohio to sit beside Neon's us-east-2.**
A cross-region pair adds ~60 ms to every query, and these endpoints issue several
per request.

**2. The database copy that ships is a slim one.**
`deploy/export_to_neon.py` streams table-to-table with `COPY` — nothing buffered
in Python, no dump file on disk — and leaves behind three things: the raw event
JSONB, the MiniLM embeddings, and the tsvector.

**3.7 GB becomes 424 MB.** That is why the live demo carries the **entire**
66,817-possession corpus instead of a cut-down subset. The column lists are
explicit for the same reason as everywhere else: a `SELECT *` would silently
start shipping the raw JSONB again the moment someone re-adds it upstream.

**3. The dense ranker is off in production.**
torch is ~2.5 GB installed and takes resident memory from 252 MB to 570 MB — past
the free tier's 512 MB cap. So the hosted demo runs **sparse only**, retrieval
quality drops from the fused 0.608 to the sparse 0.544, and `/health` says
`"rankers": "sparse only"` rather than implying it's still fusing. The README
states this plainly.

**4. The xG model *is* served**, because unlike torch it costs almost nothing:
599 KB on disk, ~7 MB resident.

Everything is toggled by environment variables (`PITCHQUERY_DENSE`,
`PITCHQUERY_XG`, `PITCHQUERY_PRELOAD_SHAPES`, `PITCHQUERY_CORS_ORIGINS`) so a
memory-starved box has something to turn off before it starts dropping retrieval
features.

**Known rough edge, stated in the README:** the free Render instance sleeps after
15 minutes idle and takes ~50 seconds to wake. The README tells you to open
`/health` first if you want it warm.

---

## 19. Tests

Three test files, and each one exists for a specific reason rather than for
coverage.

**`tests/test_notes.py`** — enforces the citation guarantee. A claim about goals
must cite only goals. A claim about one team must cite only that team's rows.
Query constraints must not be restated. Every claim must have evidence. *This is
what makes "citations cannot be wrong" a fact rather than a hope.*

**`tests/test_xg_portable.py`** — scores all 10,858 non-penalty shots through the
pickle and through the exported model and requires **exact** equality. *This is
what makes "the model that is served is the model that was measured" a fact.*

**`tests/test_shape.py`** — the path-reduction and subsequence logic behind shape
search.

**The theme to point out:** the tests guard the project's *claims*, not its
lines. Each one turns a sentence in the README into something enforced.

---

## 20. Known problems and how to talk about them

Bring these up yourself. Volunteering a limitation is worth far more than being
caught by one.

### 1. Short possessions are over-retrieved — the real bug

TF-IDF vectors are normalised to unit length. So a **3-token** possession where
every token matches scores near-perfectly, while a **20-token** possession
containing that same passage *plus* the build-up that led to it is diluted and
scores lower.

Measured: across the top 5 for all 30 queries, the median result is **6 tokens**
against a corpus median of **13**.

**Why it inflates the headline:** most rubrics test for the *presence* of tokens
rather than for the possession being a substantial passage of play. A three-token
fragment that technically ends in F-LI is scored relevant while a human would
call it worthless.

**So the README says the retrieval numbers should be read as an upper bound** until
length normalisation is in and every eval is re-run.

**The fix:** BM25's `b` parameter, or a length floor — both are *ranker* changes,
not parser changes. Note that padding the hint differently was tried first,
measured worse (0.680 → 0.592) and reverted, which is how the cause was correctly
located in the ranker rather than the planner.

**Shape search is unaffected** — it never uses cosine similarity, and ties break
on event count.

### 2. The parser only knows its own vocabulary

No alias table, so "PSG" isn't a team. Kept in the evaluation set as a deliberate
failure rather than dropped. An LLM would resolve it; that is the concrete cost of
the trade.

### 3. "Outside the box" is unanswerable

`F-C` spans both the penalty box and the space in front of it, so the 15-zone grid
physically cannot express "outside the box". The judge for that query only counts
the unambiguous case (a shot in an M- zone), which makes that query's precision a
lower bound. It's a structural limit of the grid design, documented rather than
hidden.

### 4. The hosted demo is sparse-only

Covered above. Reported honestly on `/health`.

### 5. `RECV` may be dead weight

A completed pass implies a receipt, so `PASS@X RECV@Y CARRY@Y` spends three tokens
on one action and inflates every string by roughly a third. It's kept **precisely
so** the before/after P@5 of removing it is a measurable result rather than a
guess.

### 6. The planner's headline is in-sample

Stated openly, which is why the paraphrase set exists.

---

## 21. Numbers to memorise

**Corpus**
- 66,817 possessions · 431 matches · 1.60 M events · 11,185 shots (10,858
  excluding penalties)
- 13 competition/seasons, men's and women's
- Mean possession: 19 tokens over 23.6 s; 14.0% end in a shot, 1.57% in a goal
- 15 zones (3 bands × 5 channels) · 14 actions · 3 modifiers

**Retrieval** (25 discriminating queries of 30)
- Fused: **P@5 0.608**, P@10 0.600, **MRR 0.751**
- Sparse 0.544 · Dense 0.584
- Latency: p50 36 ms, p95 123 ms
- Human agreement on a blind 71-item audit: **87%**, rubrics net stricter
- Shape search: ~30 ms across all 66,817

**Planner**
- **1.13 ms** mean, 12.89 ms worst case
- 24/30 exact filter agreement
- Like-for-like on 19 identical-filter queries: hand 0.632, parsed **0.611**
- 23 held-out paraphrases: 0.748 vs 0.730

**xG** (held out: 2022 World Cup + 2023 Women's World Cup, 3,038 shots; trained on
7,820)
- Log-loss: baseline 0.2811 → mine **0.2581** → StatsBomb 0.2507
- **76% of the gap closed**
- AUC 0.7933 (baseline 0.7330, StatsBomb 0.8035)
- O/E 0.967 (baseline 0.893, StatsBomb 1.018)
- Served artefact: **599 KB gzipped, 7 MB resident**, verified exactly equal on
  10,858 shots

**Deployment**
- Local database 3.7 GB → hosted **424 MB**
- Render free tier: 512 MB cap; torch would need 570 MB, so dense is off
- Resident memory with shapes preloaded: 277 MB

---

## 22. Questions they will ask, and how to answer

### "Walk me through the project."
Use the 60-second pitch, then: *"The core idea is the compression step — every
possession becomes one line of a small invented language, and that line is what
gets indexed and searched. Everything else follows from that: it's why classic
text retrieval works here, it's why the query parser can be rules instead of a
model, and it's why drawing a shape is just a string comparison."*

### "Why not just use an LLM for the whole thing?"
*"Two places called for one and neither needed one. For query parsing, rules gave
me determinism, zero cost, 1 ms, and traceability — I can show the user exactly
which phrase produced which filter and which words I failed to understand. And I
measured it against 30 hand-written queries rather than assuming it was good
enough: it's slightly worse like-for-like, 0.611 against 0.632, and it holds on
held-out paraphrases. For the scouting notes, an LLM would need a citation
checker; computing the sentence from the rows it cites means there's nothing that
can lie in the first place. The honest cost is coverage — 'PSG' isn't a team,
and that case is in my evaluation set as a deliberate failure."*

### "Why 15 zones?"
*"It's a compression trade-off. Too fine and two crosses three yards apart look
unrelated so nothing matches; too coarse and everything matches. Thirds and
half-spaces are also how coaches actually talk, so a query in football language
maps onto the grid cleanly. The cost is that 'outside the box' is unanswerable,
because F-C spans both the box and the space in front of it — I documented that
rather than hiding it."*

### "Why filter before ranking?"
*"If you rank first and filter after, a search for Barcelona's corners gives you
the top 50 corners overall, then the filter deletes most of them and you're left
with whichever three happened to be Barcelona's — not their best ones. Filtering
first means ranking always operates on the right pool. The filters are
whitelisted columns with bound parameters, so user text can never reach the SQL."*

### "Why TF-IDF instead of just embeddings?"
*"Because the documents aren't natural language — they're a controlled code with
about 14 actions and 15 zones. Exact overlap of codes is what TF-IDF is built
for, and n-grams of 1 to 3 make order count, so it behaves like phrase matching.
I expected it to beat the embeddings outright. It didn't quite — dense won on P@5,
sparse won on MRR — but fusing them beat both, because they agree on only about
one of their top ten results. Where dense fails is specific and instructive: asked
for a right-wing cross it returned a left-wing one, because MiniLM sees F-L and
F-R as strings one character apart."*

### "Why RRF and not a weighted score blend?"
*"The two scores are on incomparable scales. Adding them, or normalising them into
a shared range, invents a comparison that doesn't exist. Rank position is the one
thing both lists genuinely agree on the meaning of, so fusing on rank is the
honest option. Also, it has no weight to tune, which means no hyperparameter I'd
be tempted to fit on my own evaluation set."*

### "How do you know your evaluation isn't rigged?"
*"Four things. The rubrics were written before I saw any results. They're code, so
they apply to all 66,817 possessions rather than a pooled sample and they don't
drift as I get tired. I ran a blind human audit where I was never shown what the
predicate decided — 87% agreement — and I report the direction of the
disagreements, which shows the rubrics are net stricter than me, so precision is
a floor. And I excluded the 5 queries where the SQL filter alone already
satisfies the rubric, which drops my headline from 0.667 to 0.608 — I publish the
lower number because ranking can't affect those queries."*

### "What's the weakest part of this?"
*"Short possessions are over-retrieved, and it inflates my headline number. TF-IDF
vectors are L2-normalised, so a three-token possession where everything matches
outscores a twenty-token one that contains the same passage plus the build-up.
The median result in my top 5 is 6 tokens against a corpus median of 13. So I say
in the README that the retrieval numbers are an upper bound until I add length
normalisation and re-run everything. I also found the cause properly — I tried
fixing it in the parser first by padding the hint differently, measured it worse
at 0.592, and reverted, which told me it was a ranker problem, not a parser one."*

### "Why hold out whole competitions for the xG model?"
*"Shots inside one match share a game state, a pitch, a team and often the same
shooter. With a random shot split, shot three and shot four from the same attack
land on opposite sides, so the model has effectively seen the test data and every
metric is inflated. I held out the 2022 World Cup and the 2023 Women's World Cup
entirely — one men's and one women's major tournament, chosen before I looked at
any score, so the test set isn't a single style of football. There's an assert
that fails if a competition ever appears in both splits."*

### "Your model doesn't beat StatsBomb's."
*"No, and I'd be suspicious of my own work if it did — theirs is a production
model with far more data and features. What I measure is how much of the distance
between a distance-and-angle baseline and their model my version covers, which is
76% of the log-loss gap. And I lead with log-loss rather than AUC on purpose: AUC
only cares about ordering, and an xG model that ranks perfectly can still be
systematically wrong about how many goals to expect. My O/E is 0.967 against the
baseline's 0.893."*

### "Why not just ship the pickle?"
*"Because it would pin scikit-learn and pandas on the server to whatever versions
my laptop happened to have, forever, with no warning when they drift. So I
re-serialised it into what it actually is — three LightGBM ensembles in LightGBM's
own text format plus a two-parameter sigmoid on each, scored with numpy. 599 KB,
7 MB resident, no version lock. And because the translation is hand-written, a
test scores all 10,858 non-penalty shots both ways and requires exact equality.
That test caught the real trap: scikit-learn calibrates LightGBM's raw margin,
not its probability. Calibrating the probability instead gives you values wedged
between 0.43 and 0.62 — all of which look like believable xG on a page while
being wrong on every shot. A smoke test passes that. An equality test doesn't."*

### "How does the shape search rank?"
*"By coverage — what fraction of the possession's whole journey the drawing
accounts for, after collapsing consecutive repeats. My first attempt ranked by how
tightly the drawn zones clustered, and it was wrong: it surfaced hundred-touch
possessions where three zones happened to line up somewhere in the middle, which
is incidental rather than the shape of the move. Coverage also makes the
interaction self-consistent — the number of zones you draw sets the length of
possession you get back."*

### "How would you scale this to a full season, or ten leagues?"
*"The sparse matrix is in RAM, so that's the first ceiling — 67k documents is
trivial but a few million would want a real inverted index, or moving the sparse
side into Postgres's own full-text index, which is why token_tsv already exists.
The filter step is already SQL and scales with indexes. The dense side is already
in pgvector with HNSW, so that scales fine. And the ingest is embarrassingly
parallel — it's per-match with idempotent upserts, so it parallelises without any
coordination."*

### "What would you do differently?"
*"Length normalisation from the start — BM25 rather than plain TF-IDF cosine. It's
the one defect that touches my headline number. And I'd write the evaluation
harness before the ranker rather than alongside it; several of my decisions were
made on intuition and then measured afterwards, and at least one of them —
padding the hint — turned out to be wrong."*

### "Where did the time go?"
Be honest: the grammar design and the ingest verification, because everything
downstream depends on them being right; and the evaluation harness, because
without it every claim would be an assertion.

---

## 23. If they ask "what next?"

In priority order:

1. **Length normalisation in the ranker** (BM25's `b`, or a length floor), then
   re-run every evaluation. This is the one open defect that touches a headline
   number.
2. **Test removing `RECV`.** It may be a third of every string for no gain. The
   before/after P@5 is a measurable result.
3. **An alias table for the parser** — "PSG", "Man City", "Barça". Cheap, and it
   closes the most visible failure.
4. **Fit the dense ranker on a box that can host torch**, or swap MiniLM for a
   smaller/quantised model so production can fuse rather than run sparse-only.
5. **A finer grid, or a box boundary**, so "outside the box" becomes answerable.
6. **Player-level features in xG** — shooter identity, finishing history — though
   that needs care about sample size per player.

---

## One-line summary of every file

```
core/
  zones.py       the 15-zone grid + the token grammar. The heart of the project.
  features.py    shot geometry (distance, angle) + freeze-frame features
  retrieval.py   filters, sparse ranker, dense ranker, shape search, RRF fusion
  planner.py     English → filters + token hint. Rules, no LLM, ~1 ms
  notes.py       scouting claims computed from the clips they cite
  xg.py          serving-side scorer: numpy + lightgbm, no sklearn, no pandas
  db.py          Postgres connection helper
  config.py      paths, pitch constants, model names

ingest/
  00_inventory   what StatsBomb offers
  01_probe       verify the six assumptions → docs/probes.md
  02_fetch       download + cache the JSON
  03_load_events flatten into Postgres, compute tokens and shot features
  04_build_poss  group into 66,817 possessions with token strings
  05_embed       fit TF-IDF; embed with MiniLM into pgvector; build HNSW

models/
  train_xg              train baseline + LightGBM, hold out whole competitions
  export_xg_portable    re-serialise the pickle into text format + sigmoids
  evaluate_xg           metrics + calibration plot → docs/benchmark.md
  xg_portable.json.gz   the served model, 599 KB, committed

eval/
  queries.yaml     30 queries: filters + hint + written rubric
  paraphrases.yaml 23 held-out restatements
  judge.py         each rubric as a predicate
  audit.py         blind human agreement check
  score_retrieval  P@5 / P@10 / MRR / latency → docs/retrieval_eval.md
  score_planner    parser vs hand-written → docs/planner_eval.md

api/     main.py (8 endpoints), schemas.py (response shapes)
web/     Next.js page + Pitch, ShapePicker, QueryBar, ResultCard,
         ScoutingNote, ShotPanel
deploy/  schema_deploy.sql, export_to_neon.py (3.7 GB → 424 MB)
sql/     001_schema.sql, 002_indexes.sql
tests/   test_notes (citations), test_shape (paths), test_xg_portable (exactness)
```

---

**Data source: StatsBomb Open Data**, used under their open-data licence, credited
in the README, in the app footer, and on every exported chart.
