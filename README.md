# PitchQuery — Tactical Possession Search over Open Football Data

Type a description of a passage of play in English, get back ranked animated
clips of possessions that match, plus a generated scouting note.

## The three numbers (fill in at the end)

1. Search over **66,817** possessions from **431** matches, precision@5 of
   **0.608** on a 30-query evaluation set, p95 latency **58** ms.
   Relevance comes from per-query programmatic rubrics written before any
   results were seen, which agree with human judgement **87%** of the time on a
   blind 71-item audit (`docs/retrieval_eval.md`).
2. My context-aware xG closes **76%** of the log-loss gap between a
   distance+angle baseline and StatsBomb's own production xG model, on the
   held-out 2022 World Cup and 2023 Women's World Cup (`docs/benchmark.md`).
3. Natural-language queries are translated to structured filters by a
   **deterministic rule parser, not an LLM** — free, offline, 1 ms per parse,
   and every filter traceable to the phrase that produced it. Measured against
   the same 30 queries written by hand: **0.611 vs 0.632 P@5** on the subset
   where both produce identical filters, and it holds up on held-out
   paraphrases (`docs/planner_eval.md`). Every claim in a generated scouting
   note links to the possessions it was computed from, and **cannot cite
   anything else** — the sentence and its evidence come from one expression
   (`core/notes.py`, enforced by `tests/test_notes.py`).

## Known defect: short possessions are over-retrieved

Across the top 5 results for all 30 parsed queries, the median possession is
**6 tokens against a corpus median of 13**, and **35% have 4 tokens or fewer**.
TF-IDF vectors are L2-normalised, so a 3-token possession whose every token
matches the query scores near-perfectly, while a 20-token possession containing
the same passage plus the build-up that led to it is diluted.

This inflates the headline P@5: most rubrics test for the presence of tokens,
not for the possession being a substantial passage of play, so a 1-second
fragment can satisfy the letter of the rule. Fixing it means a length floor or
a re-tuned normalisation, and re-running every eval — the numbers above should
be read as an upper bound until then.

## Two things the plan did not ask for

**Draw a possession instead of describing one.** Click zones on the pitch and
get back moves that took that journey — no text, no vectors, no model. The
query is matched against the `zone_path` every possession already carries, in
~30 ms over 66,817 of them.

Ranking is by *coverage*: what fraction of a possession's whole journey the
drawing accounts for, after collapsing consecutive repeats. Scoring by how
tightly the drawn zones cluster instead was tried first and was wrong — it
surfaced 100-touch possessions where the three drawn zones happened to line up
somewhere in the middle, which is incidental rather than the shape of the move.
Coverage also makes the interaction self-consistent: the number of zones you
draw sets the length of move you get back.

**Scouting notes whose citations cannot be wrong.** The plan specified an LLM
that cites `possession_uid`s, plus a checker to drop sentences whose citations
don't verify. Here each sentence is *computed from* the possessions it cites —
claim and evidence are the same expression — so there is nothing to check:

```python
goals = [r for r in rows if r["ended_in_goal"]]
Claim(f"{len(goals)} of the {n} are scored", uids=[r["possession_uid"] for r in goals])
```

That is a stronger guarantee than a generator plus a checker, since the checker
exists to catch a generator that can lie. `tests/test_notes.py` enforces it, and
caught a real violation while being written: a sentence counting goals was
citing every shot. The note also skips whatever the query already fixed —
filtering on corners and then being told "8 begin from a corner" is an echo, not
an observation.

The honest cost: it says only what those functions know how to say, and will
never notice something surprising the way a model might.

## A claim the plan made that the data did not support

The build plan predicted that the sparse n-gram ranker would beat the dense one,
because the tokens are a controlled vocabulary rather than natural language, and
that dense retrieval would earn its place only in fusion. Measured over 25
discriminating queries, that is only half right:

| ranker | P@5 | P@10 | MRR |
|---|--:|--:|--:|
| sparse (TF-IDF n-grams) | 0.544 | **0.540** | **0.671** |
| dense (MiniLM + pgvector) | **0.584** | 0.516 | 0.638 |
| fused (RRF) | **0.608** | **0.600** | **0.751** |

Sparse wins on P@10 and MRR — it ranks its best hit higher and holds precision
deeper. Dense wins on P@5. They disagree far more than expected: **only 1 of the
top 10 results overlaps** on a typical query. Fusion beats both on every metric,
which is the real result, and it beats them *because* the two rankers fail
differently rather than because either is strong alone.

Where dense fails is specific and worth showing. Asked for a right-wing cross,
its top hit was `CROSS@F-L>` — a left-wing cross. MiniLM sees `F-L` and `F-R` as
nearly the same string, because they are one character apart, and it has no idea
one means left. Sparse never makes that mistake. Dense earns its keep on queries
where the exact token sequence differs but the shape is right (`q10` dribble into
the box: sparse 0.0, dense 1.0).

## Deploying

**Vercel** for the frontend, **Render** for the API, **Neon** for Postgres —
all three free, no card required for any of them.

Everything about this split follows from one measurement. The local database is
3.7 GB, of which 1.35 GB is raw StatsBomb JSONB and 98 MB is MiniLM vectors.
Neither is needed to serve a query: the grammar token each event was parsed for
is stored on the row, and the dense ranker cannot run on a free tier anyway. Drop
those columns and the **entire 431-match corpus ships in 424 MB** — no demo
subset, no cutting the data down to a hundred matches.

| | local | hosted |
|---|--:|--:|
| database | 3.7 GB | **424 MB** |
| API resident memory | 570 MB | **252 MB** |
| rankers | sparse + dense (fused) | sparse only |
| P@5 (25 discriminating queries) | 0.608 | 0.544 |

The API reports which mode it is in on `/health` (`"rankers": "sparse only"`)
rather than implying it is still fusing two rankers. Dense retrieval is off
because torch is ~2.5 GB installed and takes resident memory to 570 MB, past
Render's 512 MB. Everything else — sparse retrieval, shape search, the rule
planner, scouting notes, clip playback with freeze frames — is unchanged.

### 1. Database

Create a free Neon project, then push the serving subset straight into it. The
script streams table to table with `COPY`; nothing is buffered or written to
disk.

```powershell
python deploy/export_to_neon.py --target "postgresql://...neon.tech/db?sslmode=require"
python deploy/export_to_neon.py --target "..." --dry-run      # size it first
python deploy/export_to_neon.py --target "..." --matches 250  # ~265 MB instead
```

424 MB is 85% of Neon's 500 MB free tier, which works but leaves little room for
WAL and vacuum churn. `--matches 250` ships the most recent 250 matches at
roughly 265 MB if you would rather have the headroom; search still covers every
possession in whatever you ship. `events` is 368 MB of the total and exists only
so clips can be played back.

### 2. API

Render picks up `render.yaml` from the repo root ("New > Blueprint"). Two
variables are marked `sync: false` and must be set in the dashboard:

- `DATABASE_URL` — the Neon string, including `?sslmode=require`
- `PITCHQUERY_CORS_ORIGINS` — the Vercel URL, once step 3 gives you one

No build artefact is uploaded. The TF-IDF index is fitted from the database at
startup in about two seconds, which also means it can never drift from the data
it was built out of.

### 3. Frontend

Import the repo on Vercel, set **Root Directory** to `web`, and add one
environment variable:

```
NEXT_PUBLIC_API_URL = https://your-api.onrender.com
```

Then go back and set `PITCHQUERY_CORS_ORIGINS` on Render to the Vercel URL.
Until you do, the API answers `curl` but a browser blocks every request from
the site.

### Free-tier behaviour worth knowing

Render free web services sleep after 15 minutes idle and take **~50 seconds** to
wake. That is fine for a link someone opens once, and bad in a live interview —
open the API's `/health` a minute before you demo. Neon suspends compute too but
resumes in well under a second.

## Data source

Event data from **StatsBomb Open Data**
(<https://github.com/statsbomb/open-data>), used under their open-data licence.
StatsBomb is credited here, in the web app footer, and on every exported chart.

## Status

- [x] Phase 0 — inventory + assumption probes (`docs/data_inventory.md`, `docs/probes.md`)
- [x] Phase 1 — ingest to Postgres (`docs/ingest.md`) — 431 matches, 1.60M events, 11,185 shots
- [x] Phase 2 — zone grid + token grammar — 66,817 possessions tokenised
- [x] Phase 3 — hybrid retrieval + eval (`docs/retrieval_eval.md`) — fused P@5 0.608, p95 126 ms
      *(pending: human audit of the programmatic rubrics, `python eval/audit.py`)*
- [x] Phase 4 — xG benchmark (`docs/benchmark.md`) — 76% of the log-loss gap closed
- [x] Phase 5 — FastAPI + Next.js pitch animator (no LLM involved)
- [x] Phase 6 — query planner (`docs/planner_eval.md`) — **rule-based, no LLM**
- [x] Draw-a-shape search (`GET /shape`) — retrieval with no text and no model
- [x] Scouting notes with uncheatable citations (`core/notes.py`)
- [ ] Phase 8 — ship

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt          # Phase 0-2
copy .env.example .env

python ingest/00_inventory.py            # -> docs/data_inventory.md
python ingest/01_probe_assumptions.py    # -> docs/probes.md

docker compose up -d db                  # postgres 16 + pgvector on :5433
python ingest/02_fetch.py --comp 43:106 --comp 72:107   # ... see docs/ingest.md
python ingest/03_load_events.py --init
python ingest/04_build_possessions.py
python ingest/05_embed.py                # TF-IDF + MiniLM -> pgvector

pip install -r requirements-ml.txt       # Phase 3-4 (torch, CPU only)
python eval/score_retrieval.py           # -> docs/retrieval_eval.md
python models/train_xg.py
python models/evaluate_xg.py             # -> docs/benchmark.md

pip install pytest
python -m pytest tests/ -q               # citation + shape-matching guarantees
```

### Running the app

```powershell
docker compose up -d db
uvicorn api.main:app --port 8000         # http://localhost:8000/docs
cd web; npm install; npm run dev         # http://localhost:3000
```

Endpoints: `GET /search` (takes either `q=` plain English or `sequence_hint=`
tokens), `GET /shape?zones=D-C,M-C,F-C` (retrieve by a drawn path),
`GET /plan?q=` (the English → structured-query translation on its own),
`GET /similar/{uid}`, `GET /possession/{uid}`, `GET /shot/{event_id}` (my xG and
StatsBomb's side by side), `GET /meta`.

Raw JSON is cached under `PITCHQUERY_DATA` (default `~/pitchquery-data`), which is
deliberately outside any cloud-synced folder.

