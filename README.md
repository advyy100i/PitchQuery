# PitchQuery

**Search 66,817 football possessions by what happened in them.** Describe a
passage of play in English — or draw its shape on a pitch — and get back ranked,
animated clips, with a scouting note whose every claim links to the clips it was
computed from.

Live Link: **[pitch-query.vercel.app](https://pitch-query.vercel.app)** 

Backend Deployed on: [API](https://pitchquery-api.onrender.com/health)

> The hosted API sleeps after 15 minutes idle and takes ~50 s to wake. Open
> [`/health`](https://pitchquery-api.onrender.com/health) first if you want it
> warm.

## Video Demo

https://github.com/user-attachments/assets/4093f1bf-9ba3-45d4-9c56-5f92ef6957ea



---

## The problem

Event data describes football as a list of rows: a pass here, a carry there,
coordinates and timestamps. Analysts do not think in rows. They think in
passages — *"a cross from the right that ends in a shot"*, *"playing out from
the back under pressure"* — and there is no way to query for one.

PitchQuery makes the passage the unit of retrieval. Each of the 66,817
possessions in the corpus is compressed into a single line of a controlled
grammar:

```
SETP@F-R> RECV@F-C SHOT@F-C^                       a corner, headed at goal under pressure
RECOV@M-C CARRY@M-C+ PASS@F-LI+ RECV@F-C SHOT@F-C  a counter-attack
```

`ACTION@ZONE` over a 3x5 grid, with `+` progressive, `>` into the box, `^` under
pressure. That string is what gets indexed, ranked and searched.

---

## Results

| | | how it was measured |
|---|---|---|
| **Retrieval** | P@5 **0.608**, MRR 0.751, p50 36 ms / p95 123 ms | 30 queries, relevance from per-query rubrics written *before* any results were seen; **87% agreement** with a human on a blind 71-item audit |
| **xG model** | closes **76%** of the log-loss gap to StatsBomb's production model, and is live in the demo | held out the 2022 World Cup and 2023 Women's World Cup entirely; split by competition, never by shot |
| **Query parsing** | **1 ms**, no LLM, no API key | 24/30 filter agreement with hand-written queries, and it holds on a held-out paraphrase set |

Full write-ups: [retrieval](docs/retrieval_eval.md) ·
[xG benchmark](docs/benchmark.md) · [planner](docs/planner_eval.md) ·
[data](docs/ingest.md)

---

## How it works

<img width="2720" height="2640" alt="possession_search_architecture" src="https://github.com/user-attachments/assets/efa3c625-1f24-4744-b2a1-e9f4234cb9f8" />


Hard filters run in SQL first and ranking happens second — never the reverse.
Ranking the whole corpus and filtering afterwards means a query for one team's
corners comes back with another team's open play once the filter eats the top
results.

---

## Four decisions worth explaining

### The dense ranker was supposed to lose. It didn't.

The premise was that TF-IDF over a controlled vocabulary would beat sentence
embeddings, and that dense retrieval would earn its place only in fusion. Half
right:

| ranker | P@5 | P@10 | MRR |
|---|--:|--:|--:|
| sparse (TF-IDF n-grams) | 0.544 | **0.540** | **0.671** |
| dense (MiniLM + pgvector) | **0.584** | 0.516 | 0.638 |
| **fused (RRF)** | **0.608** | **0.600** | **0.751** |

Sparse ranks its best hit higher; dense wins on P@5. They agree on only **1 of
their top 10** results for a typical query — which is exactly why fusing them
works. Where dense fails is specific: asked for a right-wing cross it returned
`CROSS@F-L>`, a *left*-wing cross, because MiniLM sees `F-L` and `F-R` as one
character apart and has no idea one means left.

### The English layer is rules, not a language model.

A parser turns *"Barcelona working the ball into the left half-space"* into
structured filters plus a synthetic token string. It runs in **1 ms**, needs no
key, and is deterministic — so a retrieval regression is never the planner's
fault. Crucially, it shows its work: the UI displays which phrase produced which
filter, and **which words it failed to understand**.

Measured against the same 30 queries written by hand, it scores **0.611 vs
0.632** P@5 on the subset where both produce identical filters — and it holds up
on paraphrases it was never tuned against. An LLM would generalise further; this
trades that for being free, instant and inspectable.

### Citations that cannot be wrong.

The scouting note's sentences are *computed from* the possessions they cite —
claim and evidence are the same expression, so there is no verification step
because there is nothing that can lie:

```python
goals = [r for r in rows if r["ended_in_goal"]]
Claim(f"{len(goals)} of the {n} are scored", uids=[r["possession_uid"] for r in goals])
```

`tests/test_notes.py` enforces it, and caught a real violation while being
written: a sentence counting goals was citing every shot.

### The xG model is deployed as arithmetic, not as a pickle.

Every possession that ends in a shot is scored by the model, and the panel says
whether that competition was held out of training — a good number on data the
model learned from is a fit, not a test. The comparison with StatsBomb belongs
in the [benchmark](docs/benchmark.md) over thousands of shots, not next to one
clip, where a 0.15 chance going in says nothing either way.

Training produces a `CalibratedClassifierCV` around three LightGBM boosters.
Shipping that pickle would mean pinning scikit-learn and pandas on the server to
whatever a laptop happened to have. So it is re-serialised into what it actually
is — three ensembles in LightGBM's own text format, a two-parameter sigmoid on
each — and the whole thing is **599 KB gzipped, committed to the repo, and
scored with numpy**. Serving it costs 7 MB of resident memory.

The translation is hand-written, so `tests/test_xg_portable.py` scores all
10,858 non-penalty shots both ways and requires exact equality. It earns its
place: scikit-learn calibrates lightgbm's *raw margin*, not its probability, and
calibrating the probability instead returns values wedged between 0.43 and 0.62
— all of which look like believable xG on a page while being wrong on every
shot.

---

## Draw a possession instead of describing one

![Shape search](docs/ui-shape.png)

Click zones on the pitch and get back moves that took that journey. No text, no
vectors, no model — the query is matched against the `zone_path` every
possession already carries, in ~30 ms across all 66,817.

Ranking is by **coverage**: what fraction of a possession's whole journey the
drawing accounts for, after collapsing consecutive repeats. Ranking by how
tightly the drawn zones cluster was tried first and was wrong — it surfaced
100-touch possessions where three zones happened to line up somewhere in the
middle, which is incidental rather than the shape of the move.

---

## Limitations

**Short possessions are over-retrieved.** TF-IDF vectors are L2-normalised, so a
3-token possession whose every token matches scores near-perfectly, while a
20-token possession containing the same passage *plus* the build-up that led to
it is diluted. Across the top 5 for all 30 parsed queries the median possession
is 6 tokens against a corpus median of 13.

This inflates the headline P@5, because most rubrics test for the presence of
tokens rather than for the possession being a substantial passage of play. **The
retrieval numbers above should be read as an upper bound** until length
normalisation (BM25's `b`, or a length floor) is in and every eval is re-run.
Shape search is unaffected — it never uses cosine similarity.

**The parser only knows the vocabulary written into it.** No alias table, so
"PSG" is not a team. That case sits in the evaluation set as a deliberate
failure rather than being quietly dropped.

**"Outside the box" is unanswerable** on a 15-zone grid: `F-C` spans both the
box and the space in front of it. A structural limit of the grid design,
documented rather than hidden.

**The hosted demo runs sparse-only.** torch is ~2.5 GB installed and takes
resident memory from 252 MB to 570 MB, past the free tier's 512 MB. `/health`
reports `"rankers": "sparse only"` rather than implying it is still fusing. The
xG model is unaffected and does run there — it is 599 KB and 7 MB resident.

---

## Running it

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env

docker compose up -d db                  # Postgres 16 + pgvector on :5433
python ingest/02_fetch.py --comp 43:106 --comp 72:107   # see docs/ingest.md
python ingest/03_load_events.py --init
python ingest/04_build_possessions.py
python ingest/05_embed.py

uvicorn api.main:app --port 8000
cd web; npm install; npm run dev         # http://localhost:3000
```

Reproduce the numbers:

```powershell
pip install -r requirements-ml.txt
python eval/score_retrieval.py    # -> docs/retrieval_eval.md
python eval/score_planner.py      # -> docs/planner_eval.md
python models/train_xg.py; python models/evaluate_xg.py
python -m pytest tests/ -q
```

`deploy/export_to_neon.py` ships a slim copy to hosted Postgres: dropping the
raw JSONB, embeddings and tsvector takes 3.7 GB down to **424 MB**, which is why
the live demo carries the entire corpus rather than a subset.

---

## Stack

Python · PostgreSQL 16 + pgvector · scikit-learn · LightGBM ·
sentence-transformers · FastAPI · Next.js · Docker ·
deployed on Vercel + Render + Neon

**Data source: [StatsBomb Open Data](https://github.com/statsbomb/open-data)**,
used under their open-data licence. StatsBomb is credited here, in the app
footer, and on every exported chart.
