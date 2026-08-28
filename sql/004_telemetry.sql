-- Phase 8: what people actually searched for, and what they opened.
--
-- The point of these two tables is not analytics. It is that 30 hand-written
-- eval queries is a small training set for the Phase 7 ranker, and the only
-- honest way to grow it is from real use — inventing thirty more queries would
-- grow the set without growing the evidence.
--
-- `unparsed_words` earns its place on its own: it is a ranked list of the
-- vocabulary core/planner.py does not recognise, which is a to-do list written
-- by users rather than guessed at.

CREATE TABLE IF NOT EXISTS search_log (
  id             BIGSERIAL PRIMARY KEY,
  ts             TIMESTAMPTZ NOT NULL DEFAULT now(),
  query_text     TEXT,
  -- The structured query the planner produced. JSONB rather than columns
  -- because the filter set is core/retrieval.Filters and will grow.
  parsed_filters JSONB,
  sequence_hint  TEXT,
  -- Words the parser could not place. The reason this table exists.
  unparsed_words TEXT[],
  ranker         TEXT,          -- 'sparse' | 'fused' | 'learned' | 'shape'
  latency_ms     INT,
  rerank_ms      INT,
  n_results      INT,
  n_candidates   INT,
  top_uids       TEXT[]
);

CREATE INDEX IF NOT EXISTS search_log_ts ON search_log (ts DESC);

CREATE TABLE IF NOT EXISTS click_log (
  id             BIGSERIAL PRIMARY KEY,
  -- ON DELETE CASCADE so that trimming old searches cannot leave clicks
  -- pointing at a search nobody can look up.
  search_id      BIGINT REFERENCES search_log(id) ON DELETE CASCADE,
  possession_uid TEXT NOT NULL,
  rank           INT NOT NULL,
  ts             TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS click_log_search ON click_log (search_id);
-- The nightly job in pipeline/telemetry.py scans for deep clicks; without this
-- it is a sequential scan of every click ever recorded.
CREATE INDEX IF NOT EXISTS click_log_rank ON click_log (rank) WHERE rank >= 5;
