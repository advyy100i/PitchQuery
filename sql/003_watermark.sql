-- Phase 2: incremental loads.
--
-- One row per competition/season recording how far the loader got. The fetch
-- step reads it and drops every match at or below `last_match_id`; the load
-- step advances it inside the same transaction as the inserts it describes, so
-- a crash mid-load leaves the watermark pointing at the last match that was
-- actually committed rather than the last one that was attempted.
CREATE TABLE IF NOT EXISTS ingest_watermark (
  competition_id INT,
  season_id      INT,
  last_match_id  BIGINT,
  last_run_at    TIMESTAMPTZ,
  rows_loaded    BIGINT,
  PRIMARY KEY (competition_id, season_id)
);
