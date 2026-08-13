-- The serving schema: everything a hosted API reads, and nothing else.
--
-- Three columns are deliberately absent, and together they are why the whole
-- 431-match corpus fits a 500 MB free tier instead of needing a demo subset:
--
--   events.raw          1353 MB of StatsBomb JSONB. Only the ingest needs it;
--                       the grammar token it was parsed for is now stored on
--                       the row (events.token), so serving never touches it.
--   possessions.embedding  98 MB of MiniLM vectors, useless without torch to
--                       embed the query — and torch does not fit a free tier.
--   possessions.token_tsv  plus its GIN index. Postgres full-text search is
--                       not on the retrieval path; the Python TF-IDF ranker is.
--
-- What that leaves: events 276 MB, possessions 19 MB, shots 13 MB (freeze
-- frames kept, since they are only 11 MB and they draw the player dots).

CREATE TABLE IF NOT EXISTS matches (
  match_id        BIGINT PRIMARY KEY,
  competition_id  INT,
  season_id       INT,
  competition     TEXT,
  season          TEXT,
  match_date      DATE,
  home_team       TEXT,
  away_team       TEXT,
  home_score      INT,
  away_score      INT,
  has_360         BOOLEAN DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS events (
  event_id        UUID PRIMARY KEY,
  match_id        BIGINT REFERENCES matches(match_id),
  idx             INT NOT NULL,
  period          SMALLINT,
  minute          SMALLINT,
  second          SMALLINT,
  type            TEXT,
  play_pattern    TEXT,
  possession      INT,
  possession_team TEXT,
  team            TEXT,
  player          TEXT,
  position        TEXT,
  x               REAL,
  y               REAL,
  end_x           REAL,
  end_y           REAL,
  under_pressure  BOOLEAN,
  duration        REAL,
  token           TEXT
);

CREATE TABLE IF NOT EXISTS shots (
  event_id        UUID PRIMARY KEY,
  match_id        BIGINT,
  competition_id  INT,
  season_id       INT,
  team            TEXT,
  player          TEXT,
  x               REAL, y REAL,
  distance        REAL,
  angle           REAL,
  body_part       TEXT,
  technique       TEXT,
  shot_type       TEXT,
  first_time      BOOLEAN,
  under_pressure  BOOLEAN,
  play_pattern    TEXT,
  is_goal         BOOLEAN NOT NULL,
  statsbomb_xg    REAL,
  freeze_frame    JSONB,
  n_def_in_cone   INT,
  dist_nearest_def REAL,
  gk_dist_to_goal REAL,
  gk_off_line     REAL
);

CREATE TABLE IF NOT EXISTS possessions (
  possession_uid  TEXT PRIMARY KEY,
  match_id        BIGINT REFERENCES matches(match_id),
  possession      INT,
  team            TEXT,
  opponent        TEXT,
  competition     TEXT,
  season          TEXT,
  play_pattern    TEXT,
  start_idx       INT,
  end_idx         INT,
  n_events        INT,
  duration_s      REAL,
  start_zone      TEXT,
  end_zone        TEXT,
  zone_path       TEXT,
  token_string    TEXT,
  ended_in_shot   BOOLEAN,
  xg_sum          REAL,
  ended_in_goal   BOOLEAN
);
