CREATE EXTENSION IF NOT EXISTS vector;

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
  idx             INT NOT NULL,          -- StatsBomb "index": authoritative order
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
  raw             JSONB NOT NULL
);

CREATE TABLE IF NOT EXISTS shots (
  event_id        UUID PRIMARY KEY REFERENCES events(event_id),
  match_id        BIGINT,
  competition_id  INT,
  season_id       INT,
  team            TEXT,
  player          TEXT,
  x               REAL, y REAL,
  distance        REAL,      -- to goal centre (120, 40)
  angle           REAL,      -- visible goal-mouth angle, radians
  body_part       TEXT,
  technique       TEXT,
  shot_type       TEXT,      -- Open Play / Free Kick / Corner / Penalty
  first_time      BOOLEAN,
  under_pressure  BOOLEAN,
  play_pattern    TEXT,
  is_goal         BOOLEAN NOT NULL,
  statsbomb_xg    REAL,      -- reference model. LABEL FOR COMPARISON ONLY, never a feature.
  freeze_frame    JSONB,
  n_def_in_cone   INT,
  dist_nearest_def REAL,
  gk_dist_to_goal REAL,
  gk_off_line     REAL
);

CREATE TABLE IF NOT EXISTS possessions (
  possession_uid  TEXT PRIMARY KEY,      -- '{match_id}:{possession}'
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
  zone_path       TEXT,        -- 'D-C D-L M-L M-C F-RI F-R F-C'
  token_string    TEXT,        -- 'RECOV@D-C PASS@D-L+ ... SHOT@F-C'
  token_tsv       TSVECTOR,
  ended_in_shot   BOOLEAN,
  xg_sum          REAL,
  ended_in_goal   BOOLEAN,
  embedding       VECTOR(384)
);
