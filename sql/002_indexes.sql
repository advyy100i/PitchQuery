CREATE INDEX IF NOT EXISTS events_match_idx      ON events (match_id, idx);
CREATE INDEX IF NOT EXISTS events_match_poss     ON events (match_id, possession);
CREATE INDEX IF NOT EXISTS possessions_team      ON possessions (team);
CREATE INDEX IF NOT EXISTS possessions_pattern   ON possessions (play_pattern);
CREATE INDEX IF NOT EXISTS possessions_shot      ON possessions (ended_in_shot);
CREATE INDEX IF NOT EXISTS possessions_tsv_idx   ON possessions USING GIN (token_tsv);

-- Build this AFTER the embeddings are loaded; creating an empty HNSW index and
-- then bulk-inserting is much slower than the other way round.
-- CREATE INDEX possessions_emb_idx ON possessions USING hnsw (embedding vector_cosine_ops);
