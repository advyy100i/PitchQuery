"""Paths and constants. Nothing here costs money or needs a key."""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Raw JSON lives OUTSIDE OneDrive on purpose: a few hundred matches is several GB
# and OneDrive's free tier is 5 GB. Override with the PITCHQUERY_DATA env var.
DATA_DIR = Path(os.getenv("PITCHQUERY_DATA", Path.home() / "pitchquery-data"))
RAW_DIR = DATA_DIR / "raw"
# Pickled TF-IDF vectorizer + sparse matrix. Rebuilt by ingest/05_embed.py,
# loaded once at API startup.
INDEX_DIR = DATA_DIR / "index"
DOCS_DIR = REPO_ROOT / "docs"

SB_BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"

# Pitch, per StatsBomb spec. Phase 0 verifies this instead of trusting it.
PITCH_X = 120.0
PITCH_Y = 80.0
GOAL_CENTRE = (120.0, 40.0)
GOAL_POSTS = ((120.0, 36.0), (120.0, 44.0))

# Dense retrieval. 384-dim, CPU-only, ~90 MB download on first use.
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384

for _d in (RAW_DIR, INDEX_DIR, DOCS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
