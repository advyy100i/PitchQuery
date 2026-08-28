"""Paths and constants. Nothing here costs money or needs a key."""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Load .env HERE, before any os.getenv below runs. core/db.py also calls
# load_dotenv, but it imports this module first — so anything set in .env and
# read at import time in this file was previously ignored, and PITCHQUERY_DATA
# silently fell back to its default. Real environment variables still win:
# load_dotenv does not override what the process was started with, which is
# what a hosted deployment relies on.
try:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
except ImportError:      # dotenv is optional at runtime
    pass

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

# How far outside the pitch a coordinate is still allowed to be.
#
# Not slop: StatsBomb genuinely records the ball marginally over the line when a
# shot is struck from on top of it, and this corpus contains two such rows
# (x = 120.5 and x = 120.1, both shots from the byline). A validity check that
# fails on correct data is worse than no check, because it teaches you to skip
# the output. Two units is enough for that and far too little to hide the
# failure these bounds exist to catch — a coordinate parsed on the wrong scale,
# or a freeze frame left in the defending team's reference frame.
PITCH_MARGIN = 2.0
GOAL_CENTRE = (120.0, 40.0)
GOAL_POSTS = ((120.0, 36.0), (120.0, 44.0))

# Dense retrieval. 384-dim, CPU-only, ~90 MB download on first use.
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384

for _d in (RAW_DIR, INDEX_DIR, DOCS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
